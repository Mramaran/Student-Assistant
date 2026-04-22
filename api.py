"""
api.py — REST API for RAG Student Assistant
=============================================
FastAPI backend with JWT authentication and Role-Based Access Control (RBAC).

ROLES:
  - student : Can request matches and ask RAG questions.
  - tutor   : Can view assigned sessions and analytics.
  - admin   : System-wide access + analytics dashboard.

ENDPOINTS:
  POST /token      : Login and get JWT.
  POST /match      : Trigger intelligent tutor matching.
  POST /rag        : Ask a question to the RAG pipeline.
  GET  /me         : Current user profile.
  GET  /admin/stats: Real-time ROI and system stats.
"""

import os
from datetime import datetime, timedelta
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from dotenv import load_dotenv

# Import Core Logic
from core.pipeline import run_fast_path, classify_tutoring_intent
from services.personalization import build_student_profile, get_tutor_pool
from services.matching_engine import match_tutors
from core.feature_flags import parse as parse_feature_flags
from feedback import analytics, impact

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET", "super-secret-key-change-it")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# pbkdf2_sha256 is pure-Python (uses hashlib from stdlib), avoiding the
# passlib↔bcrypt version-compat breakage on passlib 1.7 + bcrypt>=4.
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI(title="RAG Assistant API", version="2.0.0")

# ── Mock Database (Demo) ──────────────────────────────────────────────────────
# In production, replace with real database (PostgreSQL/MongoDB)
USERS_DB = {
    "student@cit.edu": {
        "username": "student@cit.edu",
        "hashed_password": pwd_context.hash("cit123"),
        "role": "student",
        "student_id": "S12345",
    },
    "tutor@cit.edu": {
        "username": "tutor@cit.edu",
        "hashed_password": pwd_context.hash("tutor123"),
        "role": "tutor",
    },
    "admin@cit.edu": {
        "username": "admin@cit.edu",
        "hashed_password": pwd_context.hash("admin123"),
        "role": "admin",
    },
}

# ── Schemas ───────────────────────────────────────────────────────────────────
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None
    role: str | None = None

class User(BaseModel):
    username: str
    role: str
    student_id: str | None = None

class QueryRequest(BaseModel):
    query: str
    context: dict | None = None

# ── Auth Utilities ────────────────────────────────────────────────────────────
def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, role=role)
    except JWTError:
        raise credentials_exception
    
    user = USERS_DB.get(token_data.username)
    if user is None:
        raise credentials_exception
    return User(**user)

def require_role(roles: list[str]):
    async def role_checker(current_user: Annotated[User, Depends(get_current_user)]):
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Operation not permitted for your role"
            )
        return current_user
    return role_checker

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = USERS_DB.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/me", response_model=User)
async def read_users_me(current_user: Annotated[User, Depends(get_current_user)]):
    return current_user

async def _extract_feature_flags(
    x_feature_flags: Annotated[str | None, Header()] = None,
) -> dict[str, bool]:
    """FastAPI dependency — parse the ``X-Feature-Flags`` header once."""
    return parse_feature_flags(x_feature_flags)


@app.post("/match")
async def match_tutor_endpoint(
    req: QueryRequest,
    current_user: Annotated[User, Depends(require_role(["student", "admin"]))],
    feature_flags: Annotated[dict[str, bool], Depends(_extract_feature_flags)],
):
    """Secure matching endpoint with RBAC and runtime feature flags."""
    student_ctx = req.context or {}
    student_ctx["student_id"] = current_user.student_id or "ANON"

    student = build_student_profile(req.query, student_ctx)
    tutors = get_tutor_pool()

    # FLAG: competency_graph — consumed inside match_tutors.
    # FLAG: triage_fast_pass — consumed by the pipeline; surfaced here for
    # completeness so clients can inspect the resolved flag set via debug
    # endpoints or logs when they need to.
    result = match_tutors(student, tutors, feature_flags=feature_flags)
    result["feature_flags"] = dict(feature_flags)
    return result

@app.post("/rag")
async def rag_query_endpoint(
    req: QueryRequest,
    current_user: Annotated[User, Depends(require_role(["student", "tutor", "admin"]))]
):
    """Standard RAG access for all authenticated users."""
    if classify_tutoring_intent(req.query):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tutoring query detected. Please use /match endpoint."
        )
    
    res = run_fast_path(req.query)
    return res

@app.get("/admin/stats")
async def get_system_stats(
    current_user: Annotated[User, Depends(require_role(["admin"]))]
):
    """Admin-only system overview."""
    return {
        "summary": analytics.summary(),
        "impact": impact.generate_before_after_summary(),
        "engagement": analytics.user_engagement_trend()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
