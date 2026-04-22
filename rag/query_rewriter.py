import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq()

def rewrite_query(query: str) -> str:
    """
    Rewrites a vague or messy user query into a clean, searchable academic query targeting college policies.
    """
    # If query is already long and seemingly well-formed, just return it.
    if len(query.split()) > 8:
        return query

    system_prompt = """You are a query rewriting agent for a college RAG system.
Your job is to take raw, messy, or short student questions and rewrite them into clear, explicit queries optimized for semantic search against policy documents (attendance, exams, hostel, fees).

Rules:
1. Identify the core academic intent.
2. Add necessary academic context (e.g. "What is the policy on...").
3. DO NOT answer the question. Only output the revised question.
4. Keep the rewritten question under 20 words.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Original query: {query}\n\nRewritten query:"}
    ]

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.0,
            max_tokens=60
        )
        rewritten = response.choices[0].message.content.strip()
        # Clean up quotes
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1]
        return rewritten
    except Exception as e:
        print(f"Error in query_rewriter: {e}")
        return query
