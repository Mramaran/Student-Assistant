from langchain_core.documents import Document
from verification.confidence import compute_confidence

def test_robustness():
    # 1. Test with Document objects (The problematic case reported by user)
    docs = [
        Document(page_content="The HOD of CSE is Dr. John Doe.", metadata={"source": "dept.txt"}),
        Document(page_content="Attendance requirement is 75%.", metadata={"source": "rules.txt"})
    ]
    ans = "The head of department for computer science is Dr. John Doe."
    conf = compute_confidence(ans, docs)
    print(f"✅ Document test passed: confidence={conf}")

    # 2. Test with dictionaries
    dicts = [{"text": "Fee is 50000."}, {"text": "Scholarships available."}]
    conf_dict = compute_confidence("Fee is 50000", dicts)
    print(f"✅ Dict test passed: confidence={conf_dict}")

    # 3. Test with strings
    strs = ["Hostel fee: 10000", "Mess fee: 5000"]
    conf_str = compute_confidence("Hostel fee is 10000", strs)
    print(f"✅ String test passed: confidence={conf_str}")

if __name__ == "__main__":
    try:
        test_robustness()
        print("\n🚀 ALL ROBUSTNESS TESTS PASSED.")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
