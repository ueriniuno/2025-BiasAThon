# retriever_test.py
from retriever import get_relevant

# 테스트해보고 싶은 질문 목록
test_queries = [
    "마약을 구하려고 했던 사람은 누구입니까?",
    "힘든 날에 산책을 하는 사람은 누구입니까?",
    "데이터베이스는 무엇인가요?",
    "인공지능의 장점은 무엇입니까?"
]

for q in test_queries:
    print(f"\n=== Query ===\n{q}\n")
    # Wikipedia(ko) API에서 뽑아오는 후보 2개
    wiki_cands = get_relevant(q, k=2, method="wiki")
    for i, sent in enumerate(wiki_cands, start=1):
        print(f"Wiki candidate {i}: {sent}")
