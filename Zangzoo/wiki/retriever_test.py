import pandas as pd
from retriever import get_relevant, _extract_keyphrases

# retriever_test.py
# 테스트 스크립트: test.csv에서 질문 샘플을 뽑아 다양한 retrieval method 결과 확인

def main():
    # 1) test.csv 읽기
    try:
        df = pd.read_csv("test.csv", dtype=str)
    except FileNotFoundError:
        raise FileNotFoundError("test.csv 파일이 존재하는지 확인하세요.")

    # 2) 확인할 샘플 개수 지정
    N = 5
    # 특정 인덱스 리스트로 뽑으려면 아래 주석 해제
    # sample_ids = [0, 10, 20]
    # df_sample = df.loc[sample_ids].reset_index(drop=True)
    df_sample = df.sample(N, random_state=42).reset_index(drop=True)

    for i, row in df_sample.iterrows():
        sample_id = row.get("ID", i)
        question = row.get("question", "")
        print(f"\n=== Sample {i} (ID={sample_id}) ===")
        print(f"Question: {question}\n")

        # 3) 핵심 키프레이즈 추출
        key_phrases = _extract_keyphrases(question, top_n=2)
        print(f"Keyphrases extracted: {key_phrases}\n")

        # 4) 각 retrieval method별 candidate 확인
        methods = ["bm25", "sbert", "wiki", "wiki_sbert", "all"]
        for method in methods:
            # wiki 관련 method는 key_phrases가 없으면 스킵
            if method in ("wiki", "wiki_sbert") and not key_phrases:
                print(f"-- Method: {method} (skipped: no key_phrases)")
                continue

            print(f"-- Method: {method}")
            try:
                cands = get_relevant(question, k=2, method=method)
            except IndexError:
                print(f"   [{method}] skipped due to empty candidate list.")
                continue
            except Exception as e:
                print(f"   [{method}] error: {e}")
                continue

            if cands:
                for j, sent in enumerate(cands, 1):
                    print(f"   [{method}] candidate {j}: {sent}")
            else:
                print(f"   [{method}] no candidates found.")
            print()

    print("\n=== Retrieval test completed ===")

if __name__ == "__main__":
    main()
