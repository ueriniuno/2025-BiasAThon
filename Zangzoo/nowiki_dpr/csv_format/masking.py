import pandas as pd

# # 1. 데이터 불러오기
# df = pd.read_csv("test.csv")

# # 2. 시드 고정하여 2000개 샘플링
# sampled_df = df.sample(n=2000, random_state=42)

# # 3. ID 기준 정렬
# sampled_df = sampled_df.sort_values(by="ID").reset_index(drop=True)

# # 4. 저장
# sampled_df.to_csv("test_sampled_seed42_2000.csv", index=False, encoding="utf-8-sig")
# print("✅ 정렬 후 저장 완료: test_sampled_seed42_2000.csv")
df = pd.read_csv('rag_bm25_k2_2000_real.csv')
df = df.iloc[:, 1:]
df.to_csv('rag_bm25_k2_2000_real_updated.csv', index=False)