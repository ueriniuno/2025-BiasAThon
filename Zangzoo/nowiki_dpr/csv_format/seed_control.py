# import pandas as pd

# # 1. 데이터 불러오기
# df = pd.read_csv("test.csv")

# # 2. 시드 고정하여 2000개 샘플링
# sampled_df = df.sample(n=2000, random_state=42)

# # 3. 저장
# sampled_df.to_csv("test_sampled_seed42_2000_real.csv", index=False, encoding="utf-8-sig")
# print("✅ 저장 완료: test_sampled_seed42_2000.csv")
import pandas as pd

# 1) 미리 섞인 2,000개 파일 로드
df = pd.read_csv("test_sampled_seed42_2000_real.csv")

# 2) 0-based index 1500~1999 슬라이스
df_remain = df.iloc[1500:2000].reset_index(drop=True)

# 3) 남은 500개만 별도 CSV로 저장
df_remain.to_csv("test_remain_1500_2000.csv", index=False)