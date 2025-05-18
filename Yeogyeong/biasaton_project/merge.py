import pandas as pd
import os

# 전체 테스트 데이터 로딩
test_df = pd.read_csv("data/test.csv")  # 전체 약 29만건
pred_df = pd.read_csv("results_llama_cpp/submission.csv")  # 2000개 예측 결과

# ID 기준 병합 (왼쪽 기준: 전체 test 데이터)
merged = pd.merge(test_df[["ID"]], pred_df, on="ID", how="left")

# 누락된 raw_input, raw_output, answer는 알 수 없음으로 채움
merged["raw_input"] = merged["raw_input"].fillna("알 수 없음")
merged["raw_output"] = merged["raw_output"].fillna("알 수 없음")
merged["answer"] = merged["answer"].fillna("알 수 없음")  # 판단 불가 시 C로 고정

# 저장 디렉토리 없으면 생성
os.makedirs("submission", exist_ok=True)

# 최종 저장
merged.to_csv("submission/full_submission.csv", index=False, encoding="utf-8-sig")

print("✅ full_submission.csv 생성 완료!")
