import pandas as pd

# 파일 경로 설정
merged_path = "Merged_Submission__00000_14003_.csv"
baseline_path = "baseline_submission.csv"
output_path = "final_merged_submission.csv" 

# CSV 로드
merged = pd.read_csv(merged_path)
baseline = pd.read_csv(baseline_path)

# TEST_14004 이후 데이터만 baseline에서 선택
baseline_tail = baseline[baseline["ID"] >= "TEST_14004"]

# 병합
final = pd.concat([merged, baseline_tail], ignore_index=True)

# 저장
final.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"✅ 저장 완료: {output_path}")
