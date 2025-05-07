import argparse
from data_loader import load_data
from model_runner import load_model, predict_batch_answers
import pandas as pd
from tqdm import tqdm

def run_inference(input_csv: str, output_csv: str, batch_size: int, resume_from_id: str = None):
    data = load_data(input_csv)
    tokenizer, model = load_model()

    # 🔧 열 타입 고정 (경고 방지용)
    data["raw_input"] = data["raw_input"].astype("string")
    data["raw_output"] = data["raw_output"].astype("string")
    data["answer"] = data["answer"].astype("string")

    # 🔁 resume_from_id가 지정된 경우 해당 ID 이후부터 처리
    if resume_from_id:
        try:
            start_idx = data.index[data["ID"] == resume_from_id].tolist()[0] + 1
        except IndexError:
            raise ValueError(f"{resume_from_id} not found in the dataset.")
        print(f"🔄 {resume_from_id} 이후 index {start_idx}부터 재시작합니다.")
    else:
        start_idx = 0

    for i in tqdm(range(start_idx, len(data), batch_size), desc="Processing"):
        batch = data.iloc[i:i+batch_size]
        prompts, raw_outputs, answers = predict_batch_answers(
            tokenizer, model,
            batch["context"].tolist(),
            batch["question"].tolist(),
            batch["choices"].tolist()
        )

        data.loc[batch.index, "raw_input"] = prompts
        data.loc[batch.index, "raw_output"] = raw_outputs
        data.loc[batch.index, "answer"] = answers

        if i % 500 == 0 and i > 0:
            tqdm.write(f"{i}/{len(data)}까지 저장 완료 — 중간 저장")
            data[["ID", "raw_input", "raw_output", "answer"]].to_csv(
                f"submission_checkpoint_{str(i)}.csv",
                index=False,
                encoding="utf-8-sig"
            )

    data[["ID", "raw_input", "raw_output", "answer"]].to_csv(output_csv, index=False, encoding="utf-8-sig")
    print("🎉 최종 제출 파일 저장 완료:", output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="배치 사이즈 (기본값=1)")
    parser.add_argument("--resume_from_id", type=str, default=None, help="재시작할 ID (예: TEST_11003)")
    args = parser.parse_args()

    run_inference("test.csv", "baseline_submission.csv", batch_size=args.batch_size, resume_from_id=args.resume_from_id)
