#main.py
import argparse
from data_loader import load_data
from model_runner import load_model, predict_batch_answers
import pandas as pd
from tqdm import tqdm

def run_inference(input_csv: str, output_csv: str, batch_size: int):
    data = load_data(input_csv)
    tokenizer, model = load_model()

    data["raw_input"] = data["raw_input"].astype("string") #ㅊㅇ
    data["raw_output"] = data["raw_output"].astype("string") #ㅊㅇ
    data["answer"] = data["answer"].astype("string") #ㅊㅇ

    for i in tqdm(range(0, len(data), batch_size), desc="Processing"):
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
            data[["ID", "raw_input", "raw_output", "answer"]].to_csv(
                f"submission_checkpoint_{str(i)}.csv",
                index=False,
                encoding="utf-8-sig"
            )

    data[["ID", "raw_input", "raw_output", "answer"]].to_csv(output_csv, index=False, encoding="utf-8-sig")
    print("🎉 최종 제출 파일 저장 완료:", output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="배치 사이즈 (기본값=1)")
    args = parser.parse_args()

    run_inference("test.csv", "baseline_submission.csv", batch_size=args.batch_size)
