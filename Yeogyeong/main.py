# main.py
import argparse
import gc
import torch
from tqdm import tqdm 
import pandas as pd

from data_loader import load_data
from model_runner import load_model, predict_batch_answers

# ─────────────────────────  기본 설정  ─────────────────────────
torch.backends.cudnn.benchmark = True          # MPS/CPU 속도 튜닝
torch.manual_seed(42)

# ─────────────────────────  추론 루프  ─────────────────────────
from typing import Optional

def run_inference(
    input_csv: str,
    output_csv: str,
    batch_size: int = 100,
    resume_from_id: Optional[str] = None,
    dyn_batch: int = 2,
    max_new_tokens: int = 32,
):
    # 1) 데이터 · 모델 로드
    data = load_data(input_csv)
    tokenizer, model = load_model()

    # 2) 열 타입 고정 (경고 방지)
    for col in ["raw_input", "raw_output", "answer"]:
        data[col] = data[col].astype("string")

    # 3) 재시작 위치 결정
    if resume_from_id:
        idx_list = data.index[data["ID"] == resume_from_id].tolist()
        if not idx_list:
            raise ValueError(f"{resume_from_id} not found in the dataset.")
        start_idx = idx_list[0] + 1
        print(f"🔄  {resume_from_id} 이후 index {start_idx}부터 재시작")
    else:
        start_idx = 0

    # 4) 배치 추론
    for i in tqdm(range(start_idx, len(data), batch_size), desc="Processing"):
        batch = data.iloc[i : i + batch_size]

        prompts, raws, answers = predict_batch_answers(
            tokenizer,
            model,
            batch["context"].tolist(),
            batch["question"].tolist(),
            batch["choices"].tolist(),
            max_new_tokens=max_new_tokens,
            dyn_bs=dyn_batch,
        )

        data.loc[batch.index, "raw_input"]  = prompts
        data.loc[batch.index, "raw_output"] = raws
        data.loc[batch.index, "answer"]     = answers

        # 5) 중간 저장
        if i % 500 == 0 and i > 0:
            path = f"submission_checkpoint_{i}.csv"
            tqdm.write(f"💾  {i}/{len(data)} 저장 → {path}")
            data[["ID", "raw_input", "raw_output", "answer"]].to_csv(
                path, index=False, encoding="utf-8-sig"
            )
            torch.mps.empty_cache(); gc.collect()

    # 6) 최종 저장
    data[["ID", "raw_input", "raw_output", "answer"]].to_csv(
        output_csv, index=False, encoding="utf-8-sig"
    )
    print(f"🎉  최종 제출 파일 저장 완료: {output_csv}")

# ─────────────────────────  CLI  ─────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=100, help="CSV 배치 크기")
    parser.add_argument("--dyn_batch",  type=int, default=2,   help="모델 입력 동적 배치(1‑2 권장)")
    parser.add_argument("--resume_from_id", type=str, default=None, help="재시작할 ID")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="generate 토큰 길이")
    args = parser.parse_args()

    run_inference(
        "Yeogyeong/test.csv",
        "Yeogyeong/baseline_submission.csv",
        batch_size=args.batch_size,
        resume_from_id=args.resume_from_id,
        dyn_batch=args.dyn_batch,
        max_new_tokens=args.max_new_tokens,
    )
