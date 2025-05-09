import argparse, os, gc, pandas as pd
from joblib import Parallel, delayed
from data_loader import load_data         # 그대로
from model_runner import load_model, predict_batch_answers
import torch, warnings
warnings.filterwarnings("ignore", category=UserWarning)
print(load_data.__module__)
import inspect
print(inspect.getfile(load_data))



# ---------------- Args ------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", default="test.csv")
parser.add_argument("--output_csv", default="submission.csv")
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--dyn_batch",  type=int, default=2)
parser.add_argument("--max_new_tokens", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--sample_size", type=int, default=None)   # None이면 전체 사용
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
# ----------------------------------- #
# main.py  (import 바로 아래)
SAVE_EVERY = 500                     # ✅ 500행마다 저장
RESUME = True                    # 중간 CSV가 있으면 이어서
# ----- 모델 1회 로드 ----- #
tokenizer, model = load_model()  # Llama
# ------------------------ #
# DataFrame의 일부 chunk를 받아서 한 번에 배치 단위로 inference 수행
def process_chunk(chunk):
    """chunk(DataFrame) 단위 infer"""
    bs = args.batch_size
    for i in range(0, len(chunk), bs):
        sub = chunk.iloc[i:i+bs]
        p,r,a = predict_batch_answers(
            tokenizer, model,
            sub["context"].tolist(),
            sub["question"].tolist(),
            sub["choices"].tolist(),
            max_new_tokens=args.max_new_tokens,
            dyn_bs=args.dyn_batch,
        )
        # Explicitly cast columns to str to avoid dtype warnings during assignment
        chunk["raw_input"] = chunk["raw_input"].astype(str)
        chunk["raw_output"] = chunk["raw_output"].astype(str)
        chunk["answer"] = chunk["answer"].astype(str)
        chunk.loc[sub.index, ["raw_input","raw_output","answer"]] = list(zip(p,r,a))
    torch.mps.empty_cache(); gc.collect()
    return chunk

def run_inference(input_csv: str, output_csv: str, batch_size: int):
    df = load_data(input_csv)

    if RESUME and os.path.exists(args.output_csv):
        done_df = pd.read_csv(args.output_csv)
        df = df[~df["ID"].isin(done_df["ID"])]
        print(f"⏩  Resume mode: {len(done_df)} rows already done")

    n = len(df)
    print(f"🔸 Remaining samples: {n}")

    chunk_size = 100
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, n, chunk_size)]

    buffered, total_written = [], 0
    header_written = os.path.exists(args.output_csv)

     # 🔽 병렬처리 제거하고 일반 for 루프로 변경
    for chunk in chunks:
        res = process_chunk(chunk.copy())
        buffered.append(res)

        if sum(len(x) for x in buffered) >= SAVE_EVERY:
            _flush(buffered, header_written)
            total_written += sum(len(x) for x in buffered)
            buffered.clear()
            header_written = True
            print(f"💾  {total_written} rows saved → {args.output_csv}")

    if buffered:
        _flush(buffered, header_written)
        total_written += sum(len(x) for x in buffered)
        print(f"💾  {total_written} rows saved (final)")
        
# 결과 데이터프레임들 하나로 모음. sample_submission과 merge하여 누락값 보완 
def _flush(dfs, header_written):
    out_df = pd.concat(dfs).sort_index()
    out_df = out_df.drop(columns=["context", "question", "choices"], errors="ignore")
    # Load sample_submission and merge
    sample_df = pd.read_csv("sample_submission.csv")
    merged_df = sample_df.copy()
    merged_df = merged_df.merge(out_df, on="ID", how="left", suffixes=("", "_new"))

    for col in ["answer", "raw_input", "raw_output"]:
        if f"{col}_new" in merged_df.columns:
            merged_df[col] = merged_df[f"{col}_new"].combine_first(merged_df[col])
            merged_df.drop(columns=[f"{col}_new"], inplace=True)

    out_df = merged_df
    mode   = "a" if header_written else "w"
    out_df.to_csv(args.output_csv, mode=mode, index=False, encoding="utf-8-sig",
                  header=not header_written)

if __name__ == "__main__":
    run_inference("test.csv", "baseline_submission.csv", batch_size=args.batch_size)