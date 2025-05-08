# main.py
import argparse, os, gc, pandas as pd
from joblib import Parallel, delayed
from data_loader   import load_data         # 그대로
from model_runner  import load_model, predict_batch_answers
from retriever     import get_relevant      # 필요 시 warm-up
import torch, warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
RESUME     = True                    # 중간 CSV가 있으면 이어서
# ----- 모델 1회 로드 ----- #
tokenizer, model = load_model()   # Llama
get_relevant("warm-up", k=1)      # SBERT warm-up
# ------------------------ #

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

def run_inference():
    df = load_data(args.input_csv, sample_size=args.sample_size, seed=args.seed)

    if RESUME and os.path.exists(args.output_csv):
        done_df = pd.read_csv(args.output_csv)
        df      = df[~df["ID"].isin(done_df["ID"])]
        print(f"⏩  Resume mode: {len(done_df)} rows already done")

    n = len(df)
    print(f"🔸 Remaining samples: {n}")

    chunk_size = 100
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, n, chunk_size)]

    buffered, total_written = [], 0
    header_written = os.path.exists(args.output_csv)

    for res in Parallel(n_jobs=args.num_workers,
                        backend="threading")(
            delayed(process_chunk)(c.copy()) for c in chunks):
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
    run_inference()