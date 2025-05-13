import argparse, os, gc
import pandas as pd
from data_loader import load_data
from model_runner import load_model, predict_batch_answers
from retriever import get_relevant
import torch, warnings, logging
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import logging
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- Args ------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", default="test.csv")
parser.add_argument("--output_csv", default="submission.csv")
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--dyn_batch", type=int, default=2)
parser.add_argument("--max_new_tokens", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--sample_size", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ---------------- Settings ------------- #
SAVE_EVERY = 500       # 500행마다 저장
RESUME = True          # 기존 파일 있으면 이어쓰기

# ---------------- Model Init & Warm-up ------------- #
tokenizer, model = load_model()
get_relevant("warm-up", k=1)
logging.basicConfig(
    format="%(asctime)s %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)

logging.info("🔧 Llama warm-up")
logging.info("🔧 SBERT warm-up")

# ---------------- Process Function ------------- #
def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    # Ensure result columns exist
    for col in ["raw_input", "raw_output", "answer"]:
        if col not in chunk.columns:
            # 새로 만들 때도 object dtype 로스트링("") 으로 초기화
            chunk[col] = pd.Series([""] * len(chunk), dtype="object")
        else:
            # 이미 있으면 무조건 object 로 바꿔 줌
            chunk[col] = chunk[col].astype("object")

    bs = args.batch_size
    for i in range(0, len(chunk), bs):
        sub = chunk.iloc[i : i + bs]
        p, r, a = predict_batch_answers(
            tokenizer, model,
            sub["context"].tolist(),
            sub["question"].tolist(),
            sub["choices"].tolist(),
            max_new_tokens=args.max_new_tokens,
            dyn_bs=args.dyn_batch,
        )
        chunk.loc[sub.index, ["raw_input", "raw_output", "answer"]] = list(zip(p, r, a))
    return chunk

# ---------------- Inference & Checkpoint ------------- #
def run_inference():
    df = load_data(args.input_csv, sample_size=args.sample_size, seed=args.seed)

    # Resume: skip only rows with actual outputs
    if RESUME and os.path.exists(args.output_csv):
        existing = pd.read_csv(args.output_csv)
        done_ids = existing.loc[existing['raw_input'] != '알 수 없음', 'ID']
        df = df[~df['ID'].isin(done_ids)]
        print(f"⏩  Resume mode: {len(done_ids)} rows already done")

    total = len(df)
    print(f"🔸 Remaining samples: {total}")

    buffers, written = [], 0
    header_written = os.path.exists(args.output_csv)

    # Process in chunks of 100
    chunks = [df.iloc[i : i + 100] for i in range(0, total, 100)]
    for i, chunk in enumerate(chunks, start=1):
        logging.info(f"--- Processing chunk {i}/{len(chunks)} ---")
        result = process_chunk(chunk.copy())
        buffers.append(result)
        written += len(result)
        logging.info(f"🔄 Remaining samples: {total - written}")

        # Save checkpoints every SAVE_EVERY rows
        if written % SAVE_EVERY == 0:
            _flush(buffers, header_written)
            header_written = True
            buffers.clear()
            print(f"💾  {written} rows saved → {args.output_csv}")

            # Create additional checkpoint with template
            done = pd.read_csv(args.output_csv)
            tmpl = pd.read_csv("sample_submission.csv")
            for col in ["raw_input", "raw_output", "answer"]:
                tmpl[col] = "알 수 없음"
            tmpl.set_index("ID", inplace=True)
            done_idx = done.set_index("ID")[ ["raw_input","raw_output","answer"] ]
            tmpl.update(done_idx)
            tmpl.reset_index(inplace=True)
            ckpt = f"submission_checkpoint_{written}.csv"
            tmpl.to_csv(ckpt, index=False, encoding="utf-8-sig")
            print(f"🔖  Checkpoint saved: {ckpt}")

            # Clear cache
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
            else: torch.cuda.empty_cache()
            gc.collect()

    # Final flush
    if buffers:
        _flush(buffers, header_written)
        written += sum(len(b) for b in buffers)
        print(f"💾  {written} rows saved (final)")
        print(f"🔄  Remaining samples: {total - written}")

        # Final checkpoint
        done = pd.read_csv(args.output_csv)
        tmpl = pd.read_csv("sample_submission.csv")
        for col in ["raw_input", "raw_output", "answer"]:
            tmpl[col] = "알 수 없음"
        tmpl.set_index("ID", inplace=True)
        done_idx = done.set_index("ID")[ ["raw_input","raw_output","answer"] ]
        tmpl.update(done_idx)
        tmpl.reset_index(inplace=True)
        ckpt = f"submission_checkpoint_{written}.csv"
        tmpl.to_csv(ckpt, index=False, encoding="utf-8-sig")
        print(f"🔖  Checkpoint saved: {ckpt}")
        if torch.backends.mps.is_available(): torch.mps.empty_cache()
        else: torch.cuda.empty_cache()
        gc.collect()

# ---------------- Flush Function ------------- #
def _flush(dfs, header_written: bool):
    df_all = pd.concat(dfs).sort_index()
    df_all = df_all.drop(columns=["context", "question", "choices"], errors="ignore")

    tmpl = pd.read_csv("sample_submission.csv")
    for col in ["raw_input", "raw_output", "answer"]:
        tmpl[col] = "알 수 없음"

    if header_written and os.path.exists(args.output_csv):
        prev = pd.read_csv(args.output_csv).set_index("ID")
        tmpl = tmpl.set_index("ID")
        tmpl.update(prev)
        tmpl = tmpl.reset_index()

    upd = df_all.set_index("ID")[ ["raw_input","raw_output","answer"] ]
    tmpl = tmpl.set_index("ID")
    tmpl.update(upd)
    tmpl = tmpl.reset_index()
    tmpl.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    run_inference()