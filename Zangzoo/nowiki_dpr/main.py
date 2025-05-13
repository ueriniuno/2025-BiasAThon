# main.py

import argparse, os, gc
import pandas as pd
from data_loader import load_data
from model_runner import load_model, predict_batch_answers
import torch, warnings, logging

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

# ---------------- Args ------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--input_csv",       default="test.csv")
parser.add_argument("--sample_csv",      default="sample_submission.csv")
parser.add_argument("--output_csv",      default="submission.csv")
parser.add_argument("--batch_size",      type=int, default=100)
parser.add_argument("--dyn_batch",       type=int, default=4)
parser.add_argument("--max_new_tokens",  type=int, default=32)
parser.add_argument("--num_workers",     type=int, default=2)
parser.add_argument("--sample_size",     type=int, default=2000)
parser.add_argument("--seed",            type=int, default=42)
# ── 여기부터 새 옵션 추가 ── #
parser.add_argument("--use_expansion",   action="store_true", help="T5 기반 쿼리 확장")
parser.add_argument("--use_rag",         action="store_true", help="Hybrid DPR+BM25+CE+MMR RAG")
parser.add_argument("--use_cot",         action="store_true", help="Chain-of-Thought (hidden)")
parser.add_argument("--use_debias",      action="store_true", help="Self-Debias prompting")
parser.add_argument("--use_few_shot",    action="store_true", help="Few-Shot 예시 삽입")
parser.add_argument("--few_shot_path",   type=str, default=None, help="Few-Shot JSON 파일 경로")
# ──────────────────────────────── #
args = parser.parse_args()

# ---------------- Settings ------------- #
SAVE_EVERY = 500       # 500행마다 저장
RESUME     = True      # 기존 파일 있으면 이어쓰기

# ---------------- Model Init & Warm-up ------------- #
tokenizer, model = load_model()
logging.info("🔧 Llama warm-up complete")

# ---------------- Few-Shot Loader ------------- #
def load_few_shot(path: str):
    import json
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [(ex['q'], ex['a']) for ex in data]

few_shot = None
if args.use_few_shot and args.few_shot_path:
    few_shot = load_few_shot(args.few_shot_path)

# ---------------- Process Function ------------- #
def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    # Ensure result columns exist
    for col in ["raw_input", "raw_output", "answer"]:
        if col not in chunk.columns:
            chunk[col] = ""
        else:
            chunk[col] = chunk[col].astype("object")

    bs = args.batch_size
    for i in range(0, len(chunk), bs):
        sub = chunk.iloc[i : i + bs]
        prompts, raws, ans = predict_batch_answers(
            tokenizer=tokenizer,
            model=model,
            contexts=sub.get("context", [""]*len(sub)).tolist(),
            questions=sub["question"].tolist(),
            choices_list=sub["choices"].tolist(),
            few_shot=few_shot,
            use_expansion=args.use_expansion,
            use_rag=args.use_rag,
            use_cot=args.use_cot,
            use_debias=args.use_debias,
            max_new_tokens=args.max_new_tokens,
            dyn_bs=args.dyn_batch
        )
        chunk.loc[sub.index, ["raw_input", "raw_output", "answer"]] = list(zip(prompts, raws, ans))
    return chunk

# ---------------- Inference & Checkpoint ------------- #
def _flush(dfs, header_written: bool):
    df_all = pd.concat(dfs).sort_index()
    df_all = df_all.drop(columns=["context", "question", "choices"], errors="ignore")

    tmpl = pd.read_csv(args.sample_csv)
    for col in ["raw_input", "raw_output", "answer"]:
        tmpl[col] = "알 수 없음"

    if header_written and os.path.exists(args.output_csv):
        prev = pd.read_csv(args.output_csv).set_index("ID")
        tmpl = tmpl.set_index("ID")
        tmpl.update(prev)
        tmpl = tmpl.reset_index()

    upd = df_all.set_index("ID")[["raw_input","raw_output","answer"]]
    tmpl = tmpl.set_index("ID")
    tmpl.update(upd)
    tmpl.reset_index().to_csv(args.output_csv, index=False, encoding="utf-8-sig")

def run_inference():
    df = load_data(args.input_csv, sample_size=args.sample_size, seed=args.seed)

    # Resume
    if RESUME and os.path.exists(args.output_csv):
        prev = pd.read_csv(args.output_csv)
        done = prev.loc[prev['raw_input'] != "", 'ID']
        df = df[~df['ID'].isin(done)]
        logging.info(f"⏩ Resume mode: skipped {len(done)} rows")

    total = len(df)
    logging.info(f"🔸 Remaining samples: {total}")

    buffers, written = [], 0
    header_written = os.path.exists(args.output_csv)

    # Chunk processing
    chunks = [df.iloc[i : i + 100] for i in range(0, total, 100)]
    for idx, chunk in enumerate(chunks, 1):
        logging.info(f"--- Chunk {idx}/{len(chunks)} ---")
        result = process_chunk(chunk.copy())
        buffers.append(result)
        written += len(result)
        logging.info(f"🔄 Remaining: {total - written}")

        if written % SAVE_EVERY == 0:
            _flush(buffers, header_written)
            header_written = True
            buffers.clear()
            logging.info(f"💾 Saved {written} rows → {args.output_csv}")
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
            else: torch.cuda.empty_cache()
            gc.collect()

    # Final flush
    if buffers:
        _flush(buffers, header_written)
        logging.info(f"💾 Final save: {written + sum(len(b) for b in buffers)} rows")

if __name__ == "__main__":
    run_inference()
