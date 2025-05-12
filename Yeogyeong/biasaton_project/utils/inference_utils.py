# import torch
# import pandas as pd
# from tqdm import tqdm
# import gc
# import time
# from utils.answer_extraction import extract_answer
#
# @torch.no_grad()
# def batch_tokenize(batch_texts, tokenizer, model):
#     """Tokenize a batch of texts efficiently with warning for overly long inputs"""
#     max_length = 512
#     encodings = tokenizer(batch_texts, add_special_tokens=True, truncation=False)
#
#     for i, input_ids in enumerate(encodings["input_ids"]):
#         length = len(input_ids)
#         if length > max_length:
#             overflow = length - max_length
#             print(f"⚠️ Warning: Text {i} is {overflow} tokens too long ({length} > {max_length}). It will be truncated.")
#
#     return tokenizer(
#         batch_texts,
#         padding=True,
#         return_tensors="pt",
#         truncation=True,
#         max_length=max_length
#     ).to(model.device)
#
# @torch.no_grad()
# def process_batch(batch_prompts, tokenizer, model):
#     """Process a batch of prompts with optimized inference"""
#     inputs = batch_tokenize(batch_prompts, tokenizer, model)
#
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=16,
#         do_sample=True,
#         temperature=0.2,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.pad_token_id,
#         use_cache=True,
#     )
#
#     results = []
#     for output in outputs:
#         result_text = tokenizer.decode(output, skip_special_tokens=True)
#         raw_answer, answer = extract_answer(result_text)
#         results.append((raw_answer, answer))
#
#     return results
#
# def predict_with_dynamic_batching(df, model, tokenizer, args):
#     """Process data with dynamic batch sizing and performance monitoring"""
#     all_results = []
#     total = len(df)
#     current_batch_size = args.batch_size
#     start_idx = 0
#
#     with tqdm(total=total, desc="Processing") as pbar:
#         while start_idx < total:
#             end_idx = min(start_idx + current_batch_size, total)
#             batch = df.iloc[start_idx:end_idx]
#             batch_size = end_idx - start_idx
#             batch_prompts = batch['prompt'].tolist()
#
#             try:
#                 start_time = time.time()
#                 batch_results = process_batch(batch_prompts, tokenizer, model)
#                 end_time = time.time()
#
#                 time_per_sample = (end_time - start_time) / batch_size
#
#                 if time_per_sample < 0.5 and current_batch_size < 32:
#                     current_batch_size = min(current_batch_size + 2, 32)
#                 elif time_per_sample > 2.0 and current_batch_size > 2:
#                     current_batch_size = max(current_batch_size - 2, 2)
#
#                 for i, (raw_output, answer) in enumerate(batch_results):
#                     row_idx = start_idx + i
#                     all_results.append({
#                         "ID": df.iloc[row_idx]["ID"],
#                         "raw_input": batch_prompts[i],
#                         "raw_output": raw_output,
#                         "answer": answer
#                     })
#
#                 pbar.update(batch_size)
#                 pbar.set_postfix(batch_size=current_batch_size, time_per_sample=f"{time_per_sample:.2f}s")
#
#             except RuntimeError as e:
#                 if "out of memory" in str(e).lower():
#                     print(f"\n⚠️ OOM error with batch size {current_batch_size}. Reducing batch size.")
#                     torch.cuda.empty_cache()
#                     gc.collect()
#                     current_batch_size = max(current_batch_size // 2, 1)
#                     continue
#                 else:
#                     print(f"\nError processing batch: {e}")
#                     for i, row in batch.iterrows():
#                         prompt = row['prompt']
#                         try:
#                             single_result = process_batch([prompt], tokenizer, model)[0]
#                             all_results.append({
#                                 "ID": row["ID"],
#                                 "raw_input": prompt,
#                                 "raw_output": single_result[0],
#                                 "answer": single_result[1]
#                             })
#                         except Exception as inner_e:
#                             print(f"Error processing row {i}: {inner_e}")
#                             all_results.append({
#                                 "ID": row["ID"],
#                                 "raw_input": prompt,
#                                 "raw_output": f"Error: {str(inner_e)}",
#                                 "answer": None
#                             })
#                         pbar.update(1)
#
#             if (start_idx // args.checkpoint_interval) != (end_idx // args.checkpoint_interval):
#                 checkpoint_idx = (end_idx // args.checkpoint_interval) * args.checkpoint_interval
#                 print(f"\n✅ Checkpoint {checkpoint_idx}/{total} — 중간 저장 중...")
#                 temp_df = pd.DataFrame(all_results)
#                 temp_df.to_csv(
#                     f"{args.save_dir}/checkpoint_{checkpoint_idx}.csv",
#                     index=False,
#                     encoding="utf-8-sig"
#                 )
#
#             start_idx = end_idx
#
#             if start_idx % (args.checkpoint_interval // 2) == 0:
#                 torch.cuda.empty_cache()
#                 gc.collect()
#
#     return pd.DataFrame(all_results)

##########

# utils/inference_utils.py
import os
import time
import pandas as pd
from tqdm import tqdm
from utils.answer_extraction import extract_answer
from utils.prompt_builder import make_prompt

# 1. 전체 데이터에서 2000개 샘플링 seed = 42
def load_sampled_data(test_path, sample_size=2000, seed=42):
    full_df = pd.read_csv(test_path)
    sampled_df = full_df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    return sampled_df

# 2. LLM 추론 실행
def run_llama_inference(df, llm, args):
    results = []
    total = len(df)

    print("🚀 LLM 추론 시작... (샘플 개수:", total, ")")
    try:
        for i, row in tqdm(df.iterrows(), total=total, desc="Processing"):
            prompt = row["prompt"]
            try:
                response = llm(
                    prompt,
                    max_tokens=64,
                    stop=["\n", "###", "답변:"],
                    seed=42  # ✅ seed 고정
                )
                raw_output = response["choices"][0]["text"].strip()
                raw_answer, answer = extract_answer(raw_output, row["choices"])
            except Exception as e:
                print(f"⚠️ Error at row {i}: {e}")
                raw_output = f"Error: {str(e)}"
                answer = "알 수 없음"

            results.append({
                "ID": row["ID"],
                "raw_input": prompt,
                "raw_output": raw_output,
                "answer": answer
            })

            # 중간 저장
            if len(results) % 100 == 0:
                checkpoint_path = os.path.join(args.save_dir, "checkpoint_partial.csv")
                pd.DataFrame(results).to_csv(checkpoint_path, index=False, encoding="utf-8-sig")
                print(f"💾 체크포인트 저장됨: {checkpoint_path} (총 {len(results)}개)")

    finally:
        os.makedirs(args.save_dir, exist_ok=True)
        final_path = os.path.join(args.save_dir, "submission.csv")
        final_df = pd.DataFrame(results)[["ID", "raw_input", "raw_output", "answer"]]
        final_df.to_csv(final_path, index=False, encoding="utf-8-sig")
        print(f"\n✅ 결과 저장 완료: {final_path} (총 {len(results)}개)")

    return pd.DataFrame(results)