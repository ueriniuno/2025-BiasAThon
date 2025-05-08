# ─────────────────────────  model_runner.py  ─────────────────────────
import os, re, gc, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt_engineer import make_prompt
from postprocessor import extract_answer
from tqdm.auto import tqdm     # ← 진행률 바

_RE_ABC = re.compile(r"^\s*([ABC])\b")

# ────────────────  1) 전역 캐시에 한 번만 로드  ────────────────
_MODEL_CACHE = None        # (tokenizer, model) 튜플

def load_model(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    """
    Llama-3 8B를 **최초 1회만** 로드하고 warm-up 한다.
    두 번째 호출부터는 전역 캐시를 그대로 돌려준다.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:        # 이미 불러왔으면 그대로
        return _MODEL_CACHE

    token     = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("🔧  Llama 8B warm-up (1회)…")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": "mps"},         # Mac M-시리즈
        token=token,
        torch_dtype=torch.float16
    )
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    # warm-up
    dummy = tokenizer("Warm-up", return_tensors="pt").to("mps")
    with torch.no_grad():
        model.generate(**dummy, max_new_tokens=1)
    print("✅  warm-up done\n")

    _MODEL_CACHE = (tokenizer, model)
    return _MODEL_CACHE
# ────────────────────────────────────────────────────────────────────


def _tokenize(tok, texts):
    return tok(
        texts, return_tensors="pt",
        truncation=True, max_length=512, padding=True
    ).to("mps")


# ────────────────  2) 배치-단위 추론 + 로그  ────────────────
def predict_batch_answers(
    tokenizer, model,
    contexts, questions, choices_list,
    max_new_tokens: int = 32,
    dyn_bs: int = 2,
):
    """
    dyn_bs(=2)씩 잘라서 generate → 다시 합치기
    * 🚀/✅ 로그로 generate 시작/종료 & 소요시간 확인
    * tqdm 진행률 바(전체 n 개)
    """
    prompts, raws, answers = [], [], []
    n = len(contexts)

    # tqdm 진행률
    pbar = tqdm(total=n, unit="row", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    idx = 0
    while idx < n:
        bs = min(dyn_bs, n - idx)

        batch_prompts = [
            make_prompt(contexts[j], questions[j], choices_list[j])
            for j in range(idx, idx + bs)
        ]
        inputs = _tokenize(tokenizer, batch_prompts)

        # ----------------- generate -----------------
        t0 = time.perf_counter()
        print(f"🚀  generate start: rows {idx}~{idx+bs-1}", flush=True)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        dt = time.perf_counter() - t0
        print(f"✅  generate done : rows {idx}~{idx+bs-1}  ({dt:0.1f}s)", flush=True)
        # ------------------------------------------------

        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

        for k, res in enumerate(decoded):
            raw, ans = extract_answer(res, choices_list[idx + k])
            prompts.append(batch_prompts[k])
            raws.append(raw)
            answers.append(ans)

        idx += bs
        pbar.update(bs)

        # 100 row마다 GPU 메모리 비우기
        if idx % 100 == 0:
            torch.mps.empty_cache()
            gc.collect()

    pbar.close()
    return prompts, raws, answers
# ────────────────────────────────────────────────────────────────────