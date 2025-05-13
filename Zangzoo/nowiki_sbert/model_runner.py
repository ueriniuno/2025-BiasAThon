# ─────────────────────────  model_runner.py  ─────────────────────────
import os, re, gc, time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    MarianTokenizer, MarianMTModel
)
from postprocessor import extract_answer,clean_text
from prompt_engineer import safe_parse_choices,make_prompt
from tqdm.auto import tqdm     # 진행률 바
from retriever import get_relevant
import logging
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

device = "mps" if torch.backends.mps.is_available() else "cpu"

# ──────────────── Llama 모델 로드 ────────────────
_MODEL_CACHE = None  # (tokenizer, model)

def load_model(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    token = os.getenv("HF_TOKEN")
    tok   = AutoTokenizer.from_pretrained(model_name, token=token)
    
    tok.pad_token    = tok.eos_token
    tok.padding_side = "left"

    print("🔧  Llama warm-up…")
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": "mps"},
        torch_dtype=torch.float16,
        token=token
    )
    if hasattr(torch, "compile"):
        try:
            mdl = torch.compile(mdl)
        except:
            pass

    # Warm-up
    dummy = tok("워밍업", return_tensors="pt").to("mps")
    with torch.no_grad():
        mdl.generate(**dummy, max_new_tokens=1)
    print("✅  Llama ready\n")

    _MODEL_CACHE = (tok, mdl)
    return _MODEL_CACHE

# ──────────────── 3) 배치 추론 + 번역 파이프라인 ────────────────
_RE_NUM = re.compile(r"^\s*([123])\b")

def _tokenize(tok, texts):
    return tok(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to("mps")

# 배치 추론 함수
def predict_batch_answers(
    tokenizer, model,
    contexts, questions, choices_list,
    max_new_tokens: int = 64,
    dyn_bs: int = 2,
):
    n, idx = len(contexts), 0
    prompts, raws, answers = [], [], []
    pbar = tqdm(total=n, unit="row", desc="RAG Inf", disable=True)

    while idx < n:
        bs = min(dyn_bs, n - idx)
        
        logging.info(f">>> dyn_batch size: {bs}")
        
        batch_prompts = []
        for j in range(idx, idx + bs):
            ref = get_relevant(questions[j])
            prompt = make_prompt(contexts[j], questions[j], choices_list[j])
            batch_prompts.append(prompt)
            prompts.append(prompt)
            
        # 한국어 프롬프트 → 바로 inference
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.1,
                top_k=50,
                no_repeat_ngram_size=2,
            )
        # postprocessing (기존 로직 유지)
        decoded = tokenizer.batch_decode(
            out[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        for k, model_out in enumerate(decoded):
            raw = clean_text(model_out)
            raws.append(raw)
            _, num = extract_answer(raw)
            choice = safe_parse_choices(choices_list[idx + k])
            answers.append(choice[int(num)-1] if num.isdigit() else choice[2])

        idx += bs
        
    
    return prompts, raws, answers
# ────────────────────────────────────────────────────────────────────