# model_runner.py

import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt_engineer import make_prompt
from postprocessor import extract_answer

gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

def load_model(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    token = "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" 
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    
    # 디코더-only 모델에서 padding이 없을 수 있으므로 eos로 설정
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        token=token,
        torch_dtype=torch.bfloat16  # or torch.float16 (메모리 상황에 따라)
    )
    return tokenizer, model

def predict_batch_answers(tokenizer, model, contexts, questions, choices_list, max_new_tokens=32):
    # 프롬프트 구성
    prompts = [make_prompt(ctx, q, ch) for ctx, q, ch in zip(contexts, questions, choices_list)]

    # 토크나이징
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 중립성 유지 목적
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # 후처리: A/B/C → 선택지 매핑
    parsed = [extract_answer(res, ch) for res, ch in zip(results, choices_list)]
    return prompts, [r[0] for r in parsed], [r[1] for r in parsed]
