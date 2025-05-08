#model_runner.py
import torch,gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt_engineer import make_prompt
from postprocessor import extract_answer
import pandas as pd

gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

def load_model(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    token="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"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" #ㅊㅇ

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        token=token,
        torch_dtype=torch.bfloat16
    )
    return tokenizer, model

def predict_batch_answers(tokenizer, model, contexts, questions, choices_list, max_new_tokens=16):
    prompts = [make_prompt(ctx, q, ch) for ctx, q, ch in zip(contexts, questions, choices_list)]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None, #ㅊㅇ
            top_p=None, #ㅊㅇ
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    parsed = [extract_answer(res) for res in results]
    return prompts, [r[0] for r in parsed], [r[1] for r in parsed]
