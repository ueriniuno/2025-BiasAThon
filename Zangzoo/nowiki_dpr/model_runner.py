# model_runner.py

import os, gc, logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from postprocessor import extract_answer, clean_text
from prompt_engineer import make_prompt, safe_parse_choices
from retriever import get_relevant

# Translation (optional)
try:
    from translation_utils import translate_to_en, translate_to_ko
    TRANSLATION = True
except ImportError:
    TRANSLATION = False

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
device = "mps" if torch.backends.mps.is_available() else "cpu"
_CACHE = None

def load_model(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    global _CACHE
    if _CACHE: return _CACHE
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token; tok.padding_side = "left"
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, device_map={"":device}, torch_dtype=torch.float16
    )
    # warm-up
    dummy = tok("워밍업", return_tensors="pt").to(device)
    with torch.no_grad(): mdl.generate(**dummy, max_new_tokens=1)
    _CACHE = (tok, mdl)
    return _CACHE

def predict_batch_answers(
    tokenizer, model,
    contexts, questions, choices_list,
    few_shot=None,
    use_cot=False,
    use_debias=False,
    use_rag=False,
    use_expansion=False,
    max_new_tokens=64,
    dyn_bs=2
):
    n, idx = len(questions), 0
    prompts, raws, answers = [], [], []

    while idx < n:
        bs = min(dyn_bs, n-idx)
        batch = []
        for j in range(idx, idx+bs):
            q = questions[j]
            ctx = contexts[j] if use_rag else ""
            # query expansion
            rel_kwargs = {"use_expansion":use_expansion}
            ref = get_relevant(q, method="hybrid", **rel_kwargs) if use_rag else None
            # translation
            if use_expansion and TRANSLATION:
                q   = translate_to_en(q)
                ctx = translate_to_en(ctx)
            prompt = make_prompt(
                context=ctx,
                question=q,
                choices=choices_list[j],
                few_shot=few_shot,
                use_cot=use_cot,
                use_debias=use_debias,
                ref_sents=ref
            )
            batch.append(prompt); prompts.append(prompt)

        inp = tokenizer(batch, return_tensors="pt", truncation=True,
                        padding=True, max_length=512).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                top_k=50,
                no_repeat_ngram_size=2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
        dec = tokenizer.batch_decode(
            out[:, inp["input_ids"].shape[1]:],
            skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        for k, text in enumerate(dec):
            txt = clean_text(text)
            raws.append(txt)
            raw, num = extract_answer(txt)
            opts = safe_parse_choices(choices_list[idx+k])
            ans = opts[int(num)-1] if num.isdigit() and 1<=int(num)<=3 else opts[2]
            if use_expansion and TRANSLATION:
                ans = translate_to_ko(ans)
            answers.append(ans)

        idx += bs
        if torch.backends.mps.is_available(): torch.mps.empty_cache()
        else: torch.cuda.empty_cache()
        gc.collect()

    return prompts, raws, answers
