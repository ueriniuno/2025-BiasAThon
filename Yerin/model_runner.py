import torch, gc, os, re
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt_engineer import make_prompt
from postprocessor import extract_answer

_RE_ABC = re.compile(r"^\s*([ABC])\b")

def load_model(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        token=token,
        torch_dtype=torch.bfloat16
    )

    return tokenizer, model

def _tokenize(tokenizer, texts, device=None):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    if device is not None:
        inputs = inputs.to(device)
    return inputs


def predict_batch_answers(
    tokenizer,
    model,
    contexts,
    questions,
    choices_list,
    max_new_tokens: int = 32,
    dyn_bs: int = 2,
):
    prompts, raws, answers = [], [], []
    n, idx = len(contexts), 0

    # 동적 배치 처리 추가 (프롬프트를 한 번에 토크나이즈 하지 않고 배치)
    while idx < n:
        bs = min(dyn_bs, n - idx)
        batch_prompts = [
            make_prompt(contexts[j], questions[j], choices_list[j])
            for j in range(idx, idx + bs)
        ]

        inputs = _tokenize(tokenizer, batch_prompts)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

        # choice_list (실제 선택지를 인자로 받음)
        for k, res in enumerate(decoded):
            raw, ans = extract_answer(res, choices_list[idx + k])
            prompts.append(batch_prompts[k])
            raws.append(raw)
            answers.append(ans)

        idx += bs

        # 메모리 정리
        if idx % 100 == 0:
            torch.mps.empty_cache()
            gc.collect()

    return prompts, raws, answers