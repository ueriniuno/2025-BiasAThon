import torch, gc, os, re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from prompt_engineer import make_prompt
from postprocessor import extract_answer

_RE_ABC = re.compile(r"^\s*([ABC])\b")

def load_pipeline_model(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 양자화 설정 추가
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        token=token,
        quantization_config=quant_config
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )

    return pipe

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


def predict_batch_answers_with_pipeline(pipe, prompts, choice_list, batch_size=4
):
    raws, answers = [], []
    n = len(prompts)
    idx = 0

    # 동적 배치 처리 추가 (프롬프트를 한 번에 토크나이즈 하지 않고 배치)
    while idx < n:
        batch_prompts = prompts[idx:idx+batch_size]
        outputs = pipe(batch_prompts)

        for i, output in enumerate(outputs):
            generated = output[0]["generated_text"]
            raw, ans = extract_answer(generated, choice_list[idx + i])
            raws.append(raw)
            answers.append(ans)

            print(f"✅ Sample {idx + i + 1} / {n} processed")

        idx += batch_size

        # 메모리 정리
        if idx % 100 == 0:
            gc.collect()

    return prompts, raws, answers