import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1) NLLB-200 모델 하나만 로드
device = "mps" if torch.backends.mps.is_available() else "cpu"
_MT_MODEL_NAME = "facebook/nllb-200-distilled-600M"
token = os.getenv("HF_TOKEN")

_mt_tokenizer = AutoTokenizer.from_pretrained(_MT_MODEL_NAME, use_auth_token=token)
_mt_model     = AutoModelForSeq2SeqLM.from_pretrained(_MT_MODEL_NAME, use_auth_token=token).to(device)

def translate(texts, src_lang: str, tgt_lang: str) -> list[str]:
    """
    texts: 번역할 문장들의 리스트
    src_lang: 원문 언어 코드 (예: 'eng_Latn', 'arb_Arab', 'kor_Hans', ...)
    tgt_lang: 목표 언어 코드 (예: 'kor_Hans')

    NLLB 토크나이저의 src_lang/tgt_lang 속성을 매 호출마다 바꿔주기 때문에,
    어떤 언어(영어·아랍어·그 외)→한국어 번역에도 모두 동작합니다.
    """
    # 2) 매번 src/tgt 언어 설정
    _mt_tokenizer.src_lang = src_lang
    _mt_tokenizer.tgt_lang = tgt_lang

    # 3) 인코딩·생성
    inputs = _mt_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    # 3) forced_bos_token_id 로 출력 언어 고정
    lang2id = getattr(_mt_tokenizer, "lang_code_to_id", {})
    bos_id   = lang2id.get(tgt_lang, None)

    # 4) 실제 생성. forced_bos_token_id 전달
    with torch.no_grad():
        outputs = _mt_model.generate(
            **inputs,
            forced_bos_token_id=bos_id,
        )

    # 4) 디코딩
    return [
        _mt_tokenizer.decode(o, skip_special_tokens=True)
        for o in outputs
    ]