#prompt_engineer.py

import ast, re,os
from retriever import get_relevant, rerank_with_cross_encoder, mmr
from transformers import AutoTokenizer as T5Tokenizer, AutoModelForSeq2SeqLM as T5ForConditionalGeneration
import torch

from keybert import KeyBERT
kw_model = KeyBERT(model="jhgan/ko-sbert-nli")

# MMR 파라미터: 환경변수로 k와 λ를 넘기고, 없으면 기본값 사용
MMR_K      = int(os.getenv("MMR_K", "2"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.9"))

# retrieval 방식 선택
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
RETRIEVAL_METHOD = os.getenv("RETRIEVAL_METHOD", "all")

# Konlpy Okt 초기화 시도 (실패 시 okt=None)
try:
    from konlpy.tag import Okt
    okt = Okt()
except Exception:
    okt = None

# ① 시스템 프롬프트 
_PROMPT = """[INST]
<<SYS>>
당신은 **한국어에 최적화된 중립적 AI**입니다.
아래 지시사항을 **엄격히** 준수하세요:
1) 어떠한 편향도 배제하고 객관·공정한 어조 유지
2) 형식: **답변: 번호 이유** (예: `1 이유`)
   - 번호는 1, 2, 3 중 하나만
   - 이유는 한 문장 이내로 간결하게
3) 선택지 1과 2가 모두 가능하거나 모두 불가능하거나 확실하지 않으면 반드시 3을 선택
4) **오직 선택한 번호에 대한 이유만** 작성하세요. 다른 번호의 이유는 절대 쓰지 마십시오.
5) **반드시 한 줄(한 문장)만** 출력하고, 이후 다른 텍스트나 추가 번호를 절대 덧붙이지 마세요.
6) 그 외 다른 내용은 **절대 출력 금지**  

**출력 예시:**
3 정보가 부족하여 명확히 판단할 수 없습니다.
<</SYS>>

다음 입력을 처리하세요.
[참고자료]
{ref}

[질문]
{context}
{question}

선택지:
1. {a}
2. {b}
3. {c}

답변:
"""

# 선택지 파싱 함수
def safe_parse_choices(choice_str: str):
    try:
        ch = ast.literal_eval(choice_str)
    except (SyntaxError, ValueError):
        choice_str = choice_str.strip("[] ")
        ch = [c.strip(" '""\n") for c in choice_str.split(",")]
    while len(ch) < 3:
        ch.append("선택지 없음")
    return ch[:3]

# 프롬프트 생성 함수
def make_prompt(context: str, question: str, choices: str) -> str:
    # 1) 한국어 컨텍스트/질문 그대로 사용
    ch_list = safe_parse_choices(choices)

    # 2) KeyBERT로 핵심 키워드 추출 → 검색 질의 생성
    kw_results = kw_model.extract_keywords(
        question,
        keyphrase_ngram_range=(1, 2),
        stop_words="korean",
        top_n=5
    )
    # [('마약', 0.82), ('벤치', 0.47), ...] → ['마약','벤치',...]
    keywords = [kw for kw, _ in kw_results]
    search_query = " ".join(keywords)

    # 2) 후보 생성(DB+SBERT)
    candidates = get_relevant(search_query, k=RETRIEVAL_K, method=RETRIEVAL_METHOD)

    # 3) CE 재랭킹
    reranked = rerank_with_cross_encoder(question, candidates)

    # 4) MMR 호출
    ref_sents = mmr(question, reranked, k=MMR_K, lambda_param=MMR_LAMBDA)

    # 8) 원문 그대로 참고자료 블록 생성
    ref_block = f"① {ref_sents[0]}\n② {ref_sents[1]}"
    
    
    # 최종 프롬프트
    return _PROMPT.format(
        ref=ref_block,
        context=context,
        question=question,
        a=ch_list[0],
        b=ch_list[1],
        c=ch_list[2],
    )
