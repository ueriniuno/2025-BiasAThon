# postprocessor.py
import re

# 한글·숫자·기본 문장부호만 남기는 함수
def clean_text(text: str) -> str:
    # 한글, 숫자, 공백, 마침표·쉼표·물음·느낌표·줄임표·중점·괄호·대시만 허용
    return re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣0-9\s\.\,\?\!\…·\(\)\-\:]', '', text).strip()

# 숫자로 시작하는지 체크하는 정규식
_RE_NUM = re.compile(r"^\s*([123])\b")

def extract_answer(text: str):
    """
    모델 응답에서
      - raw: '답변: …' 뒤의 전체 문장 (한국어)
      - num: '1','2','3' 중 하나 (매칭 실패 시 '3'으로)
    를 반환한다.
    """
   # 1) 외부 따옴표 제거
    stripped = text.strip().strip('"').strip("'").strip('“').strip('”')
    # 2) 멀티라인이면 첫 줄만 사용
    first = stripped.splitlines()[0].strip()

    # 3) 번호(1|2|3)와 optional '이유:' 접두사 무시, 그 뒤 이유만 캡처
    m = re.search(r"^([123])\s*(?:이유[:：]\s*)?(.*)$", first)
    if m:
        num    = m.group(1)
        reason = clean_text(m.group(2))
        raw    = f"{num} {reason}".strip()
    else:
        # fallback: 전체 텍스트에서 첫 번째 숫자만 골라서 번호로 처리
        m2  = _RE_NUM.search(stripped)
        num = m2.group(1) if m2 else "3"
        raw = num
    return raw, num