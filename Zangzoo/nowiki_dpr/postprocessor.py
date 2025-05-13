# postprocessor.py
import re

# 한글·숫자·기본 문장부호만 남기는 함수
def clean_text(text: str) -> str:
    # 한글, 숫자, 공백, 마침표·쉼표·물음·느낌표·줄임표·중점·괄호·대시만 허용
    return re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣0-9\s\.\,\?\!\…·\(\)\-\:]', '', text).strip()


_PATTERN = re.compile(r"""
    ^\s*
    (?P<circ>①|②|③)?        # 동그라미 숫자
    \s*
    \(?(?P<digit>[123])\)?   # 일반 혹은 괄호 숫자
    \s*
    (?:이유|Reason)[:：]?\s* # optional 이유/Reason
    (?P<reason>.*)           # 이유 본문
    """, re.VERBOSE)

def extract_answer(text: str):
    lines = [l.strip('“”"\' ') for l in text.strip().splitlines() if l.strip()]
    for line in reversed(lines):
        m = _PATTERN.match(line)
        if m:
            circ = m.group("circ")
            num = {"①":"1","②":"2","③":"3"}.get(circ, m.group("digit") or "3")
            reason = clean_text(m.group("reason").strip())
            return f"{num} {reason}", num
    return "3 알 수 없음", "3"