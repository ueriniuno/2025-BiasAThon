#utils/answer_extraction
import re
import ast

def extract_answer(text: str, choices: list):
    """
    모델 응답에서 선택한 답변을 추출
    """
    raw_output = text.strip()

    # 특수문자 선택지 찾기
    match = re.search(r"[①②③]", raw_output)
    if match:
        symbol_to_index = {"①": 0, "②": 1, "③": 2}
        index = symbol_to_index[match.group()]
        answer = choices[index] if index < len(choices) else "알 수 없음"
    else:
        answer = "알 수 없음"

    if "답변할 수 있는 정보가 부족합니다" in raw_output and index != 2:
        answer = "알 수 없음"

    return raw_output, answer