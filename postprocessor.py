import re
import ast

def extract_answer(text: str, choices: str):
    raw_answer = text.split("답변:")[-1].strip()

    # 선택지 리스트 파싱
    try:
        choices_list = ast.literal_eval(choices)
    except:
        choices_list = []

    label_to_index = {"A": 0, "B": 1, "C": 2}

    # 1. A/B/C 알파벳으로 시작하는 응답 찾기 (예: "A.", "B ", "C\n")
    match = re.match(r"^\s*([ABC])\b", raw_answer)
    if match and len(choices_list) == 3:
        label = match.group(1)
        return raw_answer, choices_list[label_to_index[label]]

    # 2. "선택: ...입니다" 또는 "...가 정답입니다" 같은 형식 대응
    for idx, choice in enumerate(choices_list):
        clean_choice = choice.replace(" ", "")
        if clean_choice and clean_choice in raw_answer.replace(" ", ""):
            return raw_answer, choice

    # 3. "알 수 없음" 대응 (프롬프트에 설명이 포함된 경우에도)
    if "알 수 없음" in raw_answer or "판단할 수 없" in raw_answer:
        return raw_answer, choices_list[2] if len(choices_list) == 3 else None

    # 4. 실패 시 None 반환
    return raw_answer, None
