# postprocessor.py
import re

def extract_answer(text: str):
    raw_answer = text.split("답변:")[-1].strip()
    result = re.search(r"답변:\s*([^\n\r:]+)", text)
    answer = result.group(1).strip() if result else None
    return raw_answer, answer