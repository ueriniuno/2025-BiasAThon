import ast, re
_RE_ABC = re.compile(r"^\s*([ABC])\b")

def extract_answer(text, choices, pat=_RE_ABC):
    raw = text.split("답변:")[-1].strip()
    match = pat.match(raw) 
    choice_list = ast.literal_eval(choices)
    if match and len(choice_list) == 3:
        return raw, choice_list["ABC".index(match.group(1))]
    # fallback 매칭
    for c in choice_list:
        if c.replace(" ", "")[:4] in raw.replace(" ", ""):
            return raw, c
    if "알 수 없음" in raw or "판단" in raw:
        return raw, choice_list[2]
    return raw, None
