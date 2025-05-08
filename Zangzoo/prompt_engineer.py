# prompt_engineer.py
import ast
from retriever import get_relevant

_PROMPT = """다음은 다양한 문화적 배경을 가진 인물들의 행동을 설명하는 상황입니다.

[참고자료]
{ref}

[질문]
{context}
{question}

선택지:
A. {a}
B. {b}
C. {c} (정보가 충분치 않을 때)

지시사항:
- 아래 보기 중 A, B, C **중 하나의 알파벳만** 선택해 답변을 시작하세요.
- **A 와 B 모두 가능하거나 확실하지 않으면 C**를 선택하세요.
- 선택 이유를 한 문장으로 간단히 적어주세요.

답변:
"""

def make_prompt(context: str, question: str, choices: str) -> str:
    ch  = ast.literal_eval(choices)
    ref = get_relevant(context + " " + question, k=2)
    ref_block = "① " + ref[0] + "\n② " + ref[1]
    return _PROMPT.format(
        ref=ref_block,
        context=context, question=question,
        a=ch[0], b=ch[1], c=ch[2]
    )
