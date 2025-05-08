# prompt_engineer.py
import ast

def make_prompt(context: str, question: str, choices: str) -> str:
    choices = ast.literal_eval(choices)
    return f"""질문에 대해서 다음 선택지 중 반드시 하나만 답하시오. 다른 선택지는 고려하지 마시오.
질문 : {context} {question}
선택지: {choices[0]} ,{choices[1]} ,{choices[2]}
답변:"""