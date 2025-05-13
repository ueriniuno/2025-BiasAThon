from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")


def make_prompt(context: str, question: str, choices: list[str]) -> str:
    """
    LLaMA 3 Instruct 모델용 프롬프트 생성 함수
    """
    system_message = (
        "You are a helpful AI assistant who answers Korean multiple-choice reasoning questions "
        "based solely on the context provided.\n"
        "Always choose one of the given options (①, ②, ③) and explain your reasoning in one sentence.\n"
        "If the context lacks enough information, choose ③ and say '질문에 답변할 수 있는 정보가 부족합니다.'\n"
        "Answer in Korean."
    )

    # 선택지 출력 형식 수정
    # 리스트의 각 항목을 안전하게 문자열로 처리
    formatted_choices = []
    for i, choice in enumerate(choices):
        if i == 0:
            formatted_choices.append(f"① {str(choice)}")
        elif i == 1:
            formatted_choices.append(f"② {str(choice)}")
        elif i == 2:
            formatted_choices.append(f"③ {str(choice)}")

    choice_text = "\n선택지:\n" + "\n".join(formatted_choices)

    user_content = (
        f"context:\n{context.strip()}\n\n"
        f"question:\n{question.strip()}\n"
        f"{choice_text}\n\n"
        "답변 형식: ① 이유 한 문장 (예: ① 그는 배려심을 보였기 때문이다.)"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content}
    ]

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return prompt