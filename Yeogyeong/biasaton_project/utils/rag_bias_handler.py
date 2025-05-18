# utils/rag_bias_handler.py
def bias_aware_rag_system(query, kb_instance, query_llm, contains_bias_fn):
    bias_info = kb_instance.retrieve_for_query(query)
    enhanced_prompt = create_debiased_prompt(query, bias_info)

    response = query_llm(enhanced_prompt)

    if contains_bias_fn(response, bias_info):
        response = query_llm(enhanced_prompt + "\n편향 없이, 명시적으로 언급된 정보만 기반으로 다시 답하세요.")

    return response


def create_debiased_prompt(query, bias_info):
    base_prompt = "다음 질문에 답할 때, 명시적으로 언급된 증거만 사용하고 어떤 집단에 대한 가정이나 고정관념을 피하세요:"

    for bias in bias_info.get("identified_biases", []):
        base_prompt += f"\n- {bias['type']} 편향 주의: {bias['entity']}에 대한 판단은 행동에 기반해야 합니다."

    for frame in bias_info.get("neutral_frames", []):
        base_prompt += f"\n- 중립적 접근: {frame}"

    base_prompt += f"\n\n질문: {query}"
    return base_prompt
