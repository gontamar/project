

def chat(question, history):
    logger.start_pipeline()

    # ✅ STOP old answer audio immediately
    generation_id = next_generation()

    global current_part

    if history is None:
        history = []

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": ""})

    yield "", history, history

    for updated_history in ask_llm(current_part, question, history, generation_id):
        yield "", updated_history, updated_history

    logger.end_pipeline()
