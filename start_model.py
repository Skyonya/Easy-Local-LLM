from llama_cpp import Llama

SYSTEM_PROMPT = "You are AI chatbot."
N_CTX = 2000
TOP_K = 30
TOP_P = 0.9
TEMPERATURE = 0.01
REPEAT_PENALTY = 1.1

SYSTEM_TOKEN = 1587
USER_TOKEN = 2188
BOT_TOKEN = 12435
LINEBREAK_TOKEN = 13

ROLE_TOKENS = {
    "user": USER_TOKEN,
    "bot": BOT_TOKEN,
    "system": SYSTEM_TOKEN
}

MODEL_PATH = 'E:/PROG/models/saiga_mistral_7b_q4_K/model-q4_K.gguf'


def get_message_tokens(model, role, content):
    message_tokens = model.tokenize(content.encode("utf-8"))
    message_tokens.insert(1, ROLE_TOKENS[role])
    message_tokens.insert(2, LINEBREAK_TOKEN)
    message_tokens.append(model.token_eos())
    return message_tokens


def get_system_tokens(model):
    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
    return get_message_tokens(model, **system_message)


def get_start_tokens():
    # обнуляем токены
    local_system_tokens = get_system_tokens(MODEL)
    return local_system_tokens


def get_answer(user_text: str) -> str:
    local_tokens = get_start_tokens()
    # local_tokens = tokens
    message_tokens = get_message_tokens(model=MODEL, role="user", content=user_text)
    role_tokens = [MODEL.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
    local_tokens += message_tokens + role_tokens
    generator = MODEL.generate(
        local_tokens,
        top_k=TOP_K,
        top_p=TOP_P,
        temp=TEMPERATURE,
        repeat_penalty=REPEAT_PENALTY
    )

    bot_text = ""
    for token in generator:
        token_str = MODEL.detokenize([token]).decode("utf-8", errors="ignore")
        # input_tokens.append(token)
        if token == MODEL.token_eos():
            break
        bot_text += token_str
        # print(token_str)
        # print(token_str, end="", flush=True)
    return bot_text


if __name__ == "__main__":
    MODEL = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_parts=1,
    )

    system_tokens = get_system_tokens(MODEL)
    tokens = system_tokens
    MODEL.eval(tokens)

    while True:
        user_message = input("Q: ")
        bot_message = get_answer(user_message)
        print(f"BOT: {bot_message}")
