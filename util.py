import tiktoken

EMBEDDING_MODEL = "text-embedding-ada-002"
SAVE_PATH = "wiki_embeddings.csv"
GPT_MODEL = "gpt-3.5-turbo"


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
