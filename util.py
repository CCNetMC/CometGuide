import tiktoken

EMBEDDING_MODEL = "text-embedding-3-small"
SAVE_PATH = "wiki_embeddings.csv"
# See https://platform.openai.com/docs/models
GPT_MODEL = "gpt-4o"


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
