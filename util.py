import tiktoken

EMBEDDING_MODEL = "text-embedding-3-small"
SAVE_PATH = "wiki_embeddings.csv"
# See https://platform.openai.com/docs/models
# For DeepSeek: https://api-docs.deepseek.com/quick_start/pricing
GPT_MODEL = "deepseek-reasoner"


def num_tokens(text: str) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model("gpt-4o")
    return len(encoding.encode(text))
