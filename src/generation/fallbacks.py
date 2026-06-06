import re


GREETING_PATTERNS = {
    "hi",
    "hello",
    "hey",
    "yo",
    "sup",
    "good morning",
    "good afternoon",
    "good evening",
}

AI_ML_KEYWORDS = {
    "ai",
    "ml",
    "dl",
    "machine learning",
    "deep learning",
    "neural network",
    "llm",
    "language model",
    "transformer",
    "attention",
    "self-attention",
    "rag",
    "retrieval",
    "reranking",
    "cross encoder",
    "embedding",
    "vector",
    "token",
    "tokenization",
    "bert",
    "gpt",
    "llama",
    "yolo",
    "cnn",
    "rnn",
    "lstm",
    "gan",
    "vae",
    "diffusion",
    "clip",
    "object detection",
    "segmentation",
    "classification",
    "reinforcement learning",
    "rl",
    "supervised",
    "unsupervised",
    "self-supervised",
    "contrastive",
    "fine-tuning",
    "finetuning",
    "lora",
    "adapter",
    "moe",
    "mixture of experts",
    "nli",
    "fact verification",
    "hallucination",
    "grounding",
    "backprop",
    "gradient",
    "optimizer",
    "adam",
    "dropout",
    "batch norm",
    "layer norm",
}

EXPLAINER_PATTERNS = (
    "what is",
    "what are",
    "tell me about",
    "explain",
    "explain about",
    "overview",
    "introduction to",
    "describe",
    "how does",
    "how do",
)


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower().rstrip("?!.,"))


def is_greeting(query: str) -> bool:
    normalized = normalize_query(query)
    return normalized in GREETING_PATTERNS


def greeting_response():
    return (
        "Hi. Ask me about an AI, machine learning, or deep learning topic, "
        "and I will try to answer from the paper corpus with citations when evidence is available."
    )


def is_basic_ai_ml_query(query: str) -> bool:
    normalized = normalize_query(query)
    if len(normalized.split()) > 18:
        return False

    has_explainer_shape = (
        any(normalized.startswith(pattern) for pattern in EXPLAINER_PATTERNS)
        or len(normalized.split()) <= 5
    )
    has_ai_keyword = any(keyword in normalized for keyword in AI_ML_KEYWORDS)

    return has_explainer_shape and has_ai_keyword


def should_answer_without_retrieval(query: str) -> bool:
    normalized = normalize_query(query)
    has_ai_keyword = any(keyword in normalized for keyword in AI_ML_KEYWORDS)
    if not has_ai_keyword:
        return False

    if normalized.startswith("tell me about"):
        return True

    if len(normalized.split()) <= 3:
        return True

    return "yolo" in normalized
