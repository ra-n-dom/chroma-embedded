from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
import os
from typing import List

# --- Model registry ---

MODEL_MAP = {
    "stella": "dunzhang/stella_en_400M_v5",
    "modernbert": "answerdotai/ModernBERT-large",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "default": "sentence-transformers/all-MiniLM-L6-v2",
}

TRUST_REMOTE_CODE_MODELS = {"stella", "modernbert"}


# --- Device detection ---

def detect_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# --- Client-side model loading ---

def load_model(model_name: str, device: str = None) -> SentenceTransformer:
    if device is None:
        device = detect_device()
    hf_name = MODEL_MAP.get(model_name)
    if not hf_name:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_MAP.keys())}")
    kwargs = {"trust_remote_code": True} if model_name in TRUST_REMOTE_CODE_MODELS else {}

    # Stella's custom code requires xformers for memory_efficient_attention,
    # which doesn't build on macOS. Disable it on non-CUDA devices.
    if model_name == "stella" and device != "cuda":
        kwargs["config_kwargs"] = {
            "use_memory_efficient_attention": False,
            "unpad_inputs": False,
        }

    return SentenceTransformer(hf_name, device=device, **kwargs)


def embed_documents(model: SentenceTransformer, documents: List[str], batch_size: int = 64) -> list:
    return model.encode(documents, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True).tolist()


def get_embed_batch_size(model_name: str, device: str) -> int:
    if device == "cpu":
        return 32
    sizes = {"stella": 32, "modernbert": 32, "bge-large": 64, "default": 128}
    return sizes.get(model_name, 32)


# --- Server-side embedding functions (used by server.sh / Docker) ---

class StellaEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self):
        self.model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True)

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input).tolist()

class ModernBERTEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self):
        self.model = SentenceTransformer("answerdotai/ModernBERT-large", trust_remote_code=True)

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input).tolist()

class BGEEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input).tolist()

def get_embedding_function(model_name: str):
    model_map = {
        "stella": StellaEmbeddingFunction,
        "modernbert": ModernBERTEmbeddingFunction,
        "bge-large": BGEEmbeddingFunction
    }

    if model_name not in model_map:
        raise ValueError(f"Unknown embedding model: {model_name}. Available: {list(model_map.keys())}")

    return model_map[model_name]()