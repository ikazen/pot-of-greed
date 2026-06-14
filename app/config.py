from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # databases
    pg_dsn: str
    neo4j_uri: str
    neo4j_user: str = "neo4j"
    neo4j_password: str

    # ollama — mac-server (embedding + reranker)
    ollama_base_url: str

    # ollama cloud — llm inference
    ollama_cloud_base_url: str
    ollama_api_key: str = ""

    # models
    embedding_model: str = "qwen3-embedding:8b"
    embedding_dim: int = 1024
    reranker_model: str = "bge-reranker-v2-m3"
    llm_model: str = "qwen2.5:32b"

    # retrieval
    retrieve_top_k: int = 30
    rerank_top_k: int = 5
    rrf_k: int = 60
    fallback_score_threshold: float = 0.5

    # complex mode
    sufficiency_max_iter: int = 2
    complex_mode_timeout_s: int = 20
    llm_timeout_s: int = 120

    # auth
    jwt_secret: str
    jwt_alg: str = "HS256"
    jwt_expire_min: int = 1440
    # "username:bcrypt_hash" entries, comma-separated
    auth_users: str = ""

    def get_auth_users(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for entry in self.auth_users.split(","):
            entry = entry.strip()
            if ":" in entry:
                user, hashed = entry.split(":", 1)
                result[user.strip()] = hashed.strip()
        return result


@lru_cache
def get_settings() -> Settings:
    return Settings()
