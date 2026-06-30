from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # databases
    pg_dsn: str
    neo4j_uri: str
    neo4j_user: str = "neo4j"
    neo4j_password: str

    # ollama — mac-server (embedding + reranker)
    ollama_base_url: str

    # ollama cloud — llm inference (llm_provider=ollama 시 사용)
    ollama_cloud_base_url: str = ""
    ollama_api_key: str = ""

    # llm provider
    llm_provider: str = "gemini"  # "gemini" | "ollama"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # models
    embedding_model: str = "qwen3-embedding:8b"
    embedding_dim: int = 1024
    reranker_model: str = "bge-reranker-v2-m3"
    llm_model: str = "qwen2.5:32b"  # ollama provider 전용

    # RARR 역할별 모델 (결정 N)
    rarr_draft_provider: str = "gemini"
    rarr_draft_model: str = "gemini-2.5-flash"
    rarr_edit_provider: str = "gemini"
    rarr_edit_model: str = "gemini-2.5-flash"
    rarr_reason_provider: str = "gemini"
    rarr_reason_model: str = "gemini-2.5-flash"
    rarr_aux_provider: str = "ollama"
    rarr_aux_model: str = "glm-5.2"

    # RARR 튜닝 노브 (0 = 무제한)
    rarr_max_claims: int = 0
    rarr_questions_per_claim: int = 0

    # retrieval
    retrieve_top_k: int = 30
    rerank_top_k: int = 5
    rrf_k: int = 60
    fallback_score_threshold: float = 0.5

    # complex mode
    sufficiency_max_iter: int = 2
    complex_mode_timeout_s: int = 20
    llm_timeout_s: int = 120

    # grounding gate (F1)
    grounding_action: str = "flag"  # "flag" | "strip"

    # source cards shown in chat UI
    source_top_k: int = 3

    # law.go.kr OPEN API (법제처 국가법령정보 공동활용)
    law_api_oc: str = ""
    law_api_base_url: str = "http://www.law.go.kr/DRF"

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
