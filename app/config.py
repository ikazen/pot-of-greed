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
    rarr_aux_model: str = "gpt-oss:20b"

    # RARR 튜닝 노브 (0 = 무제한)
    # rarr_max_claims=0(무제한)은 decompose LLM 실패 시 규칙기반 문장분리 폴백이
    # draft 전체를 수십 개 claim으로 쪼개고, 세마포어(rarr_max_concurrency=4)로
    # 순차 처리하다 검증 예산을 전부 소진해 나머지가 빈 evidence로 떨어지는
    # 사고(#39)로 이어졌다. 8로 캡을 걸고 초과분은 기존 deferred 경고로 표기한다.
    rarr_max_claims: int = 8
    rarr_questions_per_claim: int = 0
    rarr_max_concurrency: int = 4
    # decompose(aux LLM) 하나가 검증 예산 전체(simple_mode_timeout_s)를 독점하면
    # research 이하 전 단계가 deadline 초과로 빈 evidence 반환 → 모든 claim이
    # "근거 미확인"으로 오보되는 구조적 버그(#39)였다. decompose 자기 몫만
    # 쓰도록 별도 상한을 둔다.
    rarr_decompose_timeout_s: int = 8

    # retrieval
    retrieve_top_k: int = 30
    rerank_top_k: int = 5
    rrf_k: int = 60
    fallback_score_threshold: float = 0.5

    # complex mode
    sufficiency_max_iter: int = 2
    draft_timeout_s: int = 30
    # draft 이후 검증 단계(decompose+research+agreement+edit) 예산. simple/complex 모드가
    # 검증 단계 비용이 크게 다르므로(#14) 노브를 분리.
    # simple_mode_timeout_s=12(#35)는 decompose 자체가 4~12s를 오가는 상황에서
    # decompose+research+agreement+edit 전부를 12s 안에 넣을 수 없었다(#39) —
    # decompose가 예산을 다 쓰면 이후 단계가 통째로 스킵됐다. rarr_decompose_timeout_s
    # 분리 이후에도 agreement/edit(둘 다 이번까지 실측 0회)에 남는 시간이 필요해
    # 35s로 상향. UI 클라이언트 타임아웃 90s(ui/app.py) 및 draft 자체 소요(~18s
    # 실측)를 감안한 상한.
    simple_mode_timeout_s: int = 35
    complex_mode_timeout_s: int = 45
    llm_timeout_s: int = 120

    # source cards shown in chat UI
    source_top_k: int = 3

    # logging
    log_level: str = "INFO"

    # 디버그: RARR 단계별 수정 내역을 채팅 응답에 포함 (기동 시 ON/OFF)
    debug_pipeline: bool = False

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
