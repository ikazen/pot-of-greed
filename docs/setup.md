# 설치 / 실행

## 사전 요건

- Python 3.12+
- PostgreSQL 16+ with pgvector extension
- Neo4j 5.x
- 온프레미스 Ollama (`qwen3-embedding:8b`, `bge-reranker-v2-m3`)
- Gemini API 키 (Google AI Studio)
- Ollama Cloud 엔드포인트 + API 키 (gpt-oss:20b aux 모델용)
- 세법/판례 코퍼스는 [`lawcorpus`](https://github.com/ikazen/law-corpus) 라이브러리가 관리 — 이 repo의
  `PG_DSN`/`NEO4J_*`는 lawcorpus가 적재한 것과 **같은 DB/그래프**를 가리켜야 한다

## 설정

```bash
cp .env.example .env
```

필수 설정값:

| 항목 | 설명 |
|---|---|
| `PG_DSN` | PostgreSQL 연결 문자열 (lawcorpus와 동일 DB) |
| `NEO4J_URI` / `NEO4J_PASSWORD` | Neo4j 연결 (lawcorpus와 동일 그래프) |
| `OLLAMA_BASE_URL` | 온프레미스 Ollama (임베딩·리랭커) |
| `GEMINI_API_KEY` | Gemini API 키 (draft/edit/reason) |
| `OLLAMA_CLOUD_BASE_URL` / `OLLAMA_API_KEY` | Ollama Cloud (aux: gpt-oss:20b) |
| `JWT_SECRET` | `openssl rand -hex 32` 로 생성 |
| `AUTH_USERS` | `username:bcrypt_hash` 형식, 콤마 구분 |

bcrypt 해시 생성:
```bash
python -c "import bcrypt; print('admin:' + bcrypt.hashpw(b'password', bcrypt.gensalt()).decode())"
```

Chainlit UI 관련 (compose.yml의 ui 컨테이너):
```
CHAINLIT_AUTH_SECRET=<openssl rand -hex 32>
CHAINLIT_DB_DSN=postgresql+asyncpg://potofgreed:<pw>@postgres:5432/potofgreed
```

## DB 스키마 적용

세법/판례 코퍼스 스키마(article_chunks/case_chunks + Neo4j 제약)는 lawcorpus 쪽에서 적용한다:

```bash
lawcorpus apply-schema   # LAWCORPUS_PG_DSN/LAWCORPUS_NEO4J_* 필요
```

Chainlit UI 전용 테이블만 이 repo가 소유:

```bash
psql "$PG_DSN" -f sql/chainlit_schema.sql
```

## 실행

### Docker Compose (운영/권장)

```bash
docker compose up -d
```

api 컨테이너(`pot-of-greed-api`, :8000) + ui 컨테이너(`pot-of-greed-ui`)가 nexus 네트워크에 연결.
헬스체크는 무인증 `GET /healthz`를 호출한다(pg/neo4j ping 실패 시 503) — 컨테이너가
unhealthy로 뜨면 이 엔드포인트로 직접 원인 확인.

### 로컬 개발

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## 샘플/실 데이터 적재

데이터 적재는 전부 lawcorpus CLI로 한다 (`LAWCORPUS_LAW_API_OC` 등은 lawcorpus 쪽 `.env` 참조):

```bash
lawcorpus load-sample                                    # 개발용 소량 샘플
lawcorpus ingest-laws --law 소득세법 --law 법인세법 --law 부가가치세법
lawcorpus ingest-cases --query 소득세 --query 법인세 --query 부가가치세
lawcorpus backfill                                        # 임베딩 채우기
lawcorpus update-validity                                 # validity_flag 계산
```

자세한 옵션은 [lawcorpus docs/setup.md](https://github.com/ikazen/law-corpus/blob/main/docs/setup.md) 참조.

## RARR eval 하니스

RARR 파이프라인 품질·지연 실측 (인프라 가동 상태에서 실행):

```bash
python -m scripts.rarr_eval --mode both --report
# 결과: eval/results/<timestamp>.md
```

옵션:
- `--mode simple|complex|both` — 측정 대상 모드
- `--limit N` — 질의 수 제한
- `--out PATH` — JSON 리포트 저장
- `--queries PATH` — 커스텀 질의셋 JSON (기본: scripts/eval_queries.json)

실측 후 지연 목표(decisions.md H) 재설정 필요.
