# 설치 / 실행

## 사전 요건

- Python 3.12+
- PostgreSQL 16+ with pgvector extension
- Neo4j 5.x
- 온프레미스 Ollama (`qwen3-embedding:8b`, `bge-reranker-v2-m3`)
- Gemini API 키 (Google AI Studio)
- Ollama Cloud 엔드포인트 + API 키 (glm-5.2 aux 모델용)

## 설정

```bash
cp .env.example .env
```

필수 설정값:

| 항목 | 설명 |
|---|---|
| `PG_DSN` | PostgreSQL 연결 문자열 |
| `NEO4J_URI` / `NEO4J_PASSWORD` | Neo4j 연결 |
| `OLLAMA_BASE_URL` | 온프레미스 Ollama (임베딩·리랭커) |
| `GEMINI_API_KEY` | Gemini API 키 (draft/edit/reason) |
| `OLLAMA_CLOUD_BASE_URL` / `OLLAMA_API_KEY` | Ollama Cloud (aux: glm-5.2) |
| `JWT_SECRET` | `openssl rand -hex 32` 로 생성 |
| `AUTH_USERS` | `username:bcrypt_hash` 형식, 콤마 구분 |
| `LAW_API_OC` | 법제처 OPEN API(open.law.go.kr) 신청 ID. 인제스트 스크립트 실행 전 필수 |

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

```bash
psql "$PG_DSN" -f sql/schema.sql
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

## 샘플 데이터 적재

```bash
python scripts/load_sample.py
```

조문 3~5개, 판례 3~5개, 관계(CITES/BASED_ON/OVERRULED_BY 각 1건)를 적재하고 임베딩을 채운다.

## 실 데이터 수집 (법제처 OPEN API, 결정 D)

`LAW_API_OC` 설정 후 순서대로 실행:

```bash
python -m scripts.ingest_laws      # 조문 수입 (핵심 3법: 소득세법/법인세법/부가가치세법)
python -m scripts.ingest_cases     # 판례 수입 + 인용/근거조문 그래프
python -m scripts.backfill_embeddings  # 임베딩 채우기
python -m scripts.update_validity  # validity_flag 계산
```

## validity_flag 갱신

```bash
python scripts/update_validity.py
```

Neo4j 그래프를 읽어 `case_chunks.validity_flag`를 계산·업데이트한다. 데이터 변경 시 재실행.

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
