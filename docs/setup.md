# 설치 / 실행

## 사전 요건

- Python 3.11+
- PostgreSQL 16+ with pgvector extension
- Neo4j 5.x
- 온프레미스 Ollama (`qwen3-embedding:8b`, `bge-reranker-v2-m3`)
- Ollama Cloud 엔드포인트 (복잡 추론용)

## 설정

```bash
cp .env.example .env
# .env 에서 아래 값 설정:
#   PG_DSN, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
#   OLLAMA_BASE_URL (온프레미스)
#   OLLAMA_CLOUD_URL
#   JWT_SECRET
#   CHAINLIT_DB_DSN (postgresql+asyncpg://... 형식, UI 대화 영속화용)
```

## DB 스키마 적용

도메인 스키마와 Chainlit 대화 테이블을 한 번 적용한다.

```bash
psql "$PG_DSN" -f sql/schema.sql
psql "$PG_DSN" -f sql/chainlit_schema.sql
```

이후 UI에 로그인하면 대화가 자동 저장되고, 좌측 사이드바에서 과거 thread를 열어 이어할 수 있다.

## 실행

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## 샘플 데이터 적재

```bash
python scripts/load_sample.py
```

조문 3~5개, 판례 3~5개, 관계(CITES/BASED_ON/OVERRULED_BY 각 1건)를 적재하고 임베딩을 채운다.

## validity_flag 갱신

```bash
python scripts/update_validity.py
```

Neo4j 그래프를 읽어 `case_chunks.validity_flag`를 계산·업데이트한다. 데이터 변경 시 재실행.
