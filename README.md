# pot-of-greed

세무사용 세법/판례 질의응답 백엔드. RARR(Retrofit Attribution using Research and Revision) 파이프라인 기반.

동시 사용자 5인 이하 전제. 품질 최대화가 1순위 — 스케일이 아니라 정확성.

## 핵심 설계 원칙

- 모든 답변에 조문번호/판례번호 출처 부착
- 판례 유효성(폐기/개정)을 버리지 않고 경고와 함께 제시
- 단순 질의 2~4초 / 복잡 질의 최대 20초

## 구조

```
app/
  api/        FastAPI 엔드포인트 (/chat, /chat/stream, /health)
  auth/       JWT 인증, 고정 계정 로그인 (/auth/token)
  rarr/       RARR 파이프라인 (draft→decompose→research→agree→edit→attribution)
  retrieval/  임베딩·벡터·키워드검색·RRF·리랭킹·그래프확장
  agent/      쿼리분해·충분성루프·근거검증
  router/     단순/복잡 모드 라우팅
  reasoning/  답변 빌더·법리 검토 (legal_reasoning_layer)
  llm/        LLM provider 추상화 (Gemini / Ollama Cloud)
  ingest/     법제처 OPEN API 수집·매핑 (조문/판례)
  db/         PostgreSQL(pgvector) / Neo4j 커넥션
docs/         설계 문서
scripts/      개발·운영 도구
```

## 개발 도구

```bash
# LLM API 테스트 (대화형)
python -m scripts.llm_test
python -m scripts.llm_test "소득세법 제14조 요지는?"
python -m scripts.llm_test --provider gemini --model gemini-2.5-pro

# 데이터 수집 (법제처 OPEN API, 결정 D)
python -m scripts.ingest_laws
python -m scripts.ingest_cases
python -m scripts.backfill_embeddings

# RARR eval 하니스 (인프라 가동 시)
python -m scripts.rarr_eval --mode both --report
python -m scripts.rarr_eval --mode simple --limit 4
```

## 문서

- [아키텍처](docs/architecture.md)
- [설계](docs/design.md)
- [결정 기록](docs/decisions.md)
- [API 스펙](docs/spec.md)
- [설치/실행](docs/setup.md)
