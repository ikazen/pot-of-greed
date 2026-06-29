# pot-of-greed

세무사용 세법/판례 질의응답 RAG 백엔드.

동시 사용자 5인 이하 전제. 품질 최대화가 1순위 — 스케일이 아니라 정확성.

## 핵심 설계 원칙

- 모든 답변에 조문번호/판례번호 출처 부착
- 판례 유효성(폐기/개정)을 버리지 않고 경고와 함께 제시
- 단순 질의 2~4초 / 복잡 질의 최대 20초

## 구조

```
app/          FastAPI async 서버
retrieval/    임베딩·벡터·키워드검색·RRF·리랭킹·그래프확장
agent/        쿼리분해·도구라우팅·충분성루프·근거검증
reasoning/    LLM 클라이언트·답변 빌더
db/           PostgreSQL(pgvector) / Neo4j 커넥션
docs/         설계 문서
```

## 개발 도구

```bash
# LLM API 테스트 (대화형 — 매 턴 실제 요청 페이로드 출력)
python -m scripts.llm_test

# 1회 실행
python -m scripts.llm_test "소득세법 제14조 요지는?"

# provider/model 오버라이드
python -m scripts.llm_test --provider gemini --model gemini-2.5-pro
python -m scripts.llm_test --provider ollama --model qwen2.5:32b
```

## 문서

- [아키텍처](docs/architecture.md)
- [결정 기록](docs/decisions.md)
- [API 스펙](docs/spec.md)
- [설치/실행](docs/setup.md)
