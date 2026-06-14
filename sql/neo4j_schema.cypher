// Neo4j 제약 + 인덱스 (PotOfGreed 레이블 네임스페이스)
// Community Edition: 단일 그래프 — PotOfGreed 레이블로 워크로드 격리

// 고유 제약 (자동 인덱스 포함)
CREATE CONSTRAINT poc_article_chunk_id IF NOT EXISTS
    FOR (n:PotOfGreedArticle) REQUIRE n.chunk_id IS UNIQUE;

CREATE CONSTRAINT poc_case_chunk_id IF NOT EXISTS
    FOR (n:PotOfGreedCase) REQUIRE n.chunk_id IS UNIQUE;

CREATE CONSTRAINT poc_amendment_id IF NOT EXISTS
    FOR (n:PotOfGreedAmendment) REQUIRE n.amendment_id IS UNIQUE;

// 조회용 인덱스
CREATE INDEX poc_article_law_article IF NOT EXISTS
    FOR (n:PotOfGreedArticle) ON (n.law_name, n.article_no);

CREATE INDEX poc_case_case_no IF NOT EXISTS
    FOR (n:PotOfGreedCase) ON (n.case_no);

CREATE INDEX poc_case_validity IF NOT EXISTS
    FOR (n:PotOfGreedCase) ON (n.validity_flag);
