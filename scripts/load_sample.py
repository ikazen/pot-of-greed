"""
샘플 데이터 로더 — 개발/검증용 소량 데이터 적재.

조문 4건, 판례 4건, 관계(CITES/BASED_ON/OVERRULED_BY) 적재.
임베딩은 NULL 유지 — BON-139에서 백필.
tsv는 트리거가 자동으로 채운다.

사용:
    python scripts/load_sample.py
"""

import asyncio
import sys
from datetime import date
from pathlib import Path

import asyncpg
from neo4j import AsyncGraphDatabase

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import get_settings

# 조 레벨 parent 청크 (small-to-big 의 big). clause_path=None, parent_chunk_id=None.
# 그래프(관계 전용)에는 넣지 않고 PG 에만 적재 — 계층은 parent_chunk_id fetch 로만 표현(결정 I).
PARENT_ARTICLES = [
    {
        "chunk_id": "art_소득세법_14",
        "law_name": "소득세법",
        "article_no": "제14조",
        "clause_path": None,
        "parent_chunk_id": None,
        "text": "제14조(과세표준의 계산) 거주자의 종합소득에 대한 과세표준은 종합소득금액에서 종합소득공제를 적용한 금액으로 한다. 종합소득금액은 이자소득·배당소득·사업소득·근로소득·연금소득·기타소득의 합계액으로 한다.",
        "effective_from": "2010-01-01",
        "effective_to": None,
        "is_current": True,
    },
    {
        "chunk_id": "art_법인세법_52",
        "law_name": "법인세법",
        "article_no": "제52조",
        "clause_path": None,
        "parent_chunk_id": None,
        "text": "제52조(부당행위계산의 부인) 납세지 관할 세무서장 또는 관할 지방국세청장은 내국법인의 행위 또는 소득금액의 계산이 특수관계인과의 거래로 인하여 조세의 부담을 부당하게 감소시킨 것으로 인정되는 경우 그 법인의 각 사업연도 소득금액을 다시 계산할 수 있다. 부당행위계산의 유형과 시가의 범위는 대통령령으로 정한다.",
        "effective_from": "2019-01-01",
        "effective_to": None,
        "is_current": True,
    },
    {
        "chunk_id": "art_법인세법_52_old",
        "law_name": "법인세법",
        "article_no": "제52조",
        "clause_path": None,
        "parent_chunk_id": None,
        "text": "제52조(부당행위계산의 부인) [2018-12-31 이전 시행] 납세지 관할 세무서장 또는 관할 지방국세청장은 내국법인의 행위 또는 소득금액의 계산이 특수관계인과의 거래로 인하여 조세의 부담을 부당하게 감소시킨 것으로 인정되는 경우 그 법인의 각 사업연도 소득금액을 다시 계산할 수 있다.",
        "effective_from": "2010-01-01",
        "effective_to": "2018-12-31",
        "is_current": False,
    },
    {
        "chunk_id": "art_부가세법_26",
        "law_name": "부가가치세법",
        "article_no": "제26조",
        "clause_path": None,
        "parent_chunk_id": None,
        "text": "제26조(재화 또는 용역의 공급에 대한 면세) 다음 각 호의 재화 또는 용역의 공급에 대하여는 부가가치세를 면제한다. 면세 대상의 구체적 범위는 대통령령으로 정한다.",
        "effective_from": "2013-07-01",
        "effective_to": None,
        "is_current": True,
    },
]

ARTICLES = [
    {
        "chunk_id": "art_소득세법_14_1",
        "law_name": "소득세법",
        "article_no": "제14조",
        "clause_path": "제1항",
        "parent_chunk_id": "art_소득세법_14",
        "text": "거주자의 종합소득에 대한 과세표준은 다음 각 호의 소득의 합계액으로 한다.",
        "effective_from": "2010-01-01",
        "effective_to": None,
        "is_current": True,
    },
    {
        "chunk_id": "art_법인세법_52_1",
        "law_name": "법인세법",
        "article_no": "제52조",
        "clause_path": "제1항",
        "parent_chunk_id": "art_법인세법_52",
        "text": "납세지 관할 세무서장 또는 관할 지방국세청장은 내국법인의 행위 또는 소득금액의 계산이 특수관계인과의 거래로 인하여 그 법인의 소득에 대한 조세의 부담을 부당하게 감소시킨 것으로 인정되는 경우에는 그 법인의 행위 또는 소득금액의 계산에 관계없이 그 법인의 각 사업연도의 소득금액을 계산할 수 있다.",
        "effective_from": "2019-01-01",
        "effective_to": None,
        "is_current": True,
    },
    {
        "chunk_id": "art_법인세법_52_1_old",
        "law_name": "법인세법",
        "article_no": "제52조",
        "clause_path": "제1항",
        "parent_chunk_id": "art_법인세법_52_old",
        "text": "납세지 관할 세무서장 또는 관할 지방국세청장은 내국법인의 행위 또는 소득금액의 계산이 특수관계인과의 거래로 인하여 그 법인의 소득에 대한 조세의 부담을 부당하게 감소시킨 것으로 인정되는 경우에는 그 법인의 각 사업연도의 소득금액을 계산할 수 있다.",
        "effective_from": "2010-01-01",
        "effective_to": "2018-12-31",
        "is_current": False,
    },
    {
        "chunk_id": "art_부가세법_26_1",
        "law_name": "부가가치세법",
        "article_no": "제26조",
        "clause_path": "제1항",
        "parent_chunk_id": "art_부가세법_26",
        "text": "다음 각 호의 재화 또는 용역의 공급에 대하여는 부가가치세를 면제한다.",
        "effective_from": "2013-07-01",
        "effective_to": None,
        "is_current": True,
    },
]

CASES = [
    {
        "chunk_id": "case_2018두12345",
        "case_no": "2018두12345",
        "court": "대법원",
        "decided_at": "2020-03-15",
        "is_en_banc": False,
        "validity_flag": "law_amended",
        "text": "법인세법 제52조 제1항의 부당행위계산 부인 규정 적용에 있어서 특수관계인 간 거래가격이 시가와 다르다는 사정만으로 곧바로 부당행위계산 부인 대상이 되는 것은 아니고, 조세의 부담을 부당히 감소시킨 것으로 인정되어야 한다.",
    },
    {
        "chunk_id": "case_2015두54321",
        "case_no": "2015두54321",
        "court": "대법원",
        "decided_at": "2017-06-20",
        "is_en_banc": False,
        "validity_flag": "overruled",
        "text": "특수관계인과의 거래에서 시가와 거래가액의 차이가 있으면 원칙적으로 부당행위계산 부인 대상에 해당한다.",
    },
    {
        "chunk_id": "case_2020두99999",
        "case_no": "2020두99999",
        "court": "대법원",
        "decided_at": "2022-11-10",
        "is_en_banc": True,
        "validity_flag": "valid",
        "text": "부당행위계산 부인의 요건으로서 '조세의 부담을 부당하게 감소시킨 것'은 건전한 사회통념이나 상관행에 비추어 경제적 합리성을 결한 비정상적인 것임을 요한다. (전원합의체, 2018두12345 판시 확인)",
    },
    {
        "chunk_id": "case_2021나11111",
        "case_no": "2021나11111",
        "court": "서울고등법원",
        "decided_at": "2022-04-05",
        "is_en_banc": False,
        "validity_flag": "valid",
        "text": "소득세법 제14조 제1항에서 규정한 종합소득 합산 과세의 적용 범위에 관하여, 비거주자의 국내 원천소득은 별도 과세 원칙이 적용된다.",
    },
]

# (case_chunk_id, article_chunk_id) CITES
CITES_ARTICLE = [
    ("case_2018두12345", "art_법인세법_52_1_old"),
    ("case_2015두54321", "art_법인세법_52_1_old"),
    ("case_2020두99999", "art_법인세법_52_1"),
    ("case_2021나11111", "art_소득세법_14_1"),
]

# (case_chunk_id, article_chunk_id) BASED_ON
BASED_ON = [
    ("case_2018두12345", "art_법인세법_52_1_old"),
    ("case_2020두99999", "art_법인세법_52_1"),
]

# (newer_case_chunk_id, older_case_chunk_id) newer OVERRULED_BY -> older에서 newer로 방향 주의
# 설계: (:Case)-[:OVERRULED_BY]->(:Case) — 구판례가 신판례에 의해 폐기
# 2015두54321 이 2020두99999 에 의해 폐기됨
OVERRULED_BY = [("case_2015두54321", "case_2020두99999")]

# 법인세법 제52조 구조문(art_법인세법_52_1_old)이 2019년 개정(AMENDED_BY)
AMENDMENTS = [
    {
        "amendment_id": "amend_법인세법_52_2019",
        "law_name": "법인세법",
        "article_no": "제52조",
        "amended_at": "2019-01-01",
        "summary": "부당행위계산 부인 요건 명확화 — '행위 또는 소득금액의 계산에 관계없이' 문구 추가",
    }
]
AMENDED_BY = [("art_법인세법_52_1_old", "amend_법인세법_52_2019")]


async def load_pg(dsn: str) -> None:
    schema_sql = (Path(__file__).parent.parent / "sql" / "schema.sql").read_text()
    conn = await asyncpg.connect(dsn=dsn)
    try:
        await conn.execute(schema_sql)
        for art in PARENT_ARTICLES + ARTICLES:
            await conn.execute(
                """
                INSERT INTO article_chunks
                    (chunk_id, law_name, article_no, clause_path, parent_chunk_id,
                     text, effective_from, effective_to, is_current)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                ON CONFLICT (chunk_id) DO NOTHING
                """,
                art["chunk_id"], art["law_name"], art["article_no"],
                art["clause_path"], art["parent_chunk_id"], art["text"],
                date.fromisoformat(art["effective_from"]),
                date.fromisoformat(art["effective_to"]) if art["effective_to"] else None,
                art["is_current"],
            )
        for case in CASES:
            await conn.execute(
                """
                INSERT INTO case_chunks
                    (chunk_id, case_no, court, decided_at, is_en_banc, validity_flag, text)
                VALUES ($1,$2,$3,$4,$5,$6,$7)
                ON CONFLICT (chunk_id) DO NOTHING
                """,
                case["chunk_id"], case["case_no"], case["court"],
                date.fromisoformat(case["decided_at"]),
                case["is_en_banc"], case["validity_flag"], case["text"],
            )
        art_count = await conn.fetchval("SELECT count(*) FROM article_chunks")
        case_count = await conn.fetchval("SELECT count(*) FROM case_chunks")
        print(f"PG: article_chunks={art_count}, case_chunks={case_count}")
    finally:
        await conn.close()


async def load_neo4j(uri: str, user: str, password: str) -> None:
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        async with driver.session() as session:
            # 스키마 cypher 실행
            schema_cypher = (Path(__file__).parent.parent / "sql" / "neo4j_schema.cypher").read_text()
            for stmt in schema_cypher.split(";"):
                stmt = stmt.strip()
                if stmt and not stmt.startswith("//"):
                    await session.run(stmt)

            for art in ARTICLES:
                await session.run(
                    """
                    MERGE (n:PotOfGreedArticle {chunk_id: $chunk_id})
                    SET n.law_name = $law_name,
                        n.article_no = $article_no,
                        n.effective_from = $effective_from,
                        n.effective_to = $effective_to,
                        n.is_current = $is_current
                    """,
                    chunk_id=art["chunk_id"], law_name=art["law_name"],
                    article_no=art["article_no"], effective_from=str(art["effective_from"]),
                    effective_to=str(art["effective_to"]) if art["effective_to"] else None,
                    is_current=art["is_current"],
                )

            for case in CASES:
                await session.run(
                    """
                    MERGE (n:PotOfGreedCase {chunk_id: $chunk_id})
                    SET n.case_no = $case_no,
                        n.court = $court,
                        n.decided_at = $decided_at,
                        n.is_en_banc = $is_en_banc,
                        n.validity_flag = $validity_flag
                    """,
                    chunk_id=case["chunk_id"], case_no=case["case_no"],
                    court=case["court"], decided_at=str(case["decided_at"]),
                    is_en_banc=case["is_en_banc"], validity_flag=case["validity_flag"],
                )

            for amend in AMENDMENTS:
                await session.run(
                    """
                    MERGE (n:PotOfGreedAmendment {amendment_id: $amendment_id})
                    SET n.law_name = $law_name,
                        n.article_no = $article_no,
                        n.amended_at = $amended_at,
                        n.summary = $summary
                    """,
                    **amend,
                )

            for case_id, art_id in CITES_ARTICLE:
                await session.run(
                    """
                    MATCH (c:PotOfGreedCase {chunk_id: $cid})
                    MATCH (a:PotOfGreedArticle {chunk_id: $aid})
                    MERGE (c)-[:CITES]->(a)
                    """,
                    cid=case_id, aid=art_id,
                )

            for case_id, art_id in BASED_ON:
                await session.run(
                    """
                    MATCH (c:PotOfGreedCase {chunk_id: $cid})
                    MATCH (a:PotOfGreedArticle {chunk_id: $aid})
                    MERGE (c)-[:BASED_ON]->(a)
                    """,
                    cid=case_id, aid=art_id,
                )

            for older_case, newer_case in OVERRULED_BY:
                await session.run(
                    """
                    MATCH (old:PotOfGreedCase {chunk_id: $old_id})
                    MATCH (new:PotOfGreedCase {chunk_id: $new_id})
                    MERGE (old)-[:OVERRULED_BY]->(new)
                    """,
                    old_id=older_case, new_id=newer_case,
                )

            for art_id, amend_id in AMENDED_BY:
                await session.run(
                    """
                    MATCH (a:PotOfGreedArticle {chunk_id: $aid})
                    MATCH (m:PotOfGreedAmendment {amendment_id: $mid})
                    MERGE (a)-[:AMENDED_BY]->(m)
                    """,
                    aid=art_id, mid=amend_id,
                )

            result = await session.run(
                "MATCH (n) WHERE n:PotOfGreedArticle OR n:PotOfGreedCase OR n:PotOfGreedAmendment RETURN count(n) AS cnt"
            )
            record = await result.single()
            print(f"Neo4j: total PotOfGreed nodes={record['cnt']}")
    finally:
        await driver.close()


async def main() -> None:
    settings = get_settings()
    print("Loading PG...")
    await load_pg(settings.pg_dsn)
    print("Loading Neo4j...")
    await load_neo4j(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
