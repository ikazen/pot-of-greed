from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.auth.jwt import get_current_user
from app.db.pg import ping_pg
from app.db.neo4j import ping_neo4j

router = APIRouter(tags=["health"])


async def _check_db_health() -> dict[str, bool]:
    pg_ok = False
    neo4j_ok = False
    try:
        pg_ok = await ping_pg()
    except Exception:
        pass
    try:
        neo4j_ok = await ping_neo4j()
    except Exception:
        pass
    return {"pg": pg_ok, "neo4j": neo4j_ok}


@router.get("/health")
async def health(_: str = Depends(get_current_user)) -> dict:
    status = await _check_db_health()
    return {k: "ok" if v else "error" for k, v in status.items()}


@router.get("/healthz")
async def healthz() -> JSONResponse:
    """무인증 헬스체크. compose/오케스트레이터가 실제 DB ping 결과로 컨테이너
    상태를 판단할 수 있도록 인증 없이 노출한다(#10) — 200/503으로 성공 여부까지 표현.
    """
    status = await _check_db_health()
    ok = all(status.values())
    return JSONResponse(
        status_code=200 if ok else 503,
        content={k: "ok" if v else "error" for k, v in status.items()},
    )
