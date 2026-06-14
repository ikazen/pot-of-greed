from fastapi import APIRouter, Depends

from app.auth.jwt import get_current_user
from app.db.pg import ping_pg
from app.db.neo4j import ping_neo4j

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(_: str = Depends(get_current_user)) -> dict:
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
    return {"pg": "ok" if pg_ok else "error", "neo4j": "ok" if neo4j_ok else "error"}
