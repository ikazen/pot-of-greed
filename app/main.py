from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.db.pg import init_pg, close_pg
from app.db.neo4j import init_neo4j, close_neo4j
from app.auth.routes import router as auth_router
from app.api.health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    await init_pg(settings.pg_dsn)
    await init_neo4j(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    yield
    await close_pg()
    await close_neo4j()


app = FastAPI(title="pot-of-greed", lifespan=lifespan)
app.include_router(auth_router)
app.include_router(health_router)
