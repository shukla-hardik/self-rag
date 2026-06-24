from fastapi import APIRouter, Security
from fastapi.security import APIKeyHeader

from app.constants import RouteType
from .health import health_router
from .document import document_router
from .thread import thread_router

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

v1_api_router = APIRouter(prefix="/api/v1")

# PUBLIC — no auth required
public_router = APIRouter(prefix=f"/{RouteType.PUBLIC}")

# PRIVATE — Bearer token required (enforced by AuthMiddleware)
private_router = APIRouter(
    prefix=f"/{RouteType.PRIVATE}", dependencies=[Security(api_key_header)]
)
private_router.include_router(document_router, prefix="/document",
                              tags=["Documents"])
private_router.include_router(thread_router, prefix="/thread",
                              tags=["Threads"])

# ADMIN — Bearer token + admin role (enforced by AuthMiddleware)
admin_router = APIRouter(
    prefix=f"/{RouteType.ADMIN}", dependencies=[Security(api_key_header)]
)

# INTERNAL — static INTERNAL_TOKEN header (enforced by AuthMiddleware)
internal_router = APIRouter(
    prefix=f"/{RouteType.INTERNAL}", dependencies=[Security(api_key_header)]
)

v1_api_router.include_router(public_router)
v1_api_router.include_router(private_router)
v1_api_router.include_router(admin_router)
v1_api_router.include_router(internal_router)
v1_api_router.include_router(health_router, tags=["Health"])
