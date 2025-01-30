from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

class LargeFileMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "POST":
            if "content-length" in request.headers:
                content_length = int(request.headers["content-length"])
                # Set max size to 1GB (1024 * 1024 * 1024 bytes)
                if content_length > 1024 * 1024 * 1024:
                    raise HTTPException(
                        status_code=413,
                        detail="File too large. Maximum size is 1GB"
                    )
        
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return Response(
                content=str(e),
                status_code=500
            ) 