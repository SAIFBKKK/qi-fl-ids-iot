from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/nodes")
async def api_nodes(request: Request) -> dict:
    try:
        response = await request.app.state.http.get(f"{request.app.state.fl_server_url}/nodes")
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text) from exc
    except httpx.RequestError as exc:
        raise HTTPException(502, f"fl-server unreachable: {exc}") from exc
