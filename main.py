import os
import time
import secrets
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Header, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import db, create_document, get_documents

app = FastAPI(title="AeroMind Sandbox API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"

            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


# -----------------------------
# Sandbox: API Keys & Usage
# -----------------------------

class CreateKeyRequest(BaseModel):
    name: str = Field(..., description="Label for this key")
    permissions: Optional[List[str]] = Field(default_factory=lambda: ["invoke"]) 

class ApiKeyPublic(BaseModel):
    id: str
    name: str
    status: str
    permissions: List[str]
    last_used: Optional[str] = None
    created_at: Optional[str] = None

class InvokeRequest(BaseModel):
    model: str = Field("aeromind-small", description="Model identifier")
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 256
    metadata: Dict[str, Any] = Field(default_factory=dict)

class InvokeResponse(BaseModel):
    model: str
    output: str
    usage: Dict[str, int]
    latency_ms: int


def _mask_key(key: str) -> str:
    if not key:
        return ""
    return key[:4] + "••••" + key[-4:]


@app.get("/api/keys", response_model=List[ApiKeyPublic])
def list_keys():
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    items = db.apikey.find({}, {"key": 0})  # don't expose raw key
    results = []
    for it in items:
        results.append(ApiKeyPublic(
            id=str(it.get("_id")),
            name=it.get("name"),
            status=it.get("status", "active"),
            permissions=it.get("permissions", []),
            last_used=(it.get("last_used").isoformat() if it.get("last_used") else None),
            created_at=(it.get("created_at").isoformat() if it.get("created_at") else None),
        ))
    return results


@app.post("/api/keys")
def create_key(payload: CreateKeyRequest):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    # Generate a 40-char URL-safe token
    raw_key = "sk-" + secrets.token_urlsafe(30)
    doc = {
        "name": payload.name,
        "key": raw_key,
        "status": "active",
        "permissions": payload.permissions or ["invoke"],
    }
    inserted_id = create_document("apikey", doc)
    return {
        "id": inserted_id,
        "name": payload.name,
        "key": raw_key,  # show once
        "status": "active",
        "permissions": payload.permissions or ["invoke"],
    }


@app.post("/api/keys/{key_id}/revoke")
def revoke_key(key_id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    from bson import ObjectId
    try:
        result = db.apikey.update_one({"_id": ObjectId(key_id)}, {"$set": {"status": "revoked"}})
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Key not found")
        return {"id": key_id, "status": "revoked"}
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid key id")


# Dependency to authenticate key
async def get_active_api_key(authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    key_val = None
    if authorization and authorization.lower().startswith("bearer "):
        key_val = authorization.split(" ", 1)[1].strip()
    elif x_api_key:
        key_val = x_api_key.strip()
    if not key_val:
        raise HTTPException(status_code=401, detail="Missing API key")

    rec = db.apikey.find_one({"key": key_val})
    if not rec:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if rec.get("status") != "active":
        raise HTTPException(status_code=403, detail="API key is not active")
    return rec


@app.get("/api/usage")
def list_usage(limit: int = Query(50, ge=1, le=500)):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    cur = db.usagelog.find({}, {"prompt": 0}).sort("created_at", -1).limit(limit)
    items = []
    for it in cur:
        it["id"] = str(it.pop("_id"))
        # convert datetimes
        if it.get("created_at"):
            it["created_at"] = it["created_at"].isoformat()
        if it.get("updated_at"):
            it["updated_at"] = it["updated_at"].isoformat()
        items.append(it)
    return {"items": items}


@app.post("/api/invoke", response_model=InvokeResponse)
async def invoke(req: InvokeRequest, key_rec: Dict[str, Any] = Depends(get_active_api_key)):
    start = time.time()

    # Very simple mock generation: echo with slight transform
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    # Fake token accounting
    tokens_in = max(1, len(prompt) // 4)
    generated = ("Answer: " + prompt[:200])
    if len(generated) > req.max_tokens:
        generated = generated[: req.max_tokens]
    tokens_out = max(1, len(generated) // 4)

    latency_ms = int((time.time() - start) * 1000) + 40  # add a constant to feel realistic

    # Log usage
    try:
        log = {
            "api_key_id": str(key_rec.get("_id")),
            "api_key_name": key_rec.get("name"),
            "model": req.model,
            "prompt_chars": len(prompt),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency_ms,
            "status": "success",
            "error": None,
            "meta": {"temperature": req.temperature, **(req.metadata or {})},
        }
        create_document("usagelog", log)
        # update last_used on key
        from bson import ObjectId
        db.apikey.update_one({"_id": ObjectId(str(key_rec.get("_id")))}, {"$set": {"last_used": db.command({'isMaster': 1}) and None}})
        # simpler: just set to current time using Python if DB command fails
        try:
            from datetime import datetime, timezone
            db.apikey.update_one({"_id": ObjectId(str(key_rec.get("_id")))}, {"$set": {"last_used": datetime.now(timezone.utc)}})
        except Exception:
            pass
    except Exception:
        pass

    return InvokeResponse(
        model=req.model,
        output=generated,
        usage={"input_tokens": tokens_in, "output_tokens": tokens_out, "total_tokens": tokens_in + tokens_out},
        latency_ms=latency_ms,
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
