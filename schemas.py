"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any
from datetime import datetime

# Example schemas (you can keep or remove later):

class User(BaseModel):
    """
    Users collection schema
    Collection name: "user" (lowercase of class name)
    """
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    address: str = Field(..., description="Address")
    age: Optional[int] = Field(None, ge=0, le=120, description="Age in years")
    is_active: bool = Field(True, description="Whether user is active")

class Product(BaseModel):
    """
    Products collection schema
    Collection name: "product" (lowercase of class name)
    """
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Product description")
    price: float = Field(..., ge=0, description="Price in dollars")
    category: str = Field(..., description="Product category")
    in_stock: bool = Field(True, description="Whether product is in stock")

# Sandbox API Platform Schemas

class ApiKey(BaseModel):
    """
    API keys for authenticating requests in the sandbox
    Collection: "apikey"
    """
    name: str = Field(..., description="Label for this key, e.g., 'Dev Laptop' or 'Server'")
    key: str = Field(..., description="Secret key value (shown once on create)")
    status: Literal["active", "revoked"] = Field("active", description="Key status")
    permissions: List[str] = Field(default_factory=lambda: ["invoke"], description="Allowed operations")
    last_used: Optional[datetime] = Field(None, description="Last time this key was used")

class Usagelog(BaseModel):
    """
    Usage logs for API invocations
    Collection: "usagelog"
    """
    api_key_id: Optional[str] = Field(None, description="Reference to the ApiKey document ID")
    api_key_name: Optional[str] = Field(None, description="Human label for the key at time of use")
    model: str = Field(..., description="Model identifier")
    prompt_chars: int = Field(..., ge=0, description="Character count of prompt")
    tokens_in: int = Field(..., ge=0, description="Estimated input tokens")
    tokens_out: int = Field(..., ge=0, description="Estimated output tokens")
    latency_ms: int = Field(..., ge=0, description="End-to-end latency in ms")
    status: Literal["success", "error"] = Field("success", description="Invocation status")
    error: Optional[str] = Field(None, description="Error message if any")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata like temperature, etc.")

# Add your own schemas here:
# --------------------------------------------------

# Note: The Flames database viewer will automatically:
# 1. Read these schemas from GET /schema endpoint
# 2. Use them for document validation when creating/editing
# 3. Handle all database operations (CRUD) directly
# 4. You don't need to create any database endpoints!
