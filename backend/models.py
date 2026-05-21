from pydantic import BaseModel, EmailStr, ConfigDict
from typing import Optional, List
from datetime import datetime

# =========================
# USER MODELS
# =========================

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str


class UserResponse(BaseModel):

    model_config = ConfigDict(
        from_attributes=True
    )

    id: str
    name: str
    email: EmailStr
    image: Optional[str] = None


class UserInDB(UserResponse):
    hashed_password: str


# =========================
# AUTH MODELS
# =========================

class Token(BaseModel):
    access_token: str
    token_type: str


# =========================
# CHAT MODELS
# =========================

class ChatQuery(BaseModel):
    query: str


class SourceResponse(BaseModel):
    source: str
    page: Optional[int] = None


class ChatResponse(BaseModel):
    response: str
    sources: List[SourceResponse] = []


# =========================
# DOCUMENT MODELS
# =========================

class SummarizeRequest(BaseModel):
    document_id: str


# OPTIONAL FUTURE MODEL
# =========================

class DocumentResponse(BaseModel):

    model_config = ConfigDict(
        from_attributes=True
    )

    id: str
    filename: str
    uploaded_at: datetime