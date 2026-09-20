from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from . import config

Action = Literal[tuple(sorted(config.ACTIONS))]

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

CATEGORICAL_FIELDS = ("product_type", "opened_status", "order_status")
NUMERIC_FIELDS = ("order_value_inr", "days_since_delivery", "days_since_dispatch")
UNKNOWN = "unknown"

class RegisterIn(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=72)

    @field_validator("email")
    @classmethod
    def _valid_email(cls, value: str) -> str:
        value = value.strip().lower()
        if not _EMAIL_RE.match(value):
            raise ValueError("not a valid email address")
        return value

class LoginIn(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, value: str) -> str:
        return value.strip().lower()

class TicketIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    order_value_inr: float | None = Field(default=None, ge=0)
    days_since_delivery: int | None = Field(default=None, ge=0)
    days_since_dispatch: int | None = Field(default=None, ge=0)
    product_type: str | None = None
    opened_status: str | None = None
    order_status: str | None = None

    @field_validator("message")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("message must not be blank")
        return text

    @field_validator(*CATEGORICAL_FIELDS, mode="before")
    @classmethod
    def _normalise_category(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        return text or None

class Decision(BaseModel):
    action: Action
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    sources: list[str] = Field(default_factory=list)

    @field_validator("action", mode="before")
    @classmethod
    def _known_action(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip().upper()
        if value not in config.ACTIONS:
            raise ValueError(
                f"{value!r} is not one of the {len(config.ACTIONS)} permitted actions"
            )
        return value

class DecisionOut(Decision):
    path: str
    prompt_tokens: int | None = None
    context_signal: float | None = None

class TicketOut(BaseModel):
    id: int
    message: str
    order_value_inr: float | None = None
    days_since_delivery: int | None = None
    days_since_dispatch: int | None = None
    product_type: str | None = None
    opened_status: str | None = None
    order_status: str | None = None
    created_at: str
    decision: DecisionOut | None = None

class UserOut(BaseModel):
    id: int
    email: str
    created_at: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

if __name__ == "__main__":
    from pydantic import ValidationError

    Decision(action=next(iter(config.ACTIONS)), confidence=0.5, reason="self check", sources=[])
    try:
        Decision(action="NOT_A_REAL_ACTION", confidence=0.5, reason="self check", sources=[])
    except ValidationError:
        pass
    else:
        raise AssertionError("an action outside the vocabulary must be rejected")
    try:
        TicketIn(message="   ")
    except ValidationError:
        pass
    else:
        raise AssertionError("a blank message must be rejected")
    print(f"{len(config.ACTIONS)} actions accepted, everything else refused")
