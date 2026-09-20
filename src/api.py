from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, status

from . import auth, config, database, decision, models

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    database.init_db()
    yield

app = FastAPI(
    title="Support Ticket Decision Assistant",
    version="0.1.0",
    description=(
        "Submits a support ticket and returns a structured, policy-grounded decision. "
        "Decisions are stored per user; `path` records whether the model or the offline "
        "rule engine produced each one."
    ),
    lifespan=lifespan,
)

@app.post("/register", response_model=models.UserOut, status_code=status.HTTP_201_CREATED)
def register(payload: models.RegisterIn) -> models.UserOut:
    if database.get_user_by_email(payload.email) is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "An account with that email already exists"
        )
    user = database.create_user(payload.email, auth.hash_password(payload.password))
    return models.UserOut(**user)

@app.post("/login", response_model=models.TokenOut)
def login(payload: models.LoginIn) -> models.TokenOut:
    user = database.get_user_by_email(payload.email)
    if user is None or not auth.verify_password(payload.password, user["password_hash"]):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return models.TokenOut(
        access_token=auth.create_token(user["id"]),
        expires_in=config.TOKEN_TTL_HOURS * 3600,
    )

@app.get("/me", response_model=models.UserOut)
def me(user: dict[str, Any] = Depends(auth.current_user)) -> models.UserOut:
    return models.UserOut(**user)

@app.post("/tickets", response_model=models.TicketOut, status_code=status.HTTP_201_CREATED)
def submit_ticket(
    payload: models.TicketIn, user: dict[str, Any] = Depends(auth.current_user)
) -> models.TicketOut:
    ticket = payload.model_dump()
    try:
        served = decision.decide(ticket)
    except decision.ModelUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    stored = database.insert_ticket(user["id"], ticket, served.as_record())
    return models.TicketOut(**stored)

@app.get("/tickets", response_model=list[models.TicketOut])
def list_tickets(user: dict[str, Any] = Depends(auth.current_user)) -> list[models.TicketOut]:
    return [models.TicketOut(**row) for row in database.list_tickets(user["id"])]

@app.get("/tickets/{ticket_id}", response_model=models.TicketOut)
def get_ticket(
    ticket_id: int, user: dict[str, Any] = Depends(auth.current_user)
) -> models.TicketOut:
    row = database.get_ticket(user["id"], ticket_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Ticket not found")
    return models.TicketOut(**row)
