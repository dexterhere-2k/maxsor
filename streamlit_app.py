import os

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT_SECONDS = 30

PRODUCT_TYPES = ["unknown", "non_food", "food", "mixed"]
OPENED_STATUSES = ["unknown", "unopened", "opened"]
ORDER_STATUSES = ["unknown", "processing", "confirmed", "dispatched", "delivered"]

st.set_page_config(page_title="Support Ticket Decisions", page_icon="🎫", layout="centered")


def call(method: str, path: str, authenticated: bool = False, **kwargs):
    headers = dict(kwargs.pop("headers", {}))
    if authenticated:
        headers["Authorization"] = f"Bearer {st.session_state['token']}"
    try:
        return requests.request(
            method, f"{API_BASE}{path}", headers=headers, timeout=TIMEOUT_SECONDS, **kwargs
        )
    except requests.RequestException as exc:
        st.error(
            f"The API is not reachable at {API_BASE}. Start it with "
            f"`uv run uvicorn src.api:app --port 8000`. ({exc})"
        )
        return None


def explain(response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = response.text
    if isinstance(detail, list):
        detail = "; ".join(str(item.get("msg", item)) for item in detail)
    return str(detail or f"HTTP {response.status_code}")


def show_decision(decision: dict) -> None:
    st.subheader(decision["action"])
    st.metric("Confidence", f"{decision['confidence']:.2f}")
    st.write(decision["reason"])
    sources = ", ".join(decision["sources"]) or "none"
    st.caption(f"sources: {sources}")
    answered_by = "the model" if decision["path"] == "cag" else "the offline rule engine"
    tokens = decision.get("prompt_tokens")
    st.caption(f"answered by {answered_by} (path={decision['path']})" + (f" · {tokens} prompt tokens" if tokens else ""))


if "token" not in st.session_state:
    st.session_state["token"] = None
    st.session_state["email"] = None

st.title("Support ticket decisions")

sign_in_tab, new_decision_tab, history_tab = st.tabs(["Sign in", "New decision", "History"])

with sign_in_tab:
    if st.session_state["token"]:
        st.success(f"Signed in as {st.session_state['email']}")
        if st.button("Sign out"):
            st.session_state["token"] = None
            st.session_state["email"] = None
            st.rerun()
    else:
        email = st.text_input("Email", key="auth_email")
        password = st.text_input("Password", type="password", key="auth_password")
        register_column, login_column = st.columns(2)

        if register_column.button("Register", use_container_width=True):
            response = call("POST", "/register", json={"email": email, "password": password})
            if response is not None:
                if response.status_code == 201:
                    st.success("Account created. You can sign in now.")
                else:
                    st.error(explain(response))

        if login_column.button("Sign in", use_container_width=True, type="primary"):
            response = call("POST", "/login", json={"email": email, "password": password})
            if response is not None:
                if response.status_code == 200:
                    st.session_state["token"] = response.json()["access_token"]
                    whoami = call("GET", "/me", authenticated=True)
                    st.session_state["email"] = whoami.json()["email"] if whoami is not None else email
                    st.rerun()
                else:
                    st.error(explain(response))

with new_decision_tab:
    if not st.session_state["token"]:
        st.info("Sign in to submit a ticket.")
    else:
        message = st.text_area(
            "Ticket message",
            height=120,
            placeholder="My order arrived damaged and the box is crushed.",
            help="Paste what the customer said. This is the only field you must fill in.",
        )

        with st.expander("Add order details (optional)"):
            st.caption(
                "A policy can only decide if it has the facts it needs. Leave anything "
                "you do not know blank, and the decision will say what it is missing."
            )
            left, middle, right = st.columns(3)
            order_value = left.number_input("Order value (₹)", min_value=0, value=None, step=100)
            days_delivered = middle.number_input("Days since delivery", min_value=0, value=None)
            days_dispatched = right.number_input("Days since dispatch", min_value=0, value=None)

            second_left, second_middle, second_right = st.columns(3)
            product_type = second_left.selectbox("Product type", PRODUCT_TYPES, index=0)
            opened_status = second_middle.selectbox("Opened?", OPENED_STATUSES, index=0)
            order_status = second_right.selectbox("Order status", ORDER_STATUSES, index=0)

        if st.button("Get decision", type="primary"):
            payload = {
                "message": message,
                "order_value_inr": order_value,
                "days_since_delivery": days_delivered,
                "days_since_dispatch": days_dispatched,
                "product_type": product_type,
                "opened_status": opened_status,
                "order_status": order_status,
            }
            response = call("POST", "/tickets", authenticated=True, json=payload)
            if response is not None:
                if response.status_code == 201:
                    ticket = response.json()
                    show_decision(ticket["decision"])
                    if ticket["decision"]["action"] == "NEEDS_MORE_INFORMATION":
                        st.info(
                            "Add the missing details above and submit again if you have them."
                        )
                    st.caption(f"ticket #{ticket['id']}, saved to your history")
                elif response.status_code == 503:
                    st.error(
                        "The model could not be reached, so no decision was made or stored. "
                        "A configured model is never silently replaced by the offline engine."
                    )
                else:
                    st.error(explain(response))

with history_tab:
    if not st.session_state["token"]:
        st.info("Sign in to see your history.")
    else:
        listed = call("GET", "/tickets", authenticated=True)
        tickets = listed.json() if listed is not None and listed.status_code == 200 else []
        if not tickets:
            st.write("No tickets yet.")
        else:
            st.write(f"{len(tickets)} ticket(s) on this account.")
            st.dataframe(
                [
                    {
                        "id": ticket["id"],
                        "action": (ticket["decision"] or {}).get("action", "—"),
                        "path": (ticket["decision"] or {}).get("path", "—"),
                        "message": ticket["message"][:70],
                        "created_at": ticket["created_at"],
                    }
                    for ticket in tickets
                ],
                use_container_width=True,
                hide_index=True,
            )

            chosen = st.selectbox("Open a ticket", [ticket["id"] for ticket in tickets])
            if st.button("View"):
                response = call("GET", f"/tickets/{chosen}", authenticated=True)
                if response is not None:
                    if response.status_code == 200:
                        detail = response.json()
                        st.write(detail["message"])
                        if detail["decision"]:
                            show_decision(detail["decision"])
                    else:
                        st.error(explain(response))
