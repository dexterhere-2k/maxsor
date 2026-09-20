import os

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT_SECONDS = 30

PRODUCT_TYPES = ["unknown", "non_food", "food", "mixed"]
OPENED_STATUSES = ["unknown", "unopened", "opened"]
ORDER_STATUSES = ["unknown", "processing", "confirmed", "dispatched", "delivered"]

BLOCK_LABELS = {
    "APPROVE_REFUND_OR_REPLACEMENT": "green",
    "APPROVE_RETURN": "green",
    "APPROVE_CANCELLATION": "green",
    "REJECT_OUTSIDE_WINDOW": "red",
    "REJECT_NO_PROOF": "red",
    "REJECT_REASON": "red",
    "REQUEST_PHOTOS": "orange",
    "REQUEST_DEFECT_EVIDENCE": "orange",
    "REQUEST_MORE_PROOF": "orange",
}

st.set_page_config(page_title="Support Ticket Decisions", page_icon="✳", layout="centered")

CSS = """
    <style>
        .block-container { padding-top: 3rem; padding-bottom: 5rem; max-width: 46rem; }
        h1, h2, h3 { letter-spacing: -0.02em; }
        .stTabs [data-baseweb="tab-list"] { gap: 1.75rem; border-bottom: none; }
        .stTabs [data-baseweb="tab"] { padding: 0.3rem 0; }
        .stTabs [aria-selected="true"] h5 { color: #2f6d5f; }
        .stTabs [data-baseweb="tab-highlight"] { background-color: #2f6d5f; }
        textarea::placeholder { color: #b9b4ab; }
        .stTextArea textarea { background-color: #ffffff; }
        .stMetric { background: #f0eee9; border-radius: 12px; padding: 1rem 1.25rem; }
        .stMetric label { color: #6b6459; }
        .stMetric > div { font-variant-numeric: tabular-nums; }
    </style>
"""
st.markdown(CSS, unsafe_allow_html=True)


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


def submit_ticket() -> None:
    payload = {
        "message": st.session_state["ticket_message"],
        "order_value_inr": st.session_state["order_value"],
        "days_since_delivery": st.session_state["days_delivered"],
        "days_since_dispatch": st.session_state["days_dispatched"],
        "product_type": st.session_state["product_type"],
        "opened_status": st.session_state["opened_status"],
        "order_status": st.session_state["order_status"],
    }
    st.session_state["last_submission"] = None
    response = call("POST", "/tickets", authenticated=True, json=payload)
    if response is None:
        return
    if response.status_code == 201:
        ticket = response.json()
        st.session_state["last_submission"] = {"decision": ticket["decision"], "id": ticket["id"]}
    elif response.status_code == 503:
        st.session_state["last_submission"] = {"unavailable": True}
    else:
        st.session_state["last_submission"] = {"error": explain(response), "status": response.status_code}


def show_decision(decision: dict) -> None:
    tone = BLOCK_LABELS.get(decision["action"], "blue")
    icon = {"green": "✅", "red": "⛔", "orange": "📸", "blue": "❔"}[tone]
    st.markdown(f":{tone}[{icon} {decision['action']}]")
    st.metric("Confidence", f"{decision['confidence']:.2f}")
    st.write(decision["reason"])
    sources = ", ".join(decision["sources"]) or "none"
    st.caption(f"sources: {sources}")
    answered_by = "the model" if decision["path"] == "cag" else "the offline rule engine"
    tokens = decision.get("prompt_tokens")
    st.caption(
        f"answered by {answered_by} · path=`{decision['path']}`"
        + (f" · {tokens} prompt tokens" if tokens else "")
    )


if "token" not in st.session_state:
    st.session_state["token"] = None
    st.session_state["email"] = None

st.title("Support ticket decisions")
st.caption(
    "Paste what the customer said. The policy corpus decides the call — every answer "
    "names the policy it leaned on."
)
sign_in_tab, new_decision_tab, history_tab = st.tabs(["Sign in", "New decision", "History"])

with sign_in_tab:
    if st.session_state["token"]:
        st.success(f"Signed in as {st.session_state['email']}")
        if st.button("Sign out"):
            st.session_state["token"] = None
            st.session_state["email"] = None
            st.rerun()
    else:
        st.caption("New here? Create an account. Returning? Just sign in.")
        email = st.text_input("Email", key="auth_email", placeholder="you@company.com")
        password = st.text_input(
            "Password", type="password", key="auth_password", placeholder="at least 8 characters"
        )
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
        st.text_area(
            "Ticket message",
            key="ticket_message",
            height=140,
            placeholder="My order arrived damaged and the box is crushed.",
            help="Paste what the customer said. This is the only field you must fill in.",
        )

        with st.expander("Add order details (optional)"):
            st.caption(
                "A policy can only decide if it has the facts it needs. Leave anything "
                "you do not know blank, and the decision will say what it is missing."
            )
            left, middle, right = st.columns(3)
            left.number_input("Order value (₹)", key="order_value", min_value=0, value=None, step=100)
            middle.number_input("Days since delivery", key="days_delivered", min_value=0, value=None)
            right.number_input("Days since dispatch", key="days_dispatched", min_value=0, value=None)

            second_left, second_middle, second_right = st.columns(3)
            second_left.selectbox("Product type", PRODUCT_TYPES, key="product_type", index=0)
            second_middle.selectbox("Opened?", OPENED_STATUSES, key="opened_status", index=0)
            second_right.selectbox("Order status", ORDER_STATUSES, key="order_status", index=0)

        st.button(
            "Get decision",
            type="primary",
            use_container_width=True,
            on_click=submit_ticket,
        )
        submission = st.session_state.get("last_submission")
        if submission is not None:
            if "decision" in submission:
                show_decision(submission["decision"])
                if submission["decision"]["action"] == "NEEDS_MORE_INFORMATION":
                    st.info("Add the missing details above and submit again if you have them.")
                st.caption(f"ticket #{submission['id']}, saved to your history")
            elif submission.get("unavailable"):
                st.error(
                    "The model could not be reached, so no decision was made or stored. "
                    "A configured model is never silently replaced by the offline engine."
                )
            else:
                st.error(submission.get("error", "the request failed"))

with history_tab:
    if not st.session_state["token"]:
        st.info("Sign in to see your history.")
    else:
        listed = call("GET", "/tickets", authenticated=True)
        tickets = listed.json() if listed is not None and listed.status_code == 200 else []
        if not tickets:
            st.info(
                "Nothing here yet. Submit your first ticket in **New decision** and it will "
                "show up in this table."
            )
        else:
            st.caption(f"{len(tickets)} ticket{'s' if len(tickets) != 1 else ''} on this account")
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
            if st.button("View", type="primary"):
                response = call("GET", f"/tickets/{chosen}", authenticated=True)
                if response is not None:
                    if response.status_code == 200:
                        detail = response.json()
                        st.write(detail["message"])
                        if detail["decision"]:
                            show_decision(detail["decision"])
                    else:
                        st.error(explain(response))
