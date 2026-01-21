import os
import json
import requests
import streamlit as st


SEX_MAP = {1: "Male", 2: "Female"}

EDUCATION_MAP = {
    1: "Graduate school",
    2: "University",
    3: "High school",
    4: "Others",
    5: "Unknown",
    6: "Unknown",
}

MARRIAGE_MAP = {
    1: "Married",
    2: "Single",
    3: "Others",
}

PAY_STATUS_MAP = {
    -1: "Pay duly",
    0: "No delay / Revolving",
    1: "Delay 1 month",
    2: "Delay 2 months",
    3: "Delay 3 months",
    4: "Delay 4 months",
    5: "Delay 5 months",
    6: "Delay 6 months",
    7: "Delay 7 months",
    8: "Delay 8 months",
    9: "Delay 9+ months",
}

PAY_LABELS = {
    "PAY_0": "Repayment status — Sep 2005",
    "PAY_2": "Repayment status — Aug 2005",
    "PAY_3": "Repayment status — Jul 2005",
    "PAY_4": "Repayment status — Jun 2005",
    "PAY_5": "Repayment status — May 2005",
    "PAY_6": "Repayment status — Apr 2005",
}

BILL_LABELS = {
    "BILL_AMT1": "Bill amount — Sep 2005 (NT$)",
    "BILL_AMT2": "Bill amount — Aug 2005 (NT$)",
    "BILL_AMT3": "Bill amount — Jul 2005 (NT$)",
    "BILL_AMT4": "Bill amount — Jun 2005 (NT$)",
    "BILL_AMT5": "Bill amount — May 2005 (NT$)",
    "BILL_AMT6": "Bill amount — Apr 2005 (NT$)",
}

PAYAMT_LABELS = {
    "PAY_AMT1": "Payment amount — Sep 2005 (NT$)",
    "PAY_AMT2": "Payment amount — Aug 2005 (NT$)",
    "PAY_AMT3": "Payment amount — Jul 2005 (NT$)",
    "PAY_AMT4": "Payment amount — Jun 2005 (NT$)",
    "PAY_AMT5": "Payment amount — May 2005 (NT$)",
    "PAY_AMT6": "Payment amount — Apr 2005 (NT$)",
}

API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(
    page_title="Credit Default Prediction", page_icon="💳", layout="centered"
)

st.title("💳 Credit Default Prediction")
st.caption(
    "Заполните форму и получите вероятность дефолта. Пустые (опциональные) поля будут отправлены как 0."
)

# Пример из схемы
EXAMPLE = {
    "LIMIT_BAL": 150,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 24,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 3913,
    "BILL_AMT2": 3102,
    "BILL_AMT3": 689,
    "BILL_AMT4": 300,
    "BILL_AMT5": 300,
    "BILL_AMT6": 300,
    "PAY_AMT1": 100,
    "PAY_AMT2": 689,
    "PAY_AMT3": 100,
    "PAY_AMT4": 100,
    "PAY_AMT5": 100,
    "PAY_AMT6": 100,
}

ALL_FEATURES = list(EXAMPLE.keys())


# -------- Helpers --------
def to_number_or_none(x: str):
    s = (x or "").strip()
    if s == "":
        return None
    s = s.replace(" ", "").replace(",", ".")
    return float(s)


def coalesce(v, default=0):
    return default if v is None else v


def build_payload_from_state() -> dict:
    # REQUIRED
    payload = {
        "LIMIT_BAL": float(st.session_state["LIMIT_BAL"]),
        "SEX": int(st.session_state["SEX"]),
        "EDUCATION": int(st.session_state["EDUCATION"]),
        "MARRIAGE": int(st.session_state["MARRIAGE"]),
        "AGE": int(st.session_state["AGE"]),
    }

    # OPTIONAL in UI
    for k in [
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]:
        payload[k] = coalesce(st.session_state.get(k), 0)

    # типы
    for k in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        payload[k] = int(payload[k])
    for k in [
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]:
        payload[k] = float(payload[k])

    return payload


def set_example():
    for k, v in EXAMPLE.items():
        st.session_state[k] = v


def reset_form():
    st.session_state["LIMIT_BAL"] = 1000.0
    st.session_state["SEX"] = 2
    st.session_state["EDUCATION"] = 2
    st.session_state["MARRIAGE"] = 1
    st.session_state["AGE"] = 24

    # PAY_* (оставим 0 как безопасный дефолт)
    for k in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        st.session_state[k] = 0

    # BILL_* и PAY_AMT_* (0)
    for k in [
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]:
        st.session_state[k] = 0.0


# init session defaults
if "LIMIT_BAL" not in st.session_state:
    reset_form()

# -------- Top buttons --------
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    st.button("✨ Заполнить пример", on_click=set_example, use_container_width=True)
with c2:
    st.button("🧹 Сброс", on_click=reset_form, use_container_width=True)

st.divider()

# -------- Form --------
with st.form("predict_form"):
    tab_main, tab_pay, tab_bill, tab_payamt = st.tabs(
        ["Основное", "Просрочки (PAY_*)", "Счета (BILL_*)", "Платежи (PAY_AMT_*)"]
    )

    with tab_main:
        st.subheader("Профиль клиента")
        col1, col2 = st.columns(2)

        with col1:
            st.number_input(
                "LIMIT_BAL",
                min_value=0.0,
                step=1000.0,
                key="LIMIT_BAL",
                help="Кредитный лимит",
            )
            st.number_input(
                "AGE",
                min_value=18,
                max_value=120,
                step=1,
                key="AGE",
                help="Возраст",
            )

        with col2:
            st.selectbox(
                "SEX",
                options=list(SEX_MAP.keys()),
                format_func=lambda x: f"{SEX_MAP[x]} ({x})",
                key="SEX",
            )

            st.selectbox(
                "EDUCATION",
                options=list(EDUCATION_MAP.keys()),
                format_func=lambda x: f"{EDUCATION_MAP[x]} ({x})",
                key="EDUCATION",
            )

            st.selectbox(
                "MARRIAGE",
                options=list(MARRIAGE_MAP.keys()),
                format_func=lambda x: f"{MARRIAGE_MAP[x]} ({x})",
                key="MARRIAGE",
            )
        st.info("Остальные поля можно не заполнять — они уйдут как 0.", icon="ℹ️")

    with tab_pay:
        st.subheader("Статусы просрочки (PAY_*)")

        keys = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
        cols = st.columns(2)

        for i, k in enumerate(keys):
            with cols[i % 2]:
                st.selectbox(
                    PAY_LABELS[k],
                    options=list(PAY_STATUS_MAP.keys()),
                    format_func=lambda x: f"{PAY_STATUS_MAP[x]} ({x})",
                    key=k,
                    help="Код статуса просрочки из датасета",
                )

    with tab_bill:
        st.subheader("Суммы счетов (BILL_AMT*)")
        st.caption("Сумма выставленного счёта за соответствующий месяц (NT$).")

        keys = [
            "BILL_AMT1",
            "BILL_AMT2",
            "BILL_AMT3",
            "BILL_AMT4",
            "BILL_AMT5",
            "BILL_AMT6",
        ]
        cols = st.columns(2)

        for i, k in enumerate(keys):
            with cols[i % 2]:
                st.number_input(
                    BILL_LABELS[k],
                    min_value=0.0,
                    step=100.0,
                    key=k,
                )

    with tab_payamt:
        st.subheader("Суммы платежей (PAY_AMT*)")
        st.caption("Сумма платежа за соответствующий месяц (NT$).")

        keys = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
        cols = st.columns(2)

        for i, k in enumerate(keys):
            with cols[i % 2]:
                st.number_input(
                    PAYAMT_LABELS[k],
                    min_value=0.0,
                    step=100.0,
                    key=k,
                )

    st.divider()
    submit = st.form_submit_button("Predict", use_container_width=True)

# -------- Result --------
if submit:
    payload = build_payload_from_state()

    with st.expander("Payload (что реально уходит в API)", expanded=False):
        st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")

    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()

        pred = data.get("prediction")
        prob = float(data.get("probability"))

        left, right = st.columns([1, 1])
        with left:
            st.metric("Prediction", pred)
        with right:
            st.metric("Probability", f"{prob:.4f}")

        st.progress(min(max(prob, 0.0), 1.0))

    except requests.HTTPError:
        st.error(f"API error: {r.status_code}\n\n{r.text}")
    except Exception as e:
        st.error(str(e))
