"""
Lead Score Agent — Streamlit UI
Run with: streamlit run app.py
Make sure uvicorn is already running: uvicorn main:app --port 8000
"""

import streamlit as st
import requests
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────
API_URL    = os.getenv("API_URL", "http://127.0.0.1:8000")
GROQ_KEY   = os.getenv("GROQ_API_KEY")
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "groq/compound-mini"

INCOME_SOURCES = [
    "salaried_govt", "salaried_mnc", "salaried_private",
    "self_employed_professional", "business_owner", "freelance", "unemployed",
]
LOAN_HISTORY_OPTIONS = [
    "no_history", "all_paid_on_time", "minor_delays",
    "one_default", "multiple_defaults",
]
CATEGORY_COLORS = {
    "Excellent Lead": "#22c55e",
    "Good Lead":      "#3b82f6",
    "Average Lead":   "#f59e0b",
    "Poor Lead":      "#ef4444",
}
CATEGORY_ICONS = {
    "Excellent Lead": "🟢",
    "Good Lead":      "🔵",
    "Average Lead":   "🟡",
    "Poor Lead":      "🔴",
}

# ── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Lead Score Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e2e8f0; }
    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2d3148;
    }
    .metric-card {
        background: #1e2235;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2d3148;
        margin-bottom: 12px;
    }
    .metric-value { font-size: 2rem; font-weight: 700; margin: 0; }
    .metric-label { font-size: 0.78rem; color: #94a3b8; margin: 4px 0 0 0;
                    text-transform: uppercase; letter-spacing: 1px; }
    .lead-card {
        background: #1e2235;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #2d3148;
        margin-bottom: 10px;
    }
    .ai-box {
        background: #12192b;
        border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        font-size: 0.92rem;
        line-height: 1.7;
        color: #cbd5e1;
        margin-top: 8px;
    }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .chat-user {
        background: #1e3a5f;
        border-radius: 12px 12px 4px 12px;
        padding: 10px 14px;
        margin: 6px 0 6px auto;
        max-width: 75%;
        font-size: 0.9rem;
    }
    .chat-agent {
        background: #1e2235;
        border-radius: 12px 12px 12px 4px;
        padding: 10px 14px;
        margin: 6px 0;
        max-width: 80%;
        font-size: 0.9rem;
        border: 1px solid #2d3148;
    }
    .stButton > button {
        background: #6366f1;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover { background: #4f52d4; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

# ── Fallback Local Database/Scoring Logic ──────────────────────
import storage
from ml_scoring import predict_lead as ml_predict_lead   # uses ML if model.pkl exists, else rules
from datetime import datetime

def check_server():
    # We return True because we have a local fallback for storage & scoring
    return True

def api_get(path):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=2)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    
    # Fallback to direct python imports
    if path == "/leads" or path == "/leads/":
        return storage.get_all()
    elif path.startswith("/leads/"):
        try:
            record_id = int(path.split("/")[-1])
            return storage.get_by_id(record_id)
        except Exception:
            return None
    return None

def api_post(path, data):
    try:
        r = requests.post(f"{API_URL}{path}", json=data, timeout=2)
        return r.json(), r.status_code
    except Exception:
        pass

    # Fallback to local execution
    if path == "/score" or path == "/score/":
        try:
            if data.get("annual_income", 0) <= 0:
                return {"detail": "Invalid income: must be greater than 0"}, 400

            # Uses ML model if model.pkl exists, otherwise falls back to rules automatically
            scoring_output = ml_predict_lead(
                cibil_score=data["cibil_score"],
                annual_income=data["annual_income"],
                assets_value=data["assets_value"],
                income_source=data["income_source"],
                previous_loan_history=data["previous_loan_history"],
            )

            result = {
                "lead_id": data.get("lead_id"),
                "name": data.get("name"),
                "source": data.get("source", "manual"),
                "phone": data.get("phone"),
                "email": data.get("email"),
                "city": data.get("city"),
                "cibil_score": data["cibil_score"],
                "annual_income": data["annual_income"],
                "assets_value": data["assets_value"],
                "income_source": data["income_source"],
                "previous_loan_history": data["previous_loan_history"],
                "total_score": scoring_output["total_score"],
                "category": scoring_output["category"],
                "recommendation": scoring_output["recommendation"],
                "breakdown": scoring_output["breakdown"],
                "reason_codes": scoring_output["reason_codes"],
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

            stored = storage.save_result(result)
            return stored, 200
        except Exception as e:
            return {"detail": str(e)}, 500
            
    return {"detail": "Not Found"}, 404


def call_llm(prompt, max_tokens=300):
    if not GROQ_KEY:
        return "⚠️ Set GROQ_API_KEY in your .env file."
    for attempt in range(3):
        try:
            r = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_KEY}",
                         "Content-Type": "application/json"},
                json={"model": GROQ_MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": max_tokens, "temperature": 0.5},
                timeout=30,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            elif r.status_code == 429:
                if attempt < 2:
                    time.sleep(15)
                    continue
                return "⚠️ Rate limit hit. Wait a moment and try again."
            else:
                return f"⚠️ Groq error {r.status_code}"
        except Exception as e:
            return f"⚠️ {e}"
    return "⚠️ Failed after retries."

def badge_html(category):
    color = CATEGORY_COLORS.get(category, "#64748b")
    return (f'<span class="badge" style="background:{color}22; '
            f'color:{color}; border:1px solid {color}44">{category}</span>')

def progress_bar_html(score, color):
    return (f'<div style="background:#2d3148;border-radius:8px;height:8px;margin-top:6px">'
            f'<div style="width:{score}%;background:{color};height:8px;'
            f'border-radius:8px"></div></div>')


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📊 Lead Score Agent")
    st.markdown("---")

    server_ok = True
    try:
        api_reachable = requests.get(f"{API_URL}/", timeout=1).status_code == 200
    except Exception:
        api_reachable = False

    if api_reachable:
        st.success("✅ Connected to API Server")
    else:
        st.info("ℹ️ Running in Local Mode (API Server Offline)")

    # Show which scoring method is active
    try:
        from ml_scoring import model_exists
        if model_exists():
            st.success("🤖 ML Model: Active (Random Forest)")
        else:
            st.warning("⚙️ ML Model: Not trained (using rules)")
    except Exception:
        st.warning("⚙️ Scoring: Rule-based only")

    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Dashboard",
        "➕ Score New Lead",
        "🔍 Search Lead",
        "💬 Chat with Leads",
        "📝 AI Summaries",
    ], label_visibility="collapsed")

    # Quick stats
    if server_ok:
        leads = api_get("/leads") or []
        if leads:
            st.markdown("---")
            st.markdown("**Quick Stats**")
            st.markdown(f"Total &nbsp;&nbsp;&nbsp; **{len(leads)}**")
            st.markdown(f"🟢 Excellent &nbsp; **{sum(1 for l in leads if l.get('category')=='Excellent Lead')}**")
            st.markdown(f"🔵 Good &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **{sum(1 for l in leads if l.get('category')=='Good Lead')}**")
            st.markdown(f"🟡 Average &nbsp;&nbsp; **{sum(1 for l in leads if l.get('category')=='Average Lead')}**")
            st.markdown(f"🔴 Poor &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **{sum(1 for l in leads if l.get('category')=='Poor Lead')}**")


# ══════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════

if page == "🏠 Dashboard":
    st.markdown("# 📊 Dashboard")
    st.markdown("---")

    if not server_ok:
        st.error("Server is offline. Run: `uvicorn main:app --reload --port 8000`")
        st.stop()

    leads = api_get("/leads") or []
    if not leads:
        st.info("No leads yet. Go to **Score New Lead** to get started.")
        st.stop()

    total     = len(leads)
    excellent = sum(1 for l in leads if l.get("category") == "Excellent Lead")
    good      = sum(1 for l in leads if l.get("category") == "Good Lead")
    average   = sum(1 for l in leads if l.get("category") == "Average Lead")
    poor      = sum(1 for l in leads if l.get("category") == "Poor Lead")

    # Metric row
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, label, color in [
        (c1, total,     "Total Leads",  "#6366f1"),
        (c2, excellent, "Excellent",    "#22c55e"),
        (c3, good,      "Good",         "#3b82f6"),
        (c4, average,   "Average",      "#f59e0b"),
        (c5, poor,      "Poor",         "#ef4444"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <p class="metric-value" style="color:{color}">{val}</p>
            <p class="metric-label">{label}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        search_q = st.text_input("🔍 Search by name or city")
    with f2:
        cat_filter = st.selectbox("Category", ["All", "Excellent Lead", "Good Lead",
                                                "Average Lead", "Poor Lead"])
    with f3:
        sort_by = st.selectbox("Sort by", ["Score ↓", "Score ↑", "Name A-Z"])

    filtered = leads
    if search_q:
        q = search_q.lower()
        filtered = [l for l in filtered if
                    q in (l.get("name") or "").lower() or
                    q in (l.get("city") or "").lower()]
    if cat_filter != "All":
        filtered = [l for l in filtered if l.get("category") == cat_filter]
    if sort_by == "Score ↓":
        filtered = sorted(filtered, key=lambda x: x.get("total_score", 0), reverse=True)
    elif sort_by == "Score ↑":
        filtered = sorted(filtered, key=lambda x: x.get("total_score", 0))
    else:
        filtered = sorted(filtered, key=lambda x: x.get("name") or "")

    st.markdown(f"**{len(filtered)} leads**")
    st.markdown("---")

    for lead in filtered:
        cat   = lead.get("category", "")
        color = CATEGORY_COLORS.get(cat, "#64748b")
        icon  = CATEGORY_ICONS.get(cat, "⚪")
        score = lead.get("total_score", 0)

        c1, c2, c3 = st.columns([3, 2, 1])
        with c1:
            st.markdown(f"""
            <div class="lead-card">
                <b>{icon} {lead.get('name','Unknown')}</b>
                &nbsp; {badge_html(cat)}
                <br><small style="color:#94a3b8">
                    {lead.get('lead_id','—')} · {lead.get('city') or 'N/A'} · {lead.get('source','—')}
                </small>
                {progress_bar_html(score, color)}
                <small style="color:{color}; font-weight:600">Score: {score}/100</small>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="lead-card">
                <small style="color:#94a3b8">FINANCIAL</small><br>
                CIBIL: <b>{lead.get('cibil_score')}</b><br>
                Income: <b>INR {lead.get('annual_income',0):,.0f}</b><br>
                Assets: <b>INR {lead.get('assets_value',0):,.0f}</b>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="lead-card" style="text-align:center">
                <small style="color:#94a3b8">ACTION</small><br>
                <span style="color:{color}; font-weight:600; font-size:0.82rem">
                    {lead.get('recommendation','—')}
                </span>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: SCORE NEW LEAD
# ══════════════════════════════════════════════════════════════

elif page == "➕ Score New Lead":
    st.markdown("# ➕ Score New Lead")
    st.markdown("---")

    if not server_ok:
        st.error("Server is offline.")
        st.stop()

    with st.form("score_form"):
        st.markdown("#### Customer Info")
        c1, c2, c3 = st.columns(3)
        with c1:
            lead_id = st.text_input("Lead ID", placeholder="LEAD-041")
            name    = st.text_input("Full Name *", placeholder="Rahul Sharma")
        with c2:
            phone  = st.text_input("Phone", placeholder="9876543210")
            email  = st.text_input("Email", placeholder="rahul@gmail.com")
        with c3:
            city   = st.text_input("City", placeholder="Mumbai")
            source = st.selectbox("Lead Source", ["CRM", "Website", "App", "Manual"])

        st.markdown("#### Financial Details")
        c4, c5, c6 = st.columns(3)
        with c4:
            cibil  = st.number_input("CIBIL Score", 300, 900, 700)
        with c5:
            income = st.number_input("Annual Income (INR)", 0, value=600000, step=10000)
        with c6:
            assets = st.number_input("Assets Value (INR)", 0, value=0, step=50000)

        st.markdown("#### Background")
        c7, c8 = st.columns(2)
        with c7:
            inc_src  = st.selectbox("Income Source", INCOME_SOURCES)
        with c8:
            loan_hist = st.selectbox("Loan History", LOAN_HISTORY_OPTIONS)

        submitted = st.form_submit_button("🚀 Score This Lead", use_container_width=True)

    if submitted:
        if not name:
            st.error("Customer name is required.")
        else:
            with st.spinner("Scoring..."):
                result, status = api_post("/score", {
                    "lead_id": lead_id or None, "name": name,
                    "source": source, "phone": phone or None,
                    "email": email or None, "city": city or None,
                    "cibil_score": cibil, "annual_income": income,
                    "assets_value": assets, "income_source": inc_src,
                    "previous_loan_history": loan_hist,
                })

            if status == 200:
                cat   = result.get("category", "")
                color = CATEGORY_COLORS.get(cat, "#64748b")
                score = result.get("total_score", 0)

                st.success("✅ Lead scored!")
                st.markdown("---")

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value" style="color:{color}">{score}</p>
                        <p class="metric-label">out of 100</p>
                        <br>
                        <p style="color:{color};font-weight:700">{cat}</p>
                        <p style="color:#94a3b8;font-size:0.85rem">
                            {result.get('recommendation')}
                        </p>
                    </div>""", unsafe_allow_html=True)
                with col2:
                    st.markdown("**Score Breakdown**")
                    bd = result.get("breakdown", {})
                    for key, (label, max_m) in {
                        "cibil_score_marks":    ("CIBIL Score",   35),
                        "annual_income_marks":  ("Annual Income", 25),
                        "assets_value_marks":   ("Assets Value",  15),
                        "income_source_marks":  ("Income Source", 10),
                        "previous_loans_marks": ("Loan History",  15),
                    }.items():
                        got = bd.get(key, 0)
                        st.markdown(f"**{label}** — {got}/{max_m}")
                        st.progress(int((got / max_m) * 100))

                st.markdown("**Reason Codes**")
                for r in result.get("reason_codes", []):
                    st.markdown(f"✦ {r}")

                st.markdown("---")
                st.markdown("**🤖 AI Summary**")
                with st.spinner("Generating..."):
                    summary = call_llm(
                        f"You are a loan lead analyst. Write a 3-sentence plain-English "
                        f"summary for the sales team. Mention actual numbers. End with "
                        f"whether to pursue this lead.\n\n"
                        f"Name:{name} CIBIL:{cibil} Income:INR {income:,} "
                        f"Assets:INR {assets:,} Source:{inc_src} "
                        f"History:{loan_hist} Score:{score}/100 "
                        f"Category:{cat} Action:{result.get('recommendation')}\n\n"
                        f"Write ONLY the 3-sentence summary.", 150)
                st.markdown(f'<div class="ai-box">{summary}</div>',
                            unsafe_allow_html=True)
            else:
                st.error(f"Error: {result.get('detail','Unknown error')}")


# ══════════════════════════════════════════════════════════════
#  PAGE: SEARCH LEAD
# ══════════════════════════════════════════════════════════════

elif page == "🔍 Search Lead":
    st.markdown("# 🔍 Search Lead")
    st.markdown("---")

    c1, c2 = st.columns([4, 1])
    with c1:
        query = st.text_input("", placeholder="Record ID (e.g. 15) or Lead ID (e.g. LEAD-015)")
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        go = st.button("Search", use_container_width=True)

    if go and query:
        with st.spinner("Fetching..."):
            if query.strip().isdigit():
                lead = api_get(f"/leads/{query.strip()}")
            else:
                all_leads = api_get("/leads") or []
                matched   = [l for l in all_leads
                             if l.get("lead_id","").upper() == query.strip().upper()]
                lead = matched[0] if matched else None

        if not lead:
            st.error(f"No lead found for '{query}'")
        else:
            cat   = lead.get("category", "")
            color = CATEGORY_COLORS.get(cat, "#64748b")
            icon  = CATEGORY_ICONS.get(cat, "")
            score = lead.get("total_score", 0)

            # Metric row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Score",  f"{score}/100")
            m2.metric("CIBIL",  lead.get("cibil_score"))
            m3.metric("Income", f"INR {lead.get('annual_income',0):,.0f}")
            m4.metric("Assets", f"INR {lead.get('assets_value',0):,.0f}")

            st.markdown(f"""
            <div class="lead-card">
                <b style="font-size:1.1rem">{icon} {lead.get('name')}</b>
                &nbsp; {badge_html(cat)}
                <br><small style="color:#94a3b8">
                    {lead.get('lead_id','—')} &nbsp;·&nbsp;
                    📍 {lead.get('city') or 'N/A'} &nbsp;·&nbsp;
                    📱 {lead.get('phone') or 'N/A'} &nbsp;·&nbsp;
                    ✉️ {lead.get('email') or 'N/A'}
                </small>
            </div>""", unsafe_allow_html=True)

            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("**Score Breakdown**")
                bd = lead.get("breakdown", {})
                for key, (label, max_m) in {
                    "cibil_score_marks":    ("CIBIL Score",   35),
                    "annual_income_marks":  ("Annual Income", 25),
                    "assets_value_marks":   ("Assets Value",  15),
                    "income_source_marks":  ("Income Source", 10),
                    "previous_loans_marks": ("Loan History",  15),
                }.items():
                    got = bd.get(key, 0)
                    st.markdown(f"**{label}** — {got}/{max_m}")
                    st.progress(int((got / max_m) * 100))

            with col_r:
                st.markdown("**Reason Codes**")
                for r in lead.get("reason_codes", []):
                    st.markdown(f"✦ {r}")
                st.markdown(f"**Recommendation:** `{lead.get('recommendation')}`")
                st.markdown(f"**Source:** {lead.get('source')}  |  "
                            f"**Scored:** {lead.get('timestamp','')[:10]}")

            st.markdown("---")
            st.markdown("**🤖 AI Summary**")
            with st.spinner("Generating..."):
                summary = call_llm(
                    f"You are a loan lead analyst. Write a 3-sentence plain-English "
                    f"summary for the sales team. Mention actual numbers. End with "
                    f"whether to pursue this lead.\n\n"
                    f"Name:{lead.get('name')} CIBIL:{lead.get('cibil_score')} "
                    f"Income:INR {lead.get('annual_income',0):,} "
                    f"Assets:INR {lead.get('assets_value',0):,} "
                    f"Source:{lead.get('income_source')} "
                    f"History:{lead.get('previous_loan_history')} "
                    f"Score:{score}/100 Category:{cat}\n\n"
                    f"Write ONLY the 3-sentence summary.", 150)
            st.markdown(f'<div class="ai-box">{summary}</div>',
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: CHAT
# ══════════════════════════════════════════════════════════════

elif page == "💬 Chat with Leads":
    st.markdown("# 💬 Chat with Leads")
    st.markdown("---")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    all_leads     = api_get("/leads") or []
    leads_context = "\n".join([
        f"ID:{l.get('record_id')} LeadID:{l.get('lead_id','-')} "
        f"Name:{l.get('name','-')} City:{l.get('city','-')} "
        f"Score:{l.get('total_score')} Category:{l.get('category')} "
        f"CIBIL:{l.get('cibil_score')} Income:{l.get('annual_income')} "
        f"Source:{l.get('income_source')} Action:{l.get('recommendation')}"
        for l in all_leads
    ])

    # Suggestion buttons
    suggestions = ["Who has the highest score?", "Show all excellent leads",
                   "Which leads to call first?", "How many poor leads?",
                   "Best leads from Mumbai", "Compare top 3 leads"]
    cols = st.columns(3)
    for i, s in enumerate(suggestions):
        if cols[i % 3].button(s, key=f"s{i}"):
            st.session_state.pending = s

    st.markdown("---")

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">👤 {msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-agent">🤖 {msg["content"]}</div>',
                        unsafe_allow_html=True)

    user_input = st.chat_input("Ask anything about your leads...")
    if "pending" in st.session_state:
        user_input = st.session_state.pop("pending")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        history_text = "".join(
            f"{'User' if t['role']=='user' else 'Assistant'}: {t['content']}\n"
            for t in st.session_state.chat_history[-6:]
        )
        with st.spinner("Thinking..."):
            reply = call_llm(
                f"You are a loan lead analyst. Answer about this lead database.\n"
                f"Plain text only, no markdown.\n\n"
                f"LEADS:\n{leads_context}\n\n{history_text}", 400)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# ══════════════════════════════════════════════════════════════
#  PAGE: AI SUMMARIES
# ══════════════════════════════════════════════════════════════

elif page == "📝 AI Summaries":
    st.markdown("# 📝 AI Summaries")
    st.markdown("All summaries saved from lead searches.")
    st.markdown("---")

    summary_file = "ai_summaries.json"
    if not os.path.exists(summary_file):
        st.info("No summaries yet. Search a lead to generate one.")
        st.stop()

    with open(summary_file, "r", encoding="utf-8") as f:
        try:
            summaries = json.load(f)
        except Exception:
            st.error("Could not read ai_summaries.json")
            st.stop()

    if not summaries:
        st.info("No summaries yet. Run `python client.py 15` to generate one.")
        st.stop()

    search_s = st.text_input("🔍 Filter by Lead ID", placeholder="e.g. LEAD-015")
    if search_s:
        summaries = [s for s in summaries
                     if search_s.lower() in (s.get("lead_id") or "").lower()]

    st.markdown(f"**{len(summaries)} summaries found**")
    st.markdown("---")

    for s in reversed(summaries):
        with st.expander(f"📄 Lead ID: {s.get('lead_id', 'N/A')}"):
            st.markdown(f'<div class="ai-box">{s.get("summary","")}</div>',
                        unsafe_allow_html=True)