"""emAI — Streamlit multi-tab dashboard.

Launch with:
    streamlit run src/ui/app.py

Tabs:
  1. Admin    — provision API keys per user (password-protected)
  2. Usuario  — configure email, WhatsApp, interval
  3. Dashboard — processed emails chart + service status
"""

from __future__ import annotations

import streamlit as st
from sqlmodel import SQLModel, Session, select, func

from config.settings import settings
from src.storage.credentials import CredentialStore, UserCredential
from src.storage.user_settings import UserSettingsStore, UserSetting
from src.storage.models import ProcessedEmail
from src.storage.state import _build_engine


# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="emAI — Painel de Controle",
    page_icon="📧",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Shared engine + stores (cached per session)
# --------------------------------------------------------------------------- #

@st.cache_resource
def _get_engine():
    engine = _build_engine(settings.database_url)
    SQLModel.metadata.create_all(engine)
    return engine


def _get_stores():
    engine = _get_engine()
    return CredentialStore(engine), UserSettingsStore(engine)


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #

tab_admin, tab_user, tab_dashboard = st.tabs(["Admin", "Usuario", "Dashboard"])

cred_store, settings_store = _get_stores()

# ---- Admin Tab (password-protected) ----
with tab_admin:
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    if not st.session_state.admin_authenticated:
        st.subheader("Autenticacao Admin")
        pwd = st.text_input("Admin Password", type="password", key="admin_pwd")
        if st.button("Entrar", key="btn_admin_login"):
            if pwd == settings.admin_password:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Senha incorreta.")
    else:
        from src.ui.admin_tab import render as render_admin
        render_admin(cred_store, settings_store)

# ---- User Tab ----
with tab_user:
    from src.ui.user_tab import render as render_user
    render_user(cred_store, settings_store)

# ---- Dashboard Tab ----
with tab_dashboard:
    st.header("Dashboard — Metricas de Processamento")

    # ---- User selection for dashboard ----
    users = cred_store.list_users()
    if not users:
        st.info("Nenhum usuario provisionado ainda.")
    else:
        dash_user = st.selectbox(
            "Selecione o usuario", options=users, key="dash_user"
        )

        if dash_user:
            engine = _get_engine()

            # ---- Service status ----
            user_creds = cred_store.get_decrypted(dash_user)
            user_settings = settings_store.get(dash_user)

            has_creds = bool(user_creds and user_creds.get("anthropic_key"))
            has_settings = bool(user_settings and user_settings.get("email"))

            col1, col2, col3 = st.columns(3)
            with col1:
                if has_creds and has_settings:
                    st.metric("Status do Servico", "Ativo")
                elif has_creds:
                    st.metric("Status do Servico", "Parcial")
                else:
                    st.metric("Status do Servico", "Inativo")
            with col2:
                interval = user_settings.get("run_interval_minutes", "N/A") if user_settings else "N/A"
                st.metric("Intervalo", f"{interval} min")
            with col3:
                whatsapp = user_settings.get("whatsapp_to", "N/A") if user_settings else "N/A"
                st.metric("WhatsApp", str(whatsapp)[-13:] if whatsapp != "N/A" else "N/A")

            # ---- Bar chart: delivered vs irrelevant ----
            st.subheader("Emails Processados vs Irrelevantes")
            with Session(engine) as session:
                delivered_count = session.exec(
                    select(func.count()).where(
                        ProcessedEmail.delivery_status == "delivered",
                        ProcessedEmail.user_id == dash_user,
                    )
                ).one()
                irrelevant_count = session.exec(
                    select(func.count()).where(
                        ProcessedEmail.delivery_status == "skipped_irrelevant",
                        ProcessedEmail.user_id == dash_user,
                    )
                ).one()

            import pandas as pd

            chart_data = pd.DataFrame({
                "Categoria": ["Entregues", "Irrelevantes"],
                "Quantidade": [delivered_count, irrelevant_count],
            })
            st.bar_chart(chart_data, x="Categoria", y="Quantidade")

            # ---- Recent activity table (metadata only) ----
            st.subheader("Atividade Recente (ultimos 20)")
            with Session(engine) as session:
                recent = session.exec(
                    select(
                        ProcessedEmail.id,
                        ProcessedEmail.uid,
                        ProcessedEmail.relevance,
                        ProcessedEmail.priority,
                        ProcessedEmail.delivery_status,
                        ProcessedEmail.processed_at,
                    )
                    .where(ProcessedEmail.user_id == dash_user)
                    .order_by(ProcessedEmail.processed_at.desc())  # type: ignore[attr-defined]
                    .limit(20)
                ).all()

            if recent:
                rows = [
                    {
                        "ID": r[0],
                        "UID": r[1],
                        "Relevante": "Sim" if r[2] else "Nao",
                        "Prioridade": r[3],
                        "Status": r[4],
                        "Processado em": str(r[5])[:19],
                    }
                    for r in recent
                ]
                st.dataframe(rows, use_container_width=True)
            else:
                st.info("Nenhum email processado ainda para este usuario.")
