"""Streamlit User tab — self-service configuration for the lawyer.

The user (advogado) can:
  1. Select their user_id (must be pre-provisioned by Admin)
  2. Configure their Gmail address, WhatsApp number, and run interval
  3. Test the IMAP connection before saving
"""

from __future__ import annotations

import streamlit as st

from src.storage.credentials import CredentialStore
from src.storage.user_settings import UserSettingsStore
from src.ui.validators import validate_imap


def render(cred_store: CredentialStore, settings_store: UserSettingsStore) -> None:
    """Render the User tab content."""
    st.header("Configuracao do Usuario")

    # ---- User selection (only provisioned users) ----
    users = cred_store.list_users()
    if not users:
        st.warning("Nenhum usuario provisionado. Peca ao Admin para criar seu user_id.")
        return

    user_id = st.selectbox("Selecione seu user_id", options=users, key="user_select")
    if not user_id:
        return

    # ---- Load existing settings ----
    existing = settings_store.get(user_id)

    st.subheader(f"Configuracoes para: `{user_id}`")

    # ---- Email (IMAP) ----
    email = st.text_input(
        "Email (Gmail)",
        value=existing.get("email", ""),
        placeholder="seu.email@gmail.com",
        key="user_email",
    )
    imap_host = st.text_input(
        "IMAP Host",
        value=existing.get("imap_host", "imap.gmail.com"),
        key="user_imap_host",
    )
    imap_port = st.number_input(
        "IMAP Port",
        value=int(existing.get("imap_port", 993)),
        min_value=1,
        max_value=65535,
        key="user_imap_port",
    )

    # ---- Test IMAP ----
    if email and st.button("Testar Conexao IMAP", key="btn_test_imap"):
        creds = cred_store.get_decrypted(user_id)
        gmail_pw = creds.get("gmail_app_password", "")
        if not gmail_pw:
            st.error("Gmail App Password nao foi provisionada pelo Admin.")
        else:
            with st.spinner("Conectando ao IMAP..."):
                ok, msg = validate_imap(imap_host, int(imap_port), email, gmail_pw)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    # ---- WhatsApp ----
    whatsapp_to = st.text_input(
        "WhatsApp (destino)",
        value=existing.get("whatsapp_to", ""),
        placeholder="5511999999999",
        key="user_whatsapp",
    )

    # ---- Scheduling ----
    interval = st.slider(
        "Intervalo do pipeline (minutos)",
        min_value=1,
        max_value=120,
        value=int(existing.get("run_interval_minutes", 30)),
        key="user_interval",
    )

    # ---- Save ----
    st.divider()
    if st.button("Salvar Configuracoes", type="primary", key="btn_save_settings"):
        if not email:
            st.error("Email e obrigatorio.")
            return
        if not whatsapp_to or not whatsapp_to.strip().replace("+", "").isdigit():
            st.error("WhatsApp deve conter apenas digitos (ex: 5511999999999)")
            return

        try:
            settings_store.upsert(
                user_id=user_id,
                email=email,
                whatsapp_to=whatsapp_to,
                imap_host=imap_host,
                imap_port=int(imap_port),
                run_interval_minutes=interval,
            )
            st.success(f"Configuracoes salvas para `{user_id}`!")
        except Exception as exc:
            st.error(f"Erro ao salvar: {exc}")
