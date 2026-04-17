"""Streamlit Admin tab — provision API keys for users.

Protected by a simple admin password. Allows the admin to:
  1. Select or create a user_id
  2. Enter/update infrastructure keys (Anthropic, Twilio, Supabase, Gmail)
  3. Validate each key with a live test before saving
"""

from __future__ import annotations

import streamlit as st

from src.storage.credentials import CredentialStore
from src.ui.validators import (
    validate_anthropic_key,
    validate_imap,
    validate_openai_key,
    validate_twilio,
)


def render(cred_store: CredentialStore, user_settings_store: "UserSettingsStore") -> None:
    """Render the Admin tab content."""
    st.header("Admin — Provisionamento de Credenciais")

    # ---- User selection ----
    existing_users = cred_store.list_users()
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_user = st.selectbox(
            "Selecione um usuario existente",
            options=[""] + existing_users,
            index=0,
            key="admin_select_user",
        )
    with col2:
        new_user = st.text_input("Ou crie um novo user_id", key="admin_new_user")

    user_id = new_user.strip() if new_user.strip() else selected_user
    if not user_id:
        st.info("Selecione ou crie um usuario para continuar.")
        return

    st.subheader(f"Credenciais para: `{user_id}`")

    # ---- Load existing values (show masked) ----
    existing = cred_store.get_decrypted(user_id) if cred_store.has_user(user_id) else {}

    def _mask(val: str | None) -> str:
        if not val:
            return ""
        return val[:8] + "..." + val[-4:] if len(val) > 16 else "****"

    # ---- Anthropic ----
    st.markdown("**LLM — Anthropic**")
    anthropic_hint = _mask(existing.get("anthropic_key"))
    anthropic_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder=f"Atual: {anthropic_hint}" if anthropic_hint else "sk-ant-...",
        key="admin_anthropic",
    )
    if anthropic_key and st.button("Testar Anthropic", key="btn_test_anthropic"):
        with st.spinner("Validando..."):
            ok, msg = validate_anthropic_key(anthropic_key)
        if ok:
            st.success(msg)
        else:
            st.error(msg)

    # ---- OpenAI (optional fallback) ----
    st.markdown("**LLM — OpenAI (fallback)**")
    openai_hint = _mask(existing.get("openai_key"))
    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder=f"Atual: {openai_hint}" if openai_hint else "sk-proj-...",
        key="admin_openai",
    )
    if openai_key and st.button("Testar OpenAI", key="btn_test_openai"):
        with st.spinner("Validando..."):
            ok, msg = validate_openai_key(openai_key)
        if ok:
            st.success(msg)
        else:
            st.error(msg)

    # ---- Twilio ----
    st.markdown("**WhatsApp — Twilio**")
    twilio_sid_hint = _mask(existing.get("twilio_sid"))
    twilio_sid = st.text_input(
        "Twilio Account SID",
        type="password",
        placeholder=f"Atual: {twilio_sid_hint}" if twilio_sid_hint else "ACxxxxxxxx",
        key="admin_twilio_sid",
    )
    twilio_token_hint = _mask(existing.get("twilio_token"))
    twilio_token = st.text_input(
        "Twilio Auth Token",
        type="password",
        placeholder=f"Atual: {twilio_token_hint}" if twilio_token_hint else "",
        key="admin_twilio_token",
    )
    twilio_number = st.text_input(
        "Twilio WhatsApp From",
        value=existing.get("twilio_number", ""),
        placeholder="whatsapp:+14155238886",
        key="admin_twilio_number",
    )
    if twilio_sid and twilio_token and st.button("Testar Twilio", key="btn_test_twilio"):
        with st.spinner("Validando..."):
            ok, msg = validate_twilio(twilio_sid, twilio_token)
        if ok:
            st.success(msg)
        else:
            st.error(msg)

    # ---- Gmail App Password ----
    st.markdown("**Gmail — App Password**")
    gmail_hint = _mask(existing.get("gmail_app_password"))
    gmail_password = st.text_input(
        "Gmail App Password",
        type="password",
        placeholder=f"Atual: {gmail_hint}" if gmail_hint else "xxxx xxxx xxxx xxxx",
        key="admin_gmail",
    )

    # ---- Supabase (optional per-user project) ----
    st.markdown("**Supabase (opcional — projeto do usuario)**")
    supabase_url = st.text_input(
        "Supabase URL",
        value=existing.get("supabase_url", ""),
        key="admin_supa_url",
    )
    supabase_key_hint = _mask(existing.get("supabase_key"))
    supabase_key = st.text_input(
        "Supabase Service Key",
        type="password",
        placeholder=f"Atual: {supabase_key_hint}" if supabase_key_hint else "",
        key="admin_supa_key",
    )

    # ---- Save ----
    st.divider()
    if st.button("Salvar Credenciais", type="primary", key="btn_save_creds"):
        kwargs: dict[str, str | None] = {"user_id": user_id, "admin_name": "admin"}
        if anthropic_key:
            kwargs["anthropic_key"] = anthropic_key
        if openai_key:
            kwargs["openai_key"] = openai_key
        if twilio_sid:
            kwargs["twilio_sid"] = twilio_sid
        if twilio_token:
            kwargs["twilio_token"] = twilio_token
        if twilio_number:
            kwargs["twilio_number"] = twilio_number
        if gmail_password:
            kwargs["gmail_app_password"] = gmail_password
        if supabase_url:
            kwargs["supabase_url"] = supabase_url
        if supabase_key:
            kwargs["supabase_key"] = supabase_key

        try:
            cred_store.upsert(**kwargs)
            st.success(f"Credenciais salvas para `{user_id}`!")
            st.rerun()
        except Exception as exc:
            st.error(f"Erro ao salvar: {exc}")
