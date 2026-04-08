# python -m streamlit run .\DATACloner_Interface.py -- server.port 8502
import os
import json
import re
from io import StringIO
from pathlib import Path

import pandas as pd
import sqlparse
import streamlit as st
from openai import AuthenticationError

from DACloner import (
    DesignState,
    runnable,
    load_list_available_dimensions_to_state,
    load_SAP_2_Snowflake_data_types_mapping_to_state,
)

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
st.set_page_config(page_title="DAcloner", page_icon="💬", layout="centered")

# Wymagamy klucza do LLM z ENV (zgodnie z Twoim kodem .env → load_dotenv())
if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].strip():
    st.warning("Brakuje OPENAI_API_KEY w środowisku (.env). Ustaw i odpal ponownie.")

st.title("💬 DAcloner – a data architect’s best friend ")

# ------------------------------------------------------------
# Session initialization
# ------------------------------------------------------------
if "initialized" not in st.session_state:
    # Pierwsze uruchomienie: zbuduj stan i doładuj globalne pliki
    st.session_state.state = DesignState()
    st.session_state.state = load_list_available_dimensions_to_state(st.session_state.state)
    st.session_state.state = load_SAP_2_Snowflake_data_types_mapping_to_state(st.session_state.state)

    # Metadane sesji do configu invoke (z Twojego kodu)
    st.session_state.session_id = "streamlit-user"
    st.session_state.thread_id = "ui-thread"

    # Historia bąbelków: list[dict(role, content)]
    st.session_state.chat = []
    st.session_state.initialized = True

# ------------------------------------------------------------
# Helpers – ekstrakcja WIELU plików z jednej wiadomości
# ------------------------------------------------------------
# Wspierane znaczniki (case-insensitive):
#  1) **SQL_SCRIPT_START**...**SQL_SCRIPT_END**           → .sql
#  2) **CSV_START**...**CSV_END**                         → .csv
#  3) **FILE_START: <nazwa z rozszerzeniem>**...**FILE_END**  → dowolny typ (np. .ddl, .sql, .csv, .yaml)
#     Przykład: **FILE_START: dim_customer.ddl** ... **FILE_END**

_GENERIC_FILE_RE = re.compile(r"\*\*FILE_START:\s*([^*\n]+?)\*\*(.*?)\*\*FILE_END\*\*", re.IGNORECASE | re.DOTALL)
_SQL_FILE_RE     = re.compile(r"\*\*SQL_SCRIPT_START\*\*(.*?)\*\*SQL_SCRIPT_END\*\*", re.IGNORECASE | re.DOTALL)
_CSV_FILE_RE     = re.compile(r"\*\*CSV_START\*\*(.*?)\*\*CSV_END\*\*", re.IGNORECASE | re.DOTALL)
_DDL_FILE_RE     = re.compile(r"\*\*DDL_SCRIPT_START\*\*(.*?)\*\*DDL_SCRIPT_(?:END|STOP)\*\*", re.IGNORECASE | re.DOTALL)

def _extract_files_and_clean_text(text: str):
    """Zwraca (cleaned_text, files), gdzie files to lista słowników:
       { 'type': 'sql'|'csv'|'file', 'ext': 'sql'|'csv'|inny, 'name': str|None, 'content': str }
       Obsługuje WIELE bloków w jednej wiadomości, zachowuje kolejność.
    """
    if not text:
        return "", []

    matches = []  # każdy: dict(start, end, name, type, ext, content)

    # 3) FILE_START: <name>
    for m in _GENERIC_FILE_RE.finditer(text):
        raw_name = m.group(1).strip()
        body = m.group(2).strip()
        ext = Path(raw_name).suffix.lower().lstrip(".") or "txt"
        ftype = "sql" if ext in {"sql", "ddl"} else ("csv" if ext == "csv" else "file")
        matches.append({
            "start": m.start(),
            "end": m.end(),
            "name": raw_name,
            "type": ftype,
            "ext": ext,
            "content": body,
        })

    # 1) **SQL_SCRIPT_START**...**SQL_SCRIPT_END** (może być wiele)
    for m in _SQL_FILE_RE.finditer(text):
        matches.append({
            "start": m.start(),
            "end": m.end(),
            "name": None,
            "type": "sql",
            "ext": "sql",
            "content": m.group(1).strip(),
        })

    # 2) **CSV_START**...**CSV_END** (może być wiele)
    for m in _CSV_FILE_RE.finditer(text):
        matches.append({
            "start": m.start(),
            "end": m.end(),
            "name": None,
            "type": "csv",
            "ext": "csv",
            "content": m.group(1).strip(),
        })

    # **DDL_SCRIPT_START**...**DDL_SCRIPT_END/STOP** → traktuj jak SQL
    for m in _DDL_FILE_RE.finditer(text):
        matches.append({
            "start": m.start(),
            "end": m.end(),
            "name": None,
            "type": "sql",   # <— kluczowe: identycznie jak SQL
            "ext": "sql",    # możesz dać "ddl" jeśli wolisz domyślną końcówkę .ddl
            "content": m.group(1).strip(),
        })

    # Sortuj po pozycji w tekście, żeby zachować naturalną kolejność wyświetlania
    matches.sort(key=lambda d: d["start"]) 

    # Usuń wszystkie bloki z treści bąbla (od końca, by nie przesuwać indeksów)
    cleaned = text
    for m in sorted(matches, key=lambda d: d["start"], reverse=True):
        cleaned = cleaned[: m["start"]] + cleaned[m["end"] :]

    # Zwróć listę "plików" w kolejności
    files = [
        {
            "type": m["type"],
            "ext": m["ext"],
            "name": m["name"],
            "content": m["content"],
        }
        for m in matches
    ]

    # Final polish: zbędne białe znaki po wycięciu bloków
    cleaned = cleaned.strip()
    return cleaned, files


def _pretty_guess_filename(base: str, i: int, ext: str, provided: str | None) -> str:
    """Wymyśla sensowną nazwę: jeśli nadana w znaczniku – użyj, wpp. {base}_{i}.{ext}."""
    if provided:
        return provided
    base = (base or "attachment").replace(" ", "_")
    ext = (ext or "txt").lstrip(".")
    return f"{base}_{i}.{ext}"


def run_until_interrupt(user_text_or_none: str | None):
    """
    - Jeśli user_text_or_none jest podany: wrzuca do `state.last_user_message`.
    - Wykonuje kroki grafu aż:
        a) pojawi się `__interrupt__` (wtedy dodajemy bąbelek asystenta i STOP),
        b) albo graf skończy (END) – wtedy dorzucamy finalny stan/info.
    - Zwraca (status, reply_text_or_None)
        status: "need_input" | "done" | "auth_error"
    """
    state = st.session_state.state

    if user_text_or_none:
        state.last_user_message = user_text_or_none

    _auth_msg = (
        "**Błąd uwierzytelniania OpenAI (401).** Klucz w `OPENAI_API_KEY` jest "
        "nieprawidłowy, wygasły lub odwołany. Zaktualizuj `.env` (np. w katalogu głównym "
        "repozytorium lub obok tego pliku), zrestartuj Streamlit. "
        "Nowy klucz: https://platform.openai.com/account/api-keys"
    )

    # Pętla bezpieczeństwa (żeby nie zapętlić UI)
    for _ in range(20):
        try:
            result = runnable.invoke(
                state,
                config={
                    "configurable": {
                        "session_id": st.session_state.session_id,
                        "thread_id": st.session_state.thread_id,
                    }
                },
            )
        except AuthenticationError:
            return "auth_error", _auth_msg

        if "__interrupt__" in result:
            interrupt_obj = result["__interrupt__"][0].value
            msg = interrupt_obj["message"]
            st.session_state.state = interrupt_obj["next_state"]
            return "need_input", msg

        # brak interruptu → graf przesunął stan do przodu
        st.session_state.state = result

        # Czy to koniec?
        if not getattr(st.session_state.state, "awaiting_input_for", None):
            return "done", None

    return "need_input", "Coś się przyblokowało – przerwano pętlę bezpieczeństwa."


# AUTO-START: pierwszy krok grafu, żeby dostać pierwszą wiadomość bota
if not st.session_state.chat:
    status, reply = run_until_interrupt(None)  # brak wiadomości od usera
    if status == "auth_error" and reply:
        st.session_state.chat.append({"role": "assistant", "content": reply})
        st.rerun()
    elif status == "need_input" and reply:
        st.session_state.chat.append({"role": "assistant", "content": reply})
        st.rerun()  # odśwież UI, żeby bąbelek się pojawił

# ------------------------------------------------------------
# 4) Wgrywanie analiz źródeł (CSV) -> zapis do state.source_table_analyze
# ------------------------------------------------------------
with st.expander("📎 Przydatne analizy źródeł (CSV)"):
    st.markdown(
        "Wrzuć jeden lub więcej plików CSV z analizą kolumn. "
        "Domyślnie nazwa pliku (bez rozszerzenia) stanie się nazwą źródła. "
        "Jeśli nazwa nie pasuje do tabel używanych w modelu, możesz ją zmienić poniżej."
    )

    uploaded_files = st.file_uploader(
        "Pliki CSV",
        type=["csv", "txt"],
        accept_multiple_files=True,
        key="csv_uploader",
    )

    # Przycisk zapisu
    save_clicked = st.button("💾 Zapisz wgrane pliki do stanu", type="primary", disabled=not uploaded_files)

    if save_clicked and uploaded_files:
        if st.session_state.state.source_table_analyze is None:
            st.session_state.state.source_table_analyze = {}

        saved = []
        for uf in uploaded_files:
            source_name = Path(uf.name).stem  # np. BKPF z BKPF.csv
            # pobierz bytes z pamięci, zdekoduj jako tekst
            raw = uf.getvalue()
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = raw.decode("latin-1", errors="ignore")
            # normalizacja końców linii
            text = text.replace("\r\n", "\n").replace("\r", "\n")

            # zapis do stanu: { "NAZWA_TABELI": "surowy_csv" }
            st.session_state.state.source_table_analyze[source_name] = text
            saved.append(source_name)

        st.success(f"Zapisano: {', '.join(saved)}")

    # Podgląd i szybka edycja nazw
    if st.session_state.state.source_table_analyze:
        st.markdown("**Wgrane analizy:**")

        # rename pojedynczego wpisu
        with st.form("rename_source_key"):
            keys = list(st.session_state.state.source_table_analyze.keys())
            selected = st.selectbox("Zmień nazwę źródła", options=keys)
            new_name = st.text_input("Nowa nazwa źródła", value=selected)
            rename_ok = st.form_submit_button("Zmień nazwę")
            if rename_ok and new_name and new_name != selected:
                # przenieś treść pod nowy klucz
                st.session_state.state.source_table_analyze[new_name] = (
                    st.session_state.state.source_table_analyze.pop(selected)
                )
                st.success(f"Zmieniono nazwę: {selected} → {new_name}")

        # listowanie z podglądem
        for name, txt in st.session_state.state.source_table_analyze.items():
            with st.expander(f"🔎 {name} — podgląd (pierwsze wiersze)"):
                # próbuj pokazać 8 wierszy CSV jako tabelę
                try:
                    df = pd.read_csv(StringIO(txt))
                    st.dataframe(df.head(8))
                except Exception:
                    st.code("\n".join(txt.splitlines()[:12]))

        # czyszczenie całości
        if st.button("🗑️ Wyczyść wszystkie analizy"):
            st.session_state.state.source_table_analyze = {}
            st.info("Wyczyszczono wszystkie wgrane analizy.")


# ------------------------------------------------------------
# 5) Render historii – teraz obsługa WIELU plików na bąbelek
# ------------------------------------------------------------
for idx, m in enumerate(st.session_state.chat):
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            body, files = _extract_files_and_clean_text(m["content"])  # <— NOWE
            if body:
                st.markdown(body)

            # Sugerowana baza nazwy (np. nazwa wymiaru)
            dim_name = getattr(st.session_state.state, "currently_modeled_object", None) or "dimension"

            # Wyświetl WSZYSTKIE załączone pliki w oryginalnej kolejności
            for i, f in enumerate(files, start=1):
                label = "SQL script" if f["type"] == "sql" else ("CSV" if f["type"] == "csv" else "Plik")
                file_name = _pretty_guess_filename(dim_name, i, f["ext"], f["name"])

                st.subheader(f"{label} — {file_name}")

                # Edytowalny obszar treści
                edited = st.text_area(
                    "Edytuj zawartość (zostanie pobrane przyciskiem poniżej)",
                    value=f["content"],
                    height=240,
                    key=f"file_area_{idx}_{i}",
                )

                # Pobierz
                st.download_button(
                    "💾 Pobierz plik",
                    data=edited,
                    file_name=file_name,
                    mime = "text/sql" if f["type"] == "sql" else ("text/csv" if f["type"] == "csv" else "text/plain"),
                    key=f"dl_file_{idx}_{i}",
                )

                # Dodatkowy podgląd CSV (jeśli możliwy)
                if f["ext"] == "csv":
                    with st.expander("Podgląd CSV (pierwsze wiersze)"):
                        try:
                            df = pd.read_csv(StringIO(edited))
                            st.dataframe(df.head(8))
                        except Exception as e:
                            st.info("Nie udało się sparsować CSV – pokazuję fragment surowy.")
                            st.code("\n".join(edited.splitlines()[:12]))

        else:
            st.markdown(m["content"])

# ------------------------------------------------------------
# 6) Input użytkownika + wykonanie kroku
# ------------------------------------------------------------
prompt = st.chat_input("Napisz wiadomość…")
if prompt:
    # 1) pokaż bąbelek usera
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) uruchom graf aż poprosi o kolejne dane (Interrupt) lub skończy
    status, reply = run_until_interrupt(prompt)

    if status == "auth_error" and reply:
        st.session_state.chat.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

    elif status == "need_input" and reply:
        st.session_state.chat.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            body, files = _extract_files_and_clean_text(reply)
            if body:
                st.markdown(body)

            dim_name = getattr(st.session_state.state, "currently_modeled_object", None) or "dimension"

            for i, f in enumerate(files, start=1):
                label = "SQL script" if f["type"] == "sql" else ("CSV" if f["type"] == "csv" else "Plik")
                file_name = _pretty_guess_filename(dim_name, i, f["ext"], f["name"])

                st.subheader(f"{label} — {file_name}")
                edited = st.text_area(
                    "Edytuj zawartość (zostanie pobrane przyciskiem poniżej)",
                    value=f["content"],
                    height=240,
                    key=f"file_area_live_{i}",
                )
                st.download_button(
                    "💾 Pobierz plik",
                    data=edited,
                    file_name=file_name,
                    mime = "text/sql" if f["type"] == "sql" else ("text/csv" if f["type"] == "csv" else "text/plain"),
                    key=f"dl_file_live_{i}",
                )

                if f["ext"] == "csv":
                    with st.expander("Podgląd CSV (pierwsze wiersze)"):
                        try:
                            df = pd.read_csv(StringIO(edited))
                            st.dataframe(df.head(8))
                        except Exception:
                            st.info("Nie udało się sparsować CSV – pokazuję fragment surowy.")
                            st.code("\n".join(edited.splitlines()[:12]))

    elif status == "done":
        # Opcjonalnie: pokaż podsumowanie finalne / podgląd stanu
        with st.chat_message("assistant"):
            st.markdown("✅ **Zakończono aktualny etap.** Możesz kontynuować rozmowę lub podejrzeć stan poniżej.")

# ------------------------------------------------------------
# 7) Panel diagnostyczny (rozwiń jeśli chcesz)
# ------------------------------------------------------------
with st.expander("🔎 Podgląd stanu (diag)"):
    # Pydantic v2: model_dump; jeśli masz v1 – można użyć .dict()
    try:
        state_json = st.session_state.state.model_dump()
    except Exception:
        state_json = st.session_state.state.dict()
    st.json(state_json)
