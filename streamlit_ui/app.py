import streamlit as st
import requests
import json

# ---------------------------------------
# BACKEND URL INPUT
# ---------------------------------------
BACKEND_URL = st.sidebar.text_input("Backend URL", "http://127.0.0.1:8000")

st.title("InsightQA")


# ---------------------------------------
# KB LIST FETCH FUNCTION
# ---------------------------------------
@st.cache_data(ttl=2)
def fetch_kb_list(url: str):
    try:
        resp = requests.get(f"{url}/kb/list")
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return []
    return []


# Maintain active KB
if "current_kb_id" not in st.session_state:
    st.session_state["current_kb_id"] = None

if "current_kb_name" not in st.session_state:
    st.session_state["current_kb_name"] = None


# =======================================
# SIDEBAR — KNOWLEDGE BASE LIST
# =======================================
st.sidebar.markdown("### Knowledge Bases")

kb_list = fetch_kb_list(BACKEND_URL)

if kb_list:
    kb_label_options = []
    kb_label_to_id = {}
    kb_label_to_name = {}

    for kb in kb_list:
        kb_id = kb["kb_id"]
        kb_name = kb.get("kb_name") or "Unnamed KB"

        created_at = kb.get("created_at", "")
        created_date = created_at.split("T")[0] if "T" in created_at else created_at

        # Label WITHOUT ID
        label = f"{kb_name}  ({created_date})"

        kb_label_options.append(label)
        kb_label_to_id[label] = kb_id
        kb_label_to_name[label] = kb_name

    # Determine which KB should be pre-selected
    default_index = 0
    if st.session_state["current_kb_id"]:
        for i, label in enumerate(kb_label_options):
            if kb_label_to_id[label] == st.session_state["current_kb_id"]:
                default_index = i
                break

    selected_label = st.sidebar.selectbox(
        "Select Knowledge Base",
        kb_label_options,
        index=default_index
    )

    st.session_state["current_kb_id"] = kb_label_to_id[selected_label]
    st.session_state["current_kb_name"] = kb_label_to_name[selected_label]

else:
    st.sidebar.info("No knowledge bases yet.")

# Active KB Name Display
if st.session_state["current_kb_name"]:
    st.sidebar.write(f"**Active KB:** {st.session_state['current_kb_name']}")
else:
    st.sidebar.write("**Active KB:** None")

st.sidebar.markdown("---")
st.sidebar.markdown("### Manage Active KB")

kb_id = st.session_state.get("current_kb_id")
kb_name = st.session_state.get("current_kb_name")

if kb_id:
    # ---------- Rename ----------
    if "rename_mode" not in st.session_state:
        st.session_state.rename_mode = False

    if not st.session_state.rename_mode:
        if st.sidebar.button("Rename KB"):
            st.session_state.rename_mode = True
            st.rerun()
    else:
        new_name = st.sidebar.text_input("New KB Name", value=kb_name)
        if st.sidebar.button("Save Name"):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/kb/rename",
                    data={"kb_id": kb_id, "new_name": new_name}
                )
                resp.raise_for_status()
                st.session_state["current_kb_name"] = new_name
                st.session_state.rename_mode = False
                fetch_kb_list.clear()
                st.rerun()
            except:
                st.sidebar.error("Rename failed")

    # ---------- View KB ----------
    if st.sidebar.button("View KB Contents"):
        try:
            resp = requests.get(f"{BACKEND_URL}/kb/view/{kb_id}")
            resp.raise_for_status()
            st.sidebar.json(resp.json())
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

    # ---------- Delete KB ----------
    if st.sidebar.button("Delete KB"):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/kb/delete",
                data={"kb_id": kb_id}
            )
            resp.raise_for_status()
            st.sidebar.success("KB deleted successfully")
            st.session_state["current_kb_id"] = None
            st.session_state["current_kb_name"] = None
            fetch_kb_list.clear()
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Delete failed: {e}")
else:
    st.sidebar.info("No active KB selected.")

# =======================================
# 1. BUILD KNOWLEDGE BASE
# =======================================
st.header("1. Build Knowledge Base")

kb_name_input = st.text_input("Knowledge Base Name")

main_file = st.file_uploader(
    "Primary App/Page File (HTML, main UI file)",
    key="main_file",
)

support_files = st.file_uploader(
    "Support Files (docs, specs, API notes, etc.)",
    accept_multiple_files=True,
    key="support_files",
)

st.caption(
    "- Each press of **Build Knowledge Base** creates a **new KB**.\n"
    "- The first uploaded HTML file becomes the primary UI page."
)

if st.button("Build Knowledge Base"):
    files = []

    if main_file:
        files.append(("files", (main_file.name, main_file.getvalue(), main_file.type)))

    if support_files:
        for f in support_files:
            files.append(("files", (f.name, f.getvalue(), f.type)))

    if not files:
        st.warning("Please upload at least one file.")
    else:
        try:
            resp = requests.post(
                f"{BACKEND_URL}/ingest",
                data={"name": kb_name_input},
                files=files
            )
            resp.raise_for_status()

            data = resp.json()
            st.success("Knowledge Base built successfully!")
            st.json(data)

            # Update currently selected KB
            kb_id = data.get("kb_id")
            kb_name = kb_name_input or "Unnamed KB"

            st.session_state["current_kb_id"] = kb_id
            st.session_state["current_kb_name"] = kb_name

            # Refresh sidebar KB list
            fetch_kb_list.clear()

            # Force UI rerun so sidebar updates immediately
            st.rerun()

        except Exception as e:
            st.error(f"Error building knowledge base: {e}")


# =======================================
# 2. GENERATE TEST CASES (RAG)
# =======================================
st.header("2. Generate Test Cases (RAG)")

query = st.text_input("Describe the feature to generate test cases for")

top_k = st.number_input(
    "How many chunks to retrieve (top_k)",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
)

if st.button("Generate Test Cases"):
    if not query.strip():
        st.warning("Enter a query first.")
    elif not st.session_state["current_kb_id"]:
        st.warning("No active KB selected.")
    else:
        try:
            payload = {
                "kb_id": st.session_state["current_kb_id"],
                "query": query,
                "top_k": top_k,
            }
            resp = requests.post(f"{BACKEND_URL}/agent/testcases", json=payload)
            resp.raise_for_status()

            data = resp.json()

            st.subheader("Retrieved Context")
            st.json(data.get("retrieved_chunks"))

            st.subheader("Generated Test Cases")
            st.json(data.get("testcases"))

            st.subheader("Prompt Used")
            st.code(data.get("prompt_used", ""), language="markdown")

        except Exception as e:
            st.error(f"Error generating test cases: {e}")


# =======================================
# 3. GENERATE SELENIUM SCRIPT
# =======================================
st.header("3. Generate Selenium Script")

selected_tc = st.text_area("Paste a single test case JSON object", height=250)

current_kb_id = st.session_state.get("current_kb_id")
html_files = []

kb_list = fetch_kb_list(BACKEND_URL)

if current_kb_id:
    kb_info = next((k for k in kb_list if k["kb_id"] == current_kb_id), None)
    if kb_info:
        for doc in kb_info.get("documents", []):
            if doc.get("is_html"):
                html_files.append(doc["filename"])

if current_kb_id and not html_files:
    st.info("No HTML files found for this KB.")

html_choice = None
if html_files:
    html_choice = st.selectbox("Choose HTML file", html_files)

if st.button("Generate Selenium Script"):
    if not selected_tc.strip():
        st.warning("Paste a test case JSON.")
    elif not current_kb_id:
        st.warning("No active KB selected.")
    elif not html_choice:
        st.warning("Select an HTML file first.")
    else:
        try:
            tc = json.loads(selected_tc)

            payload = {
                "kb_id": current_kb_id,
                "testcase": tc,
                "html_filename": html_choice,
            }

            resp = requests.post(f"{BACKEND_URL}/agent/generate_script", json=payload)
            resp.raise_for_status()

            st.subheader("Generated Selenium Script")
            st.code(resp.json()["script"], language="python")

        except json.JSONDecodeError:
            st.error("Invalid JSON format.")
        except Exception as e:
            st.error(f"Error generating script: {e}")
