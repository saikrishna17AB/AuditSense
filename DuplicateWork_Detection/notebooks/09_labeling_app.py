import streamlit as st
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
FILE = BASE / "outputs" / "labeling_set.csv"

st.set_page_config(
    page_title="MPLADS Duplicate Work Labeling",
    layout="wide"
)

# Load data
df = pd.read_csv(FILE)

# Make sure LABEL exists
if "LABEL" not in df.columns:
    df["LABEL"] = ""

# Current position
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

i = st.session_state.current_index

# Skip already labeled rows when possible
while i < len(df) and str(df.loc[i, "LABEL"]).strip() in ["0", "1"]:
    i += 1

st.session_state.current_index = i

if i >= len(df):
    st.success("All pairs have been labeled!")

    df.to_csv(FILE, index=False)

    st.stop()

row = df.iloc[i]

st.title("MPLADS Duplicate Work Labeling")

st.write(
    f"Pair **{i + 1} / {len(df)}**"
)

st.divider()

# Work descriptions
col1, col2 = st.columns(2)

with col1:
    st.subheader("WORK A")
    st.write(row.get("WORK_A_WORK", ""))

with col2:
    st.subheader("WORK B")
    st.write(row.get("WORK_B_WORK", ""))

st.divider()

# Additional information
st.subheader("Similarity Features")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Text Similarity", f"{row['S_text']:.3f}")
c2.metric("Location Similarity", f"{row['S_geo']:.3f}")
c3.metric("Time Similarity", f"{row['S_time']:.3f}")
c4.metric("Structured Similarity", f"{row['S_struct']:.3f}")

st.write(
    f"Similarity group: **{row['similarity_group']}**"
)

st.divider()

st.subheader("Location / Work Details")

d1, d2 = st.columns(2)

with d1:
    st.write("**State A:**", row.get("WORK_A_STATE", ""))
    st.write("**Constituency A:**", row.get("WORK_A_CONSTITUENCY", ""))
    st.write("**City A:**", row.get("WORK_A_CITY", ""))
    st.write("**Block A:**", row.get("WORK_A_BLOCK", ""))
    st.write("**Village A:**", row.get("WORK_A_VILLAGE", ""))

with d2:
    st.write("**State B:**", row.get("WORK_B_STATE", ""))
    st.write("**Constituency B:**", row.get("WORK_B_CONSTITUENCY", ""))
    st.write("**City B:**", row.get("WORK_B_CITY", ""))
    st.write("**Block B:**", row.get("WORK_B_BLOCK", ""))
    st.write("**Village B:**", row.get("WORK_B_VILLAGE", ""))

st.divider()

st.subheader("Your Label")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button(
        "1 — DUPLICATE",
        use_container_width=True
    ):
        df.loc[i, "LABEL"] = 1
        df.to_csv(FILE, index=False)

        st.session_state.current_index = i + 1
        st.rerun()

with col2:
    if st.button(
        "0 — NOT DUPLICATE",
        use_container_width=True
    ):
        df.loc[i, "LABEL"] = 0
        df.to_csv(FILE, index=False)

        st.session_state.current_index = i + 1
        st.rerun()

with col3:
    if st.button(
        "SKIP",
        use_container_width=True
    ):
        st.session_state.current_index = i + 1
        st.rerun()

st.caption(
    "1 = same/substantially overlapping work | "
    "0 = different work | Skip = uncertain"
)