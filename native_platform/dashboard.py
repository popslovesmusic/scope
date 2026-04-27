import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from .run_state import find_latest_feedback_trace, tail_jsonl, load_memory_summary

def load_data():
    path = find_latest_feedback_trace()
    if not path:
        return None
    
    records = tail_jsonl(path, 1000) # Load up to last 1000 frames
    if not records:
        return None
    
    df = pd.DataFrame(records)
    return df, path

def main():
    st.set_page_config(page_title="Wave-Residue Platform Dashboard", layout="wide")
    st.title("🌊 Native Wave-Residue Platform Dashboard")

    data = load_data()
    if data is None:
        st.warning("No feedback traces found in logs/ directory. Run the platform first.")
        return

    df, log_path = data
    latest = df.iloc[-1]

    # Sidebar: System Status
    st.sidebar.header("System Status")
    st.sidebar.info(f"Monitoring: {os.path.basename(log_path)}")
    st.sidebar.metric("Latest Hex", latest['hex'])
    st.sidebar.metric("Frames Processed", int(latest['t']))
    
    mem = load_memory_summary()
    st.sidebar.header("Residue Memory")
    st.sidebar.metric("Qualified Imprints", mem.get('qualified_residue_count', 0))
    st.sidebar.metric("Active Residues", mem.get('committed_count', 0))

    # Top Row: Physics Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Coupling (C)")
        st.line_chart(df[['t', 'C']].set_index('t'))
    with col2:
        st.subheader("Imbalance (E)")
        st.line_chart(df[['t', 'E']].set_index('t'))
    with col3:
        st.subheader("Control Bias")
        st.line_chart(df[['t', 'control_pattern']].set_index('t'))

    # Middle Row: Reasoning Metrics
    st.subheader("Reasoning: Caution vs Recovery")
    st.line_chart(df[['t', 'caution', 'recovery']].set_index('t'))

    # Bottom Row: Residue & Velocity
    col4, col5 = st.columns([2, 1])
    with col4:
        st.subheader("Trajectory Velocity (V)")
        # V is a list, let's take means or plot components
        v_df = pd.DataFrame(df['V'].tolist(), columns=['V_energy', 'V_gradient', 'V_variance'])
        v_df['t'] = df['t']
        st.line_chart(v_df.set_index('t'))
    
    with col5:
        st.subheader("Imprint Rate")
        committed_count = df['residue_committed'].sum()
        total_count = len(df)
        rate = committed_count / total_count
        st.metric("Success Rate", f"{rate:.1%}")
        st.progress(rate)

    # Raw Data view
    if st.checkbox("Show raw trace data"):
        st.write(df.tail(20))

if __name__ == "__main__":
    main()
