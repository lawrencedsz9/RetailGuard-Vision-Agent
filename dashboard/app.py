"""
RetailGuard AI — Streamlit Security Dashboard

Run with:  streamlit run dashboard/app.py

This dashboard reads from the SQLite database (incidents.db) and shows:
- Live stats (total incidents, today's count, avg confidence)
- Recent incidents with screenshots and Gemini analysis
- Search and filter capabilities
- Mark as reviewed / delete functionality

The dashboard is a SEPARATE process from the main detection pipeline.
They communicate through the shared SQLite database file.
"""

import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
import sys
import os

# Add parent directory to path so we can import AlertManager
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alerts.alert_manager import AlertManager

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="RetailGuard AI Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Database Path (relative to project root) ─────────────────────
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "incidents.db")
alerts = AlertManager(db_path=DB_PATH, clips_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "clips"))


# ── Sidebar: Search & Filter ─────────────────────────────────────
st.sidebar.title("🔍 Search & Filter")
search_term = st.sidebar.text_input("Search incidents", placeholder="e.g., concealment, bottle")
date_filter = st.sidebar.date_input("Filter by date", value=None)
status_filter = st.sidebar.selectbox("Status", ["All", "active", "reviewed", "dismissed"])

# Auto-refresh
auto_refresh = st.sidebar.checkbox("Auto-refresh (10s)", value=False)
if auto_refresh:
    import time
    st.sidebar.info("Dashboard refreshes every 10 seconds")
    time.sleep(10)
    st.rerun()

st.sidebar.divider()
st.sidebar.caption("RetailGuard AI v1.0 | Hackathon 2025")

# ── Header ────────────────────────────────────────────────────────
st.title("🛡️ RetailGuard AI — Security Dashboard")
st.caption("Real-time intelligent surveillance for retail theft prevention")

# ── Stats Row ─────────────────────────────────────────────────────
stats = alerts.get_stats()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Incidents", stats["total_incidents"])
col2.metric("Today", stats["today_incidents"])
col3.metric("Avg Confidence", f"{stats['avg_confidence']:.0%}" if stats['avg_confidence'] else "N/A")
col4.metric("Active Alerts", stats["active"])

# ── Behavior Breakdown ───────────────────────────────────────────
if stats["by_behavior"]:
    st.subheader("Incident Breakdown")
    behavior_cols = st.columns(len(stats["by_behavior"]))
    colors = {"concealment": "🔴", "unpaid_exit": "🟠", "consumption": "🟡"}
    for i, (behavior, count) in enumerate(stats["by_behavior"].items()):
        icon = colors.get(behavior, "⚪")
        behavior_cols[i].metric(f"{icon} {behavior.replace('_', ' ').title()}", count)

st.divider()

# ── Refresh Button ────────────────────────────────────────────────
col_refresh, col_spacer = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 Refresh Data"):
        st.rerun()

# ── Incidents List ────────────────────────────────────────────────
st.subheader("Recent Incidents")

# Build search params
search_q = search_term if search_term else None
date_q = date_filter.isoformat() if date_filter else None
status_q = status_filter if status_filter != "All" else None

incidents = alerts.search_incidents(query=search_q, date_filter=date_q, status=status_q)

if incidents:
    for incident in incidents:
        # Color-code by behavior type
        behavior = incident.get("behavior_type", "unknown")
        confidence = incident.get("confidence", 0)
        timestamp = incident.get("timestamp", "")
        status = incident.get("status", "active")
        incident_id = incident.get("id")

        # Status badge
        status_icon = {"active": "🔴", "reviewed": "✅", "dismissed": "⬜"}.get(status, "⚪")

        header = f"{status_icon} {timestamp} — {behavior.upper()} ({confidence:.0%})"

        with st.expander(header, expanded=(status == "active")):
            col_img, col_info = st.columns([1, 2])

            with col_img:
                screenshot = incident.get("screenshot_path", "")
                if screenshot and Path(screenshot).exists():
                    st.image(screenshot, use_container_width=True, caption="Detection Screenshot")
                else:
                    st.info("📷 No screenshot available")

            with col_info:
                st.write(f"**Description:** {incident.get('description', 'N/A')}")
                st.write(f"**Status:** `{status}`")
                st.write(f"**Confidence:** {confidence:.2%}")

                # Show Gemini analysis details
                gemini_raw = incident.get("gemini_analysis", "")
                if gemini_raw:
                    try:
                        analysis = json.loads(gemini_raw)
                        with st.container():
                            st.write("**Gemini Analysis:**")
                            st.write(f"- **Reasoning:** {analysis.get('reasoning', 'N/A')}")
                            if analysis.get("suspect_description"):
                                st.write(f"- **Suspect:** {analysis['suspect_description']}")
                    except (json.JSONDecodeError, TypeError):
                        st.text(gemini_raw)

                # Action buttons
                btn_cols = st.columns(3)
                with btn_cols[0]:
                    if status == "active" and st.button("✅ Mark Reviewed", key=f"review_{incident_id}"):
                        alerts.update_status(incident_id, "reviewed")
                        st.rerun()
                with btn_cols[1]:
                    if status != "dismissed" and st.button("⬜ Dismiss", key=f"dismiss_{incident_id}"):
                        alerts.update_status(incident_id, "dismissed")
                        st.rerun()
                with btn_cols[2]:
                    if st.button("🗑️ Delete", key=f"delete_{incident_id}"):
                        alerts.delete_incident(incident_id)
                        st.rerun()
else:
    st.info(
        "No incidents recorded yet.\n\n"
        "Start the AI agent with `python main.py --local` to begin monitoring."
    )

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption(
    "RetailGuard AI | Built with YOLO11 + Gemini Flash + ElevenLabs + Streamlit | "
    f"Database: {DB_PATH}"
)
