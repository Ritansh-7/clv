import subprocess
import sys
import os
from pathlib import Path
import streamlit as st

# ── Run training if artifacts don't exist ──────────────
artifacts = Path("artifacts")
required = [
    "best_model.pkl", "scaler.pkl", "encoders.pkl",
    "feature_cols.json", "leaderboard.csv", "best_model_name.txt"
]

if not artifacts.exists() or not all((artifacts / f).exists() for f in required):
    with st.spinner("🔧 First-time setup: Training ML models... (this takes 2-3 mins)"):
        result = subprocess.run(
            [sys.executable, "ml_training.py"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            st.error(f"Training failed:\n{result.stderr}")
            st.stop()
        st.success("✅ Models trained successfully!")
        st.rerun()

# ── Start FastAPI backend if not running ───────────────
import requests

def is_backend_running():
    try:
        r = requests.get("http://127.0.0.1:8000/", timeout=3)
        return r.json().get("loaded", False)
    except:
        return False

if not is_backend_running():
    with st.spinner("🚀 Starting backend server..."):
        subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend:app",
             "--host", "127.0.0.1", "--port", "8000"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        import time
        for _ in range(15):          # wait up to 15 seconds
            time.sleep(1)
            if is_backend_running():
                break
        
        if not is_backend_running():
            st.error("❌ Backend failed to start. Check logs.")
            st.stop()

# ── Now run the actual dashboard ───────────────────────
exec(open("app.py").read())