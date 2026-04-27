import subprocess
import sys
import os
from pathlib import Path

# Step 1 — Train if artifacts missing
artifacts = Path("artifacts")
required = ["best_model.pkl", "scaler.pkl", "encoders.pkl",
            "feature_cols.json", "leaderboard.csv", "best_model_name.txt"]

if not artifacts.exists() or not all((artifacts/f).exists() for f in required):
    print("Training models...")
    subprocess.run([sys.executable, "ml_training.py"], check=True)
    print("Training complete!")

# Step 2 — Start FastAPI on port 8000 in background
subprocess.Popen([
    sys.executable, "-m", "uvicorn", "backend:app",
    "--host", "0.0.0.0", "--port", "8000"
])

import time
time.sleep(5)  # wait for backend to start

# Step 3 — Start Streamlit on port 10000 (Render's default)
port = os.environ.get("PORT", "10000")
subprocess.run([
    sys.executable, "-m", "streamlit", "run", "app.py",
    "--server.port", port,
    "--server.address", "0.0.0.0",
    "--server.headless", "true"
])
