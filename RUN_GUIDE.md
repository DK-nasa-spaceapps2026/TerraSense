# Running TerraSense Locally

TerraSense runs with two local servers:
- **Backend**: Python Flask (`terrasense/backend/app.py`) on `http://localhost:5001`
- **Frontend**: Next.js (`terrasense/frontend`) on `http://localhost:3001` (proxies `/api/*` to Flask)

## Quick Start Guide

### 1. Start the Python Flask Backend
```bash
cd terrasense/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```
*(No API keys are required—the backend runs out-of-the-box with deterministic mock data. To fetch live NASA POWER climate data with automatic mock fallback, run `TERRASENSE_LIVE=1 python app.py`.)*

### 2. Start the Next.js Frontend
Open a new terminal window:
```bash
cd terrasense/frontend
npm install
npm run dev
```

### 3. Open in Browser
Visit **`http://localhost:3001`** in your browser to explore the 2.5D parcel grid, scrub through the 24-month observed/forecast timeline, test climate what-if scenarios, and compare crop rotation strategies.

### Optional: Backend Smoke Test
```bash
cd terrasense/backend
source .venv/bin/activate
python smoke_test.py
```
