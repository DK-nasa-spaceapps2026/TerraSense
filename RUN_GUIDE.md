# Running TerraVision Locally

TerraVision is structured as a robust full-stack application:
- **Backend**: Python Flask (`backend/app.py`) providing harmonized MODIS and VIIRS active fire hotspots, anomaly calculations, and CSV exports.
- **Frontend**: Node.js Express (`frontend/server.js`) serving the responsive EJS dashboard with Google Maps integration, interactive filter controls, calendar heatmaps, and Chart.js analytics.
- **Database**: Supabase (PostgreSQL with PostGIS) schema (`database/schema.sql`).

## Quick Start Guide

### 1. Start the Python Flask Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```
*(The Flask server will start on `http://localhost:5000` with pre-populated mock harmonized hotspot data spanning 2000–2026).*

### 2. Start the Node.js Frontend
Open a new terminal window:
```bash
cd frontend
npm install
npm start
```
*(The Node.js server will start on `http://localhost:3000` and proxy requests to the Flask backend).*

### 3. Open in Browser
Visit **`http://localhost:3000`** in your browser to explore the interactive fire monitoring map, burning activity calendar heatmap, sensor filtering (MODIS vs. VIIRS), historical anomaly comparisons, and CSV export.
