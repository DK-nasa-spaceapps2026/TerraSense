# TerraVision: Harmonized Burning Activity Calendar & Fire Monitoring Platform

TerraVision is a comprehensive full-stack web application designed for early warning and emergency responders, scientists, land managers, and firefighters. It harmonizes NASA Earthdata FIRMS active fire hotspots from MODIS and VIIRS sensors, presenting a consistent time series, burning activity calendars, anomaly detection against historical averages, and interactive mapping via Google Maps.

## Tech Stack
- **Frontend**: Node.js / Express with modern web UI (HTML5/CSS3/JavaScript, Google Maps API, Chart.js for data visualizations)
- **Backend**: Python Flask REST API
- **Database**: Supabase (PostgreSQL with PostGIS extensions for spatial queries)

## Project Structure
- `backend/`: Python Flask application, API routes, database models, and mock data generator / FIRMS harmonization processor.
- `frontend/`: Node.js Express server serving the dashboard and integrating Google Maps.
- `database/`: Supabase migration scripts and schema definitions.
- `data/`: Sample datasets and sample CSV exporters.
