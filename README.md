# TerraSense

A farmer-facing decision-support web application featuring an interactive 2.5D/3D "Civilization-style" parcel viewer. TerraSense fuses NASA Earth Observation data (NASA POWER, plus simulated SMAP, GPM, ECOSTRESS, and MODIS layers) with local soil profiles to divide a field into management zones, surface bare-ground and water/heat stress risks, and compare crop rotation strategies across a 24-month timeline.

## Tech Stack
- **Frontend**: Next.js (App Router, TypeScript, Tailwind CSS, Lucide icons, React Three Fiber / Three.js)
- **Backend**: Python (Flask, Flask-CORS, NumPy, Requests)

## Project Structure
- `terrasense/backend/`: Flask REST API (`app.py`), NASA POWER integration harness (`nasa_power.py`), and zoning/water-balance engine (`zone_engine.py`).
- `terrasense/frontend/`: Next.js App Router UI with the 2.5D isometric tile grid (`IsoGrid.tsx`), 24-month temporal scrubber (`TimelineSlider.tsx`), strategy comparison cards (`DecisionPanel.tsx`), and per-zone detail drawer (`TileDrawer.tsx`).
- `terrasense/README.md`: Detailed API contract and model notes.
