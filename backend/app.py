import os
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import json

app = Flask(__name__)
CORS(app)

# In-memory mock database for robust standalone demonstration & offline testing
# Harmonizing MODIS and VIIRS data across 2000-2026
np.random.seed(42)

def generate_mock_hotspots(n=1000):
    sensors = ['MODIS', 'VIIRS']
    satellites = {'MODIS': ['Terra', 'Aqua'], 'VIIRS': ['Suomi NPP', 'NOAA-20', 'NOAA-21']}
    
    # Bounding box roughly covering California / Western US or Global hotspots
    lats = np.random.uniform(32.0, 49.0, n)
    lons = np.random.uniform(-125.0, -100.0, n)
    
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2026, 12, 31)
    date_range_days = (end_date - start_date).days
    
    records = []
    for i in range(n):
        sensor = np.random.choice(sensors, p=[0.4, 0.6]) # VIIRS more frequent in recent years
        sat = np.random.choice(satellites[sensor])
        days_offset = np.random.randint(0, date_range_days)
        acq_date = start_date + timedelta(days=days_offset)
        
        # Harmonization factor: MODIS has coarser resolution (1km), VIIRS higher resolution (375m).
        # We apply harmonization to normalize FRP (Fire Radiative Power) and detection counts.
        raw_frp = np.random.exponential(scale=25.0) if sensor == 'VIIRS' else np.random.exponential(scale=45.0)
        harmonized_frp = raw_frp * (1.15 if sensor == 'MODIS' else 1.0) # Harmonization calibration
        
        records.append({
            "id": f"hotspot-{i+1}",
            "sensor": sensor,
            "latitude": round(float(lats[i]), 6),
            "longitude": round(float(lons[i]), 6),
            "brightness": round(float(np.random.uniform(300.0, 420.0)), 2),
            "scan": round(float(np.random.uniform(1.0, 2.0)), 2),
            "track": round(float(np.random.uniform(1.0, 2.0)), 2),
            "acq_date": acq_date.strftime("%Y-%m-%d"),
            "acq_time": f"{np.random.randint(0, 24):02d}{np.random.randint(0, 60):02d}",
            "satellite": sat,
            "confidence": int(np.random.randint(50, 100)),
            "frp": round(float(harmonized_frp), 2),
            "daynight": np.random.choice(['D', 'N'], p=[0.7, 0.3])
        })
    return records

MOCK_HOTSPOTS = generate_mock_hotspots(2000)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "TerraVision Python Flask Backend",
        "timestamp": datetime.utcnow().isoformat(),
        "hotspots_loaded": len(MOCK_HOTSPOTS)
    })

@app.route('/api/hotspots', methods=['GET'])
def get_hotspots():
    sensor = request.args.get('sensor') # MODIS, VIIRS, or all
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    min_frp = request.args.get('min_frp', type=float)
    lat_min = request.args.get('lat_min', type=float)
    lat_max = request.args.get('lat_max', type=float)
    lon_min = request.args.get('lon_min', type=float)
    lon_max = request.args.get('lon_max', type=float)
    limit = request.args.get('limit', default=500, type=int)
    
    filtered = MOCK_HOTSPOTS
    
    if sensor and sensor != 'all':
        filtered = [h for h in filtered if h['sensor'].upper() == sensor.upper()]
    if start_date:
        filtered = [h for h in filtered if h['acq_date'] >= start_date]
    if end_date:
        filtered = [h for h in filtered if h['acq_date'] <= end_date]
    if min_frp is not None:
        filtered = [h for h in filtered if h['frp'] >= min_frp]
    if lat_min is not None and lat_max is not None and lon_min is not None and lon_max is not None:
        filtered = [h for h in filtered if lat_min <= h['latitude'] <= lat_max and lon_min <= h['longitude'] <= lon_max]
        
    return jsonify({
        "count": len(filtered[:limit]),
        "total_matched": len(filtered),
        "data": filtered[:limit]
    })

@app.route('/api/calendar', methods=['GET'])
def get_burn_calendar():
    region = request.args.get('region', 'Western US')
    start_year = request.args.get('start_year', default=2010, type=int)
    end_year = request.args.get('end_year', default=2026, type=int)
    
    # Aggregate hotspots by year and month
    df = pd.DataFrame(MOCK_HOTSPOTS)
    df['dt'] = pd.to_datetime(df['acq_date'])
    df['year'] = df['dt'].dt.year
    df['month'] = df['dt'].dt.month
    
    df_filtered = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
    
    grouped = df_filtered.groupby(['year', 'month']).agg(
        total_hotspots=('id', 'count'),
        total_frp=('frp', 'sum'),
        mean_confidence=('confidence', 'mean')
    ).reset_index()
    
    # Calculate historical monthly median for anomaly detection
    monthly_medians = df.groupby('month')['id'].count().to_dict()
    
    calendar_matrix = []
    for year in range(start_year, end_year + 1):
        year_data = {"year": year, "months": {}}
        for month in range(1, 13):
            match = grouped[(grouped['year'] == year) & (grouped['month'] == month)]
            count = int(match['total_hotspots'].values[0]) if not match.empty else 0
            frp = float(match['total_frp'].values[0]) if not match.empty else 0.0
            
            historical_median = monthly_medians.get(month, 50) / (end_year - start_year + 1)
            anomaly_pct = ((count - historical_median) / historical_median * 100) if historical_median > 0 else 0
            
            year_data["months"][str(month)] = {
                "count": count,
                "frp": round(frp, 2),
                "historical_median": round(historical_median, 1),
                "anomaly_percentage": round(anomaly_pct, 1)
            }
        calendar_matrix.append(year_data)
        
    return jsonify({
        "region": region,
        "start_year": start_year,
        "end_year": end_year,
        "calendar": calendar_matrix
    })

@app.route('/api/export/csv', methods=['GET'])
def export_csv():
    sensor = request.args.get('sensor')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    filtered = MOCK_HOTSPOTS
    if sensor and sensor != 'all':
        filtered = [h for h in filtered if h['sensor'].upper() == sensor.upper()]
    if start_date:
        filtered = [h for h in filtered if h['acq_date'] >= start_date]
    if end_date:
        filtered = [h for h in filtered if h['acq_date'] <= end_date]
        
    df = pd.DataFrame(filtered)
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=terravision_harmonized_hotspots.csv"}
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
