-- TerraVision Database Schema for Supabase (PostgreSQL)
-- Enable PostGIS extension for spatial querying
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "postgis";

-- 1. Hotspots Table (Harmonized MODIS & VIIRS active fire detections)
CREATE TABLE IF NOT EXISTS public.hotspots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sensor VARCHAR(10) NOT NULL, -- 'MODIS' or 'VIIRS'
    latitude NUMERIC(9, 6) NOT NULL,
    longitude NUMERIC(9, 6) NOT NULL,
    brightness NUMERIC(6, 2), -- Brightness temperature (Kelvin)
    scan NUMERIC(5, 2),
    track NUMERIC(5, 2),
    acq_date DATE NOT NULL,
    acq_time VARCHAR(4) NOT NULL, -- HHMM
    satellite VARCHAR(2) NOT NULL, -- 'T', 'A', 'N', 'J1', 'J2'
    confidence INTEGER, -- 0-100 or nominal/high
    version VARCHAR(10),
    bright_t31 NUMERIC(6, 2),
    frp NUMERIC(8, 2) NOT NULL, -- Fire Radiative Power (MW) - key harmonization metric
    daynight VARCHAR(1) NOT NULL, -- 'D' or 'N'
    geom GEOMETRY(Point, 4326),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create spatial index for fast bounding box / area of interest queries
CREATE INDEX IF NOT EXISTS idx_hotspots_geom ON public.hotspots USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_hotspots_date ON public.hotspots (acq_date);
CREATE INDEX IF NOT EXISTS idx_hotspots_sensor ON public.hotspots (sensor);
CREATE INDEX IF NOT EXISTS idx_hotspots_frp ON public.hotspots (frp);

-- Trigger to automatically populate geom from latitude and longitude
CREATE OR REPLACE FUNCTION public.update_hotspot_geom()
RETURNS TRIGGER AS $$
BEGIN
    NEW.geom := ST_SetSRID(ST_MakePoint(NEW.longitude, NEW.latitude), 4326);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_hotspot_geom
    BEFORE INSERT OR UPDATE ON public.hotspots
    FOR EACH ROW
    EXECUTE FUNCTION public.update_hotspot_geom();

-- 2. Harmonized Monthly Burn Statistics Table (for Calendar & Anomalies)
CREATE TABLE IF NOT EXISTS public.monthly_burn_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    region_name VARCHAR(100) NOT NULL,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    total_hotspots INTEGER NOT NULL DEFAULT 0,
    total_frp NUMERIC(12, 2) NOT NULL DEFAULT 0.0,
    historical_median_hotspots NUMERIC(10, 2),
    anomaly_percentage NUMERIC(6, 2), -- percentage difference from historical median
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(region_name, year, month)
);

CREATE INDEX IF NOT EXISTS idx_monthly_stats_region_year ON public.monthly_burn_stats (region_name, year);

-- 3. Regions of Interest Table
CREATE TABLE IF NOT EXISTS public.regions_of_interest (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    boundary GEOMETRY(Polygon, 4326),
    center_lat NUMERIC(9, 6),
    center_lon NUMERIC(9, 6),
    zoom_level INTEGER DEFAULT 6,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_roi_boundary ON public.regions_of_interest USING GIST (boundary);
