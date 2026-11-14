const express = require('express');
const axios = require('axios');
const path = require('path');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;
const FLASK_BACKEND_URL = process.env.FLASK_BACKEND_URL || 'http://localhost:5000';

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.get('/', async (req, res) => {
    try {
        const sensor = req.query.sensor || 'all';
        const start_date = req.query.start_date || '2023-01-01';
        const end_date = req.query.end_date || '2026-12-31';
        
        // Fetch hotspots from Flask backend
        const hotspotsRes = await axios.get(`${FLASK_BACKEND_URL}/api/hotspots`, {
            params: { sensor, start_date, end_date, limit: 300 }
        });
        
        // Fetch calendar burn statistics
        const calendarRes = await axios.get(`${FLASK_BACKEND_URL}/api/calendar`, {
            params: { start_year: 2018, end_year: 2026 }
        });

        res.render('index', {
            hotspots: hotspotsRes.data.data,
            totalCount: hotspotsRes.data.total_matched,
            calendarData: calendarRes.data.calendar,
            googleMapsApiKey: process.env.GOOGLE_MAPS_API_KEY || 'AIzaSyMockKeyForTerraVisionDemo',
            filters: { sensor, start_date, end_date }
        });
    } catch (error) {
        console.error('Error communicating with Flask backend:', error.message);
        res.render('index', {
            hotspots: [],
            totalCount: 0,
            calendarData: [],
            googleMapsApiKey: process.env.GOOGLE_MAPS_API_KEY || 'AIzaSyMockKeyForTerraVisionDemo',
            filters: { sensor: 'all', start_date: '2023-01-01', end_date: '2026-12-31' },
            error: 'Could not connect to Python Flask backend. Please ensure backend is running on port 5000.'
        });
    }
});

app.get('/api/proxy/export', async (req, res) => {
    try {
        const response = await axios.get(`${FLASK_BACKEND_URL}/api/export/csv`, {
            params: req.query,
            responseType: 'stream'
        });
        res.setHeader('Content-Disposition', 'attachment;filename=terravision_hotspots.csv');
        res.setHeader('Content-Type', 'text/csv');
        response.data.pipe(res);
    } catch (error) {
        res.status(500).send('Export failed');
    }
});

app.listen(PORT, () => {
    console.log(`TerraVision Node.js frontend running at http://localhost:${PORT}`);
});
