#!/bin/bash
# Simple HTTP server to serve the frontend

# Use Python's built-in HTTP server
cd "$(dirname "$0")"
echo "Starting HTTP server for frontend..."
echo "Frontend available at: http://localhost:8080"
echo "Press Ctrl+C to stop"
python3 -m http.server 8080
