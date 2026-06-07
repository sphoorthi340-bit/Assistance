"""
JARVIS System 4 — Dashboard API Server
========================================
A lightweight HTTP server to provide JSON data to the S4 Web Dashboard.
Runs on port 8080 by default.

Serves:
  /api/academic  -> Academic stats (CGPA, exams)
  /api/roadmap   -> MS Roadmap stats (Phases, universities)
  /api/stats     -> General system stats
  /api/focus     -> Current focus session state
"""

import os
import json
from http.server import SimpleHTTPRequestHandler
import socketserver
import threading

from backend.logger import get_logger
from backend.database import DatabaseManager
from memory.s4_memory import S4MemoryManager
from state.academic_manager import AcademicManager
from state.ms_roadmap import MSRoadmapManager
from backend.focus_guard import FocusGuard

logger = get_logger(__name__)

PORT = 8080

class DashboardAPIHandler(SimpleHTTPRequestHandler):
    """Custom request handler for the S4 Dashboard API."""

    def do_GET(self):
        # Allow CORS
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        # Route the request
        path = self.path
        response_data = {}

        try:
            if path == '/api/academic':
                response_data = self.server.academic_manager.get_dashboard_summary()
            elif path == '/api/roadmap':
                response_data = self.server.ms_roadmap.get_dashboard_summary()
            elif path == '/api/focus':
                response_data = self.server.focus_guard.get_session_stats()
            elif path == '/api/stats':
                daily = self.server.s4_memory.get_daily_log()
                weekly = self.server.s4_memory.get_weekly_state()
                response_data = {
                    "daily": {
                        "study_hours": daily.total_study_hours,
                        "tasks_completed": len(daily.tasks_completed),
                        "pomodoros": daily.pomodoros_completed,
                        "papers_read": len(daily.papers_read),
                    },
                    "weekly": {
                        "study_hours_total": weekly.study_hours_total,
                        "study_target_hours": weekly.study_target_hours,
                        "tasks_completed": weekly.tasks_completed,
                        "papers_read": weekly.papers_read,
                    }
                }
            else:
                response_data = {"error": "Invalid API endpoint"}
                
        except Exception as e:
            logger.error("Dashboard API error on %s: %s", path, e)
            response_data = {"error": str(e)}

        self.wfile.write(json.dumps(response_data).encode('utf-8'))

    def log_message(self, format, *args):
        # Suppress standard HTTP logs to keep terminal clean
        pass


def start_dashboard_api(
    db: DatabaseManager,
    s4_memory: S4MemoryManager,
    academic_manager: AcademicManager,
    ms_roadmap: MSRoadmapManager,
    focus_guard: FocusGuard,
    port: int = PORT
):
    """Start the dashboard API server in a background thread."""
    
    # Attach managers to a custom server subclass so the handler can access them
    class S4TCPServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        
    server = S4TCPServer(("", port), DashboardAPIHandler)
    server.db = db
    server.s4_memory = s4_memory
    server.academic_manager = academic_manager
    server.ms_roadmap = ms_roadmap
    server.focus_guard = focus_guard

    logger.info("Starting Dashboard API on port %d", port)
    
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
