"""
Krishi Rakshak AI - SQLite Database Module
Team: Codex Verse
Description: Lightweight, local scan history storage and analytics engine.
             Does not store farmer images permanently for privacy and efficiency.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import config


class ScanHistoryDB:
    """Local SQLite database manager for farmer scan history and metrics."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or config.DB_PATH
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a thread-safe connection to SQLite with row dictionary factory."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database schema if it doesn't already exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scan_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    crop_name TEXT NOT NULL,
                    disease_name TEXT NOT NULL,
                    is_healthy BOOLEAN NOT NULL,
                    confidence REAL NOT NULL,
                    confidence_level TEXT NOT NULL,
                    weather_temp REAL,
                    weather_humidity REAL,
                    notes TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_scan_timestamp ON scan_history(timestamp)
            """)
            conn.commit()

    def record_scan(
        self,
        crop_name: str,
        disease_name: str,
        is_healthy: bool,
        confidence: float,
        confidence_level: str,
        weather_temp: Optional[float] = None,
        weather_humidity: Optional[float] = None,
        notes: str = ""
    ) -> int:
        """
        Record a scan result into SQLite.
        Returns the newly created scan record ID.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO scan_history (
                    timestamp, crop_name, disease_name, is_healthy,
                    confidence, confidence_level, weather_temp, weather_humidity, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, crop_name, disease_name, int(is_healthy),
                float(confidence), confidence_level, weather_temp, weather_humidity, notes
            ))
            conn.commit()
            return cursor.lastrowid

    def get_recent_scans(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent scans ordered by timestamp descending."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, crop_name, disease_name, is_healthy,
                       confidence, confidence_level, weather_temp, weather_humidity, notes
                FROM scan_history
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_analytics(self) -> Dict[str, Any]:
        """
        Compute dashboard summary analytics:
        - Total scans
        - Healthy vs Diseased count
        - Average confidence
        - Most frequently diagnosed conditions
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM scan_history")
            total_scans = cursor.fetchone()[0]
            
            if total_scans == 0:
                return {
                    "total_scans": 0,
                    "healthy_count": 0,
                    "diseased_count": 0,
                    "avg_confidence": 0.0,
                    "top_diseases": [],
                    "confidence_breakdown": {"High": 0, "Medium": 0, "Low": 0}
                }

            cursor.execute("SELECT COUNT(*) FROM scan_history WHERE is_healthy = 1")
            healthy_count = cursor.fetchone()[0]
            diseased_count = total_scans - healthy_count

            cursor.execute("SELECT AVG(confidence) FROM scan_history")
            avg_confidence = cursor.fetchone()[0] or 0.0

            cursor.execute("""
                SELECT disease_name, COUNT(*) as count
                FROM scan_history
                WHERE is_healthy = 0
                GROUP BY disease_name
                ORDER BY count DESC
                LIMIT 5
            """)
            top_diseases = [{"name": row["disease_name"], "count": row["count"]} for row in cursor.fetchall()]

            cursor.execute("""
                SELECT confidence_level, COUNT(*) as count
                FROM scan_history
                GROUP BY confidence_level
            """)
            conf_map = {row["confidence_level"]: row["count"] for row in cursor.fetchall()}

            return {
                "total_scans": total_scans,
                "healthy_count": healthy_count,
                "diseased_count": diseased_count,
                "avg_confidence": round(float(avg_confidence), 2),
                "top_diseases": top_diseases,
                "confidence_breakdown": {
                    "High": conf_map.get("High", 0),
                    "Medium": conf_map.get("Medium", 0),
                    "Low": conf_map.get("Low", 0)
                }
            }

    def clear_history(self) -> bool:
        """Clear all records from scan_history table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM scan_history")
            conn.commit()
            return True
