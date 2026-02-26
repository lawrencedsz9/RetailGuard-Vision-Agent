"""
Alert Manager — handles the response pipeline when theft is detected.

Three responsibilities:
1. VOICE ALERTS: Text-to-speech via ElevenLabs (falls back to console beep)
2. EVIDENCE SAVING: Screenshots + JSON metadata saved to clips/ folder
3. INCIDENT LOGGING: SQLite database for searchable incident history

The SQLite database is perfect for a hackathon:
- Zero config (no server needed)
- Single file (incidents.db)
- SQL-queryable from Streamlit dashboard
- Portable for demo
"""

import sqlite3
import json
import cv2
import os
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np


class AlertManager:
    """
    Handles voice alerts, incident logging, and evidence collection.
    
    Usage:
        alerts = AlertManager()
        incident_id = await alerts.trigger_alert(gemini_analysis, frame, yolo_data)
    """

    def __init__(
        self,
        db_path: str = "incidents.db",
        clips_dir: str = "clips",
        elevenlabs_api_key: Optional[str] = None,
    ):
        self.db_path = db_path
        self.clips_dir = Path(clips_dir)
        self.clips_dir.mkdir(exist_ok=True)
        self.elevenlabs_api_key = elevenlabs_api_key
        self._init_database()

    def _init_database(self):
        """Create the incidents table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                behavior_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                description TEXT,
                screenshot_path TEXT,
                gemini_analysis TEXT,
                yolo_detections TEXT,
                status TEXT DEFAULT 'active'
            )
        """)
        conn.commit()
        conn.close()

    async def trigger_alert(
        self, analysis: dict, frame: np.ndarray, yolo_data: dict
    ) -> int:
        """
        Full alert pipeline: save evidence -> log to DB -> voice alert.
        
        Args:
            analysis: Output from GeminiTheftAnalyzer.analyze_behavior()
            frame: The video frame that triggered the alert
            yolo_data: Output from RetailTheftProcessor.process()
            
        Returns:
            incident_id: The database ID of the logged incident
        """
        behavior = analysis.get("behavior_type", "unknown")
        confidence = analysis.get("confidence", 0)
        print(
            f"\n{'='*60}"
            f"\n  ALERT: {behavior.upper()} detected ({confidence:.0%} confidence)"
            f"\n{'='*60}"
        )

        # Save evidence first (fast, local)
        evidence_paths = self._save_evidence(frame, analysis, yolo_data)

        # Log to database
        incident_id = self._log_incident(analysis, evidence_paths, yolo_data)
        print(f"  Incident #{incident_id} logged to database")

        # Voice alert (slowest, runs last)
        await self._voice_alert(analysis)

        return incident_id

    async def _voice_alert(self, analysis: dict):
        """
        Generate and play voice alert using ElevenLabs TTS.
        Falls back to console output if ElevenLabs is unavailable.
        """
        behavior = analysis.get("behavior_type", "suspicious activity").replace("_", " ")
        confidence_pct = int(analysis.get("confidence", 0) * 100)
        description = analysis.get("suspect_description", "")

        message = (
            f"Security alert. Possible {behavior} detected "
            f"with {confidence_pct} percent confidence. "
            f"{description}. Please check camera feed."
        )

        if self.elevenlabs_api_key:
            try:
                from elevenlabs import generate, play, set_api_key

                set_api_key(self.elevenlabs_api_key)
                audio = generate(
                    text=message,
                    voice="Bella",
                    model="eleven_multilingual_v2",
                )
                # Play in a thread to not block the event loop
                await asyncio.to_thread(play, audio)
                return
            except ImportError:
                print("  [Voice] ElevenLabs not installed, using console alert")
            except Exception as e:
                print(f"  [Voice] ElevenLabs error: {e}, using console alert")

        # Fallback: console alert
        print(f"  [VOICE ALERT] {message}")
        # System bell
        print("\a")

    def _save_evidence(self, frame: np.ndarray, analysis: dict, yolo_data: dict) -> dict:
        """Save screenshot and metadata JSON to clips/ directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        behavior = analysis.get("behavior_type", "unknown")

        # Save screenshot
        screenshot_name = f"{timestamp}_{behavior}.jpg"
        screenshot_path = self.clips_dir / screenshot_name
        cv2.imwrite(str(screenshot_path), frame)
        print(f"  Screenshot saved: {screenshot_path}")

        # Save full metadata
        metadata = {
            "timestamp": timestamp,
            "analysis": analysis,
            "yolo_summary": {
                "persons": len(yolo_data.get("persons", [])),
                "items": len(yolo_data.get("items", [])),
                "interactions": len(yolo_data.get("interactions", [])),
            },
            "screenshot": str(screenshot_path),
        }
        metadata_path = self.clips_dir / f"{timestamp}_{behavior}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        return {"screenshot": str(screenshot_path), "metadata": str(metadata_path)}

    def _log_incident(self, analysis: dict, evidence: dict, yolo_data: dict) -> int:
        """Insert incident record into SQLite database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Serialize yolo_data safely (Detection objects aren't JSON-serializable)
        yolo_summary = {
            "persons": len(yolo_data.get("persons", [])),
            "items": len(yolo_data.get("items", [])),
            "interactions_count": len(yolo_data.get("interactions", [])),
            "interactions": yolo_data.get("interactions", []),
        }

        c.execute(
            """
            INSERT INTO incidents 
            (timestamp, behavior_type, confidence, description, 
             screenshot_path, gemini_analysis, yolo_detections, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                datetime.now().isoformat(),
                analysis.get("behavior_type", "unknown"),
                analysis.get("confidence", 0),
                analysis.get("suspect_description", ""),
                evidence.get("screenshot", ""),
                json.dumps(analysis, default=str),
                json.dumps(yolo_summary, default=str),
                "active",
            ),
        )

        incident_id = c.lastrowid
        conn.commit()
        conn.close()
        return incident_id

    # ── Query Methods (used by dashboard) ──────────────────────────

    def search_incidents(
        self, query: str = None, date_filter: str = None, status: str = None
    ) -> List[Dict]:
        """Search incidents with optional filters."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        sql = "SELECT * FROM incidents WHERE 1=1"
        params = []

        if query:
            sql += " AND (behavior_type LIKE ? OR description LIKE ?)"
            params.extend([f"%{query}%", f"%{query}%"])
        if date_filter:
            sql += " AND date(timestamp) = date(?)"
            params.append(date_filter)
        if status:
            sql += " AND status = ?"
            params.append(status)

        sql += " ORDER BY timestamp DESC LIMIT 100"
        c.execute(sql, params)
        results = [dict(row) for row in c.fetchall()]
        conn.close()
        return results

    def get_stats(self) -> dict:
        """Get aggregate statistics for the dashboard."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("SELECT COUNT(*) FROM incidents")
        total = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM incidents WHERE date(timestamp) = date('now')")
        today = c.fetchone()[0]

        c.execute("SELECT behavior_type, COUNT(*) FROM incidents GROUP BY behavior_type")
        by_behavior = dict(c.fetchall())

        c.execute("SELECT AVG(confidence) FROM incidents")
        avg_conf = c.fetchone()[0] or 0

        c.execute("SELECT COUNT(*) FROM incidents WHERE status = 'active'")
        active = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM incidents WHERE status = 'reviewed'")
        reviewed = c.fetchone()[0]

        conn.close()

        return {
            "total_incidents": total,
            "today_incidents": today,
            "by_behavior": by_behavior,
            "avg_confidence": avg_conf,
            "active": active,
            "reviewed": reviewed,
        }

    def update_status(self, incident_id: int, new_status: str):
        """Update incident status (active -> reviewed -> dismissed)."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE incidents SET status = ? WHERE id = ?", (new_status, incident_id)
        )
        conn.commit()
        conn.close()

    def delete_incident(self, incident_id: int):
        """Delete an incident by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM incidents WHERE id = ?", (incident_id,))
        conn.commit()
        conn.close()
