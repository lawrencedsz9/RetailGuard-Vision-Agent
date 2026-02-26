"""
RetailGuard AI — Local Testing Mode (No Stream SDK)

This is the LOCAL testing version that uses your webcam directly.
Use this for development and testing without needing Stream API credentials.

For the full hackathon submission, use agent.py (Vision Agents SDK).

Two modes:
  python main.py              Uses your webcam directly
  python main.py --video PATH  Uses a video file for demo/testing

The pipeline flow:
  1. Capture frame from webcam/video
  2. YOLO processes frame: detects people + items, tracks them, finds interactions
  3. IF there are interactions AND enough time has passed (0.5s):
     -> Send frame + YOLO context to Gemini for behavioral analysis
  4. IF Gemini says theft_detected with confidence > 0.75:
     -> AlertManager saves screenshot, logs to DB, plays voice alert
  5. Draw debug overlay on frame and display

Press 'q' to quit, 's' for manual screenshot.
"""

import asyncio
import argparse
import os
import sys
import time

import cv2
import numpy as np
from dotenv import load_dotenv

from processors.theft_processor import RetailTheftProcessor
from agents.theft_analyzer import GeminiTheftAnalyzer
from alerts.alert_manager import AlertManager


def load_config() -> dict:
    """Load API keys from .env file."""
    load_dotenv()
    config = {
        "gemini_api_key": os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", ""),
        "elevenlabs_api_key": os.getenv("ELEVENLABS_API_KEY", ""),
    }
    if not config["gemini_api_key"] or config["gemini_api_key"].startswith("your_"):
        print("WARNING: GOOGLE_API_KEY not set in .env file!")
        print("  Get one from: https://aistudio.google.com/app/apikey")
        print("  Running in YOLO-only mode (no behavioral analysis)\n")
    return config


class RetailGuardAI:
    """
    Main application class. Orchestrates the detection pipeline.
    
    Processing budget per frame:
    - YOLO: ~15-30ms (runs every frame)
    - Gemini: ~500-1500ms (runs every 0.5s, only on interactions)
    - Alert: ~100ms save + async voice (only on theft detection)
    """

    def __init__(self, config: dict, video_source=0):
        """
        Args:
            config: Dict with API keys
            video_source: 0 for webcam, or path to video file
        """
        self.config = config
        self.video_source = video_source

        # Initialize components
        self.processor = RetailTheftProcessor(
            model_path="yolo11n.pt",
            conf_threshold=0.45,
        )

        self.analyzer = None
        if config["gemini_api_key"] and not config["gemini_api_key"].startswith("your_"):
            self.analyzer = GeminiTheftAnalyzer(config["gemini_api_key"])
            print("[OK] Gemini analyzer initialized")
        else:
            print("[!!] Running without Gemini (YOLO-only mode)")

        self.alerts = AlertManager(
            db_path="incidents.db",
            clips_dir="clips",
            elevenlabs_api_key=config.get("elevenlabs_api_key"),
        )

        # Timing control
        self.last_analysis_time = 0
        self.analysis_interval = 0.5  # seconds between Gemini calls (2 FPS)
        self.frame_count = 0
        self.fps_timer = time.time()
        self.fps = 0

        # State
        self.current_alert = None
        self.alert_display_until = 0

    async def run(self):
        """Main loop: capture -> process -> display."""
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video source: {self.video_source}")
            print("  If using webcam, make sure it's connected and not in use.")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"\n{'='*60}")
        print(f"  RetailGuard AI - Live Monitoring")
        print(f"  Source: {'Webcam' if self.video_source == 0 else self.video_source}")
        print(f"  Resolution: {width}x{height}")
        print(f"  YOLO: Every frame | Gemini: Every {self.analysis_interval}s")
        print(f"  Press 'q' to quit")
        print(f"{'='*60}\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if self.video_source != 0:
                        # Video file ended — loop it for demo
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                self.frame_count += 1
                display_frame = await self._process_frame(frame)

                # Calculate FPS
                if self.frame_count % 30 == 0:
                    now = time.time()
                    self.fps = 30 / (now - self.fps_timer)
                    self.fps_timer = now

                # Draw FPS counter
                cv2.putText(
                    display_frame,
                    f"FPS: {self.fps:.1f}",
                    (display_frame.shape[1] - 130, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("RetailGuard AI", display_frame)

                # 'q' to quit, 's' to take manual screenshot
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    path = f"clips/manual_{int(time.time())}.jpg"
                    cv2.imwrite(path, frame)
                    print(f"  Manual screenshot saved: {path}")

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nProcessed {self.frame_count} frames. Incidents logged to incidents.db")

    async def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the full pipeline.
        
        Returns the annotated frame for display.
        """
        # ── Step 1: YOLO detection (every frame) ──────────────────
        yolo_results = self.processor.process(frame)

        # ── Step 2: Draw YOLO debug overlay ───────────────────────
        display = self.processor.draw_debug(frame.copy(), yolo_results)

        # ── Step 3: Gemini analysis (throttled, only on interactions) ──
        current_time = time.time()
        time_since_last = current_time - self.last_analysis_time

        if (
            self.analyzer
            and time_since_last > self.analysis_interval
            and yolo_results["interactions"]
        ):
            self.last_analysis_time = current_time

            analysis = await self.analyzer.analyze_behavior(
                frame, yolo_results
            )

            # Show analysis status on frame
            status_color = (0, 255, 0)  # Green = safe
            status_text = f"Gemini: {analysis.get('behavior_type', 'none')} ({analysis.get('confidence', 0):.0%})"

            if analysis.get("theft_detected") and analysis.get("confidence", 0) > 0.75:
                # ── Step 4: THEFT DETECTED — trigger alerts ───────
                status_color = (0, 0, 255)  # Red
                self.current_alert = analysis
                self.alert_display_until = current_time + 5  # Show for 5 seconds

                await self.alerts.trigger_alert(analysis, frame, yolo_results)

            cv2.putText(
                display, status_text, (10, display.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2,
            )

        # ── Draw active alert banner ─────────────────────────────
        if self.current_alert and current_time < self.alert_display_until:
            display = self._draw_alert_banner(display, self.current_alert)

        return display

    def _draw_alert_banner(self, frame: np.ndarray, analysis: dict) -> np.ndarray:
        """Draw a red alert banner at the top of the frame."""
        h, w = frame.shape[:2]

        # Semi-transparent red overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        behavior = analysis.get("behavior_type", "unknown").replace("_", " ").upper()
        confidence = analysis.get("confidence", 0)
        text = f"THEFT ALERT: {behavior} ({confidence:.0%})"
        cv2.putText(
            frame, text, (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
        )

        desc = analysis.get("suspect_description", "")[:80]
        if desc:
            cv2.putText(
                frame, desc, (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1,
            )

        return frame


def main():
    parser = argparse.ArgumentParser(
        description="RetailGuard AI - Local Testing Mode (webcam/video)"
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to video file (default: use webcam)"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.45,
        help="YOLO confidence threshold (default: 0.45)"
    )
    args = parser.parse_args()

    config = load_config()

    video_source = 0  # Default: webcam
    if args.video:
        if not os.path.exists(args.video):
            print(f"ERROR: Video file not found: {args.video}")
            sys.exit(1)
        video_source = args.video

    guard = RetailGuardAI(config=config, video_source=video_source)
    asyncio.run(guard.run())


if __name__ == "__main__":
    main()
