"""
Retail Theft Detection Processor for Vision Agents SDK.

This is a custom Processor that plugs into the Vision Agents pipeline.
It receives every video frame from Stream's WebRTC edge network and:

1. Runs YOLO11 to detect people and items (bags, bottles, etc.)
2. Tracks objects across frames with persistent IDs
3. Detects "interactions" — when a person is close to an item
4. Draws bounding boxes, labels, and interaction lines on the frame
5. Emits TheftInteractionEvent when interactions are detected
6. Publishes annotated frames back to the video stream

The annotated frames (with bounding boxes) are what Gemini Realtime sees,
giving it visual context for behavioral analysis.

This follows the Vision Agents Processor pattern — modeled after the
SecurityCameraProcessor example from the SDK.
"""

import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

import av
import cv2
import numpy as np
from collections import deque

from vision_agents.core.events.base import PluginBaseEvent
from vision_agents.core.events.manager import EventManager
from vision_agents.core.processors.base_processor import VideoProcessorPublisher
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_track import QueuedVideoTrack
from vision_agents.core.warmup import Warmable

logger = logging.getLogger(__name__)


# ── Events ─────────────────────────────────────────────────────────
# Vision Agents uses PluginBaseEvent subclasses for processor -> agent communication

@dataclass
class TheftInteractionEvent(PluginBaseEvent):
    """Emitted when YOLO detects a person physically close to an item."""
    type: str = field(default="retail.theft_interaction", init=False)
    person_id: int = 0
    item_type: str = ""
    item_id: int = 0
    distance: float = 0.0
    timestamp: str = ""


@dataclass
class Detection:
    """A single detected object in a frame."""
    label: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    track_id: Optional[int] = None


class RetailTheftProcessor(VideoProcessorPublisher, Warmable[Optional[Any]]):
    """
    Vision Agents Processor for retail theft detection using YOLO11.

    Works with the Vision Agents SDK pipeline:
    - Receives frames from Stream's WebRTC edge
    - Processes with YOLO11 for detection + tracking
    - Annotates frames with bounding boxes
    - Emits events on suspicious interactions
    - Publishes annotated frames back for Gemini to analyze

    Follows the same pattern as SecurityCameraProcessor from the SDK.

    COCO class IDs used:
        0  = person
        24 = backpack
        25 = umbrella
        26 = handbag
        28 = suitcase
        39 = bottle
    """

    name = "retail_theft"

    TARGET_CLASSES = [0, 24, 25, 26, 28, 39]

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf_threshold: float = 0.45,
        interaction_distance: float = 150,
        fps: int = 5,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.interaction_distance = interaction_distance
        self.fps = fps

        # YOLO model — loaded asynchronously via Warmable
        self.yolo_model: Optional[Any] = None

        # Tracking state
        self.track_history: Dict[int, deque] = {}
        self.interaction_memory: deque = deque(maxlen=200)
        self.frame_count = 0

        # Thread pool for CPU-intensive YOLO inference
        self.executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="retail_yolo"
        )
        self._shutdown = False

        # Video track for publishing annotated frames
        self._video_track: QueuedVideoTrack = QueuedVideoTrack()
        self._video_forwarder: Optional[VideoForwarder] = None

        # Event system — EventManager + register events (SDK pattern)
        self.events = EventManager()
        self.events.register(TheftInteractionEvent)

        logger.info("🛡️ RetailTheftProcessor initialized")

    # ── Warmable: async model loading ──────────────────────────────

    async def on_warmup(self) -> Optional[Any]:
        """Load YOLO model asynchronously (called by SDK on startup)."""
        try:
            from ultralytics import YOLO

            loop = asyncio.get_event_loop()

            def load_yolo():
                model = YOLO(self.model_path)
                return model

            model = await loop.run_in_executor(self.executor, load_yolo)
            logger.info(f"✅ YOLO model loaded: {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"❌ YOLO model failed to load: {e}")
            return None

    def on_warmed_up(self, resource: Optional[Any]) -> None:
        """Set the loaded YOLO model."""
        self.yolo_model = resource

    # ── VideoProcessorPublisher: frame processing pipeline ─────────

    async def _process_and_add_frame(self, frame: av.VideoFrame):
        """Process a single frame: YOLO detect -> annotate -> publish."""
        try:
            self.frame_count += 1

            # Convert av.VideoFrame -> BGR numpy array
            frame_rgb = frame.to_ndarray(format="rgb24")
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Run YOLO detection in thread pool (CPU-bound)
            loop = asyncio.get_event_loop()
            annotated_bgr = await loop.run_in_executor(
                self.executor, self._detect_and_annotate, frame_bgr
            )

            # Convert back to RGB -> av.VideoFrame
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            processed_frame = av.VideoFrame.from_ndarray(annotated_rgb, format="rgb24")

            await self._video_track.add_frame(processed_frame)

        except Exception as e:
            logger.exception(f"❌ Frame processing failed: {e}")
            await self._video_track.add_frame(frame)

    def _detect_and_annotate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Synchronous YOLO detection + annotation (runs in thread pool)."""
        if not self.yolo_model:
            return frame_bgr

        results = self.yolo_model.track(
            frame_bgr,
            persist=True,
            classes=self.TARGET_CLASSES,
            conf=self.conf_threshold,
            verbose=False,
        )

        detections = []
        persons = []
        items = []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()

            for box, cls, conf, track_id in zip(boxes, classes, confs, track_ids):
                det = Detection(
                    label=self.yolo_model.names[int(cls)],
                    confidence=float(conf),
                    bbox=box.tolist(),
                    track_id=int(track_id),
                )
                detections.append(det)

                if int(cls) == 0:
                    persons.append(det)
                else:
                    items.append(det)

                # Track position history
                tid = int(track_id)
                if tid not in self.track_history:
                    self.track_history[tid] = deque(maxlen=50)
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                self.track_history[tid].append(center)

        # Detect interactions and emit events
        interactions = self._analyze_interactions(persons, items)

        # Draw annotations on frame
        annotated = self._draw_annotations(frame_bgr.copy(), detections, interactions)

        return annotated

    def _analyze_interactions(
        self, persons: List[Detection], items: List[Detection]
    ) -> List[Dict]:
        """Detect when persons are close to items and emit events."""
        interactions = []

        for person in persons:
            px1, py1, px2, py2 = person.bbox
            person_center = ((px1 + px2) / 2, (py1 + py2) / 2)

            for item in items:
                ix1, iy1, ix2, iy2 = item.bbox
                item_center = ((ix1 + ix2) / 2, (iy1 + iy2) / 2)

                distance = np.sqrt(
                    (person_center[0] - item_center[0]) ** 2
                    + (person_center[1] - item_center[1]) ** 2
                )

                if distance < self.interaction_distance:
                    interaction = {
                        "person_id": person.track_id,
                        "item_type": item.label,
                        "item_id": item.track_id,
                        "distance": distance,
                        "person_bbox": person.bbox,
                        "item_bbox": item.bbox,
                    }
                    interactions.append(interaction)

                    # Record in memory
                    self.interaction_memory.append({
                        "timestamp": time.time(),
                        "person_id": person.track_id,
                        "item": item.label,
                        "distance": distance,
                    })

                    # Emit event via SDK EventManager (send, not emit)
                    self.events.send(TheftInteractionEvent(
                        plugin_name="retail_theft",
                        person_id=person.track_id,
                        item_type=item.label,
                        item_id=item.track_id,
                        distance=distance,
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    ))

        return interactions

    def _draw_annotations(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        interactions: List[Dict],
    ) -> np.ndarray:
        """Draw bounding boxes, labels, and interaction lines on the frame."""

        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            color = (0, 255, 0) if det.label == "person" else (255, 165, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{det.label} {det.confidence:.2f} #{det.track_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
            cv2.putText(
                frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

        # Draw interaction lines (red) between person and item
        for inter in interactions:
            p_center = (
                int((inter["person_bbox"][0] + inter["person_bbox"][2]) / 2),
                int((inter["person_bbox"][1] + inter["person_bbox"][3]) / 2),
            )
            i_center = (
                int((inter["item_bbox"][0] + inter["item_bbox"][2]) / 2),
                int((inter["item_bbox"][1] + inter["item_bbox"][3]) / 2),
            )
            cv2.line(frame, p_center, i_center, (0, 0, 255), 2)
            mid = ((p_center[0] + i_center[0]) // 2, (p_center[1] + i_center[1]) // 2)
            cv2.putText(
                frame, f"{inter['distance']:.0f}px",
                mid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
            )

        # HUD overlay
        n_persons = sum(1 for d in detections if d.label == "person")
        n_items = sum(1 for d in detections if d.label != "person")
        n_inter = len(interactions)
        hud = f"Persons: {n_persons} | Items: {n_items} | Interactions: {n_inter}"
        cv2.putText(frame, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # RetailGuard branding
        cv2.putText(
            frame, "RetailGuard AI", (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        return frame

    # ── VideoProcessorPublisher interface ──────────────────────────

    async def process_video(
        self,
        track,
        participant_id: Optional[str],
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """Set up video processing pipeline (called by SDK)."""
        if shared_forwarder is not None:
            self._video_forwarder = shared_forwarder
            self._video_forwarder.add_frame_handler(
                self._process_and_add_frame,
                fps=float(self.fps),
                name="retail_theft",
            )
        else:
            self._video_forwarder = VideoForwarder(
                track, max_buffer=30, fps=self.fps, name="retail_theft_forwarder"
            )
            self._video_forwarder.add_frame_handler(self._process_and_add_frame)

        logger.info("✅ RetailTheft video processing started")

    async def stop_processing(self) -> None:
        """Stop processing video tracks."""
        if self._video_forwarder:
            await self._video_forwarder.stop()

    def publish_video_track(self):
        """Return the video track for publishing."""
        return self._video_track

    # ── Query methods (used by LLM function calls) ─────────────────

    def get_recent_interactions(self, seconds: int = 10) -> List[Dict]:
        """Get recent person-item interactions."""
        cutoff = time.time() - seconds
        return [i for i in self.interaction_memory if i["timestamp"] > cutoff]

    def get_interaction_count(self) -> int:
        """Total interactions detected in the current session."""
        return len(self.interaction_memory)

    def state(self) -> Dict[str, Any]:
        """Return current processor state for LLM context."""
        return {
            "frames_processed": self.frame_count,
            "total_interactions": len(self.interaction_memory),
            "tracked_objects": len(self.track_history),
        }

    async def close(self):
        """Clean up resources."""
        self._shutdown = True
        if self._video_forwarder is not None:
            await self._video_forwarder.stop()
        self.executor.shutdown(wait=False)
        self.track_history.clear()
        self.interaction_memory.clear()
        logger.info("🛑 RetailTheftProcessor closed")
