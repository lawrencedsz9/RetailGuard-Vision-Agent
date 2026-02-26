"""
YOLO11-based retail theft detection processor.

This runs on EVERY frame for real-time tracking. It:
1. Detects people and common retail items (bags, bottles, etc.)
2. Assigns persistent track IDs so we follow the same person across frames
3. Computes "interactions" — when a person is physically close to an item
4. Maintains a short memory of interactions for behavioral context

The interaction detection is the KEY trigger for sending frames to Gemini.
Without an interaction, we don't waste API calls on empty scenes.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time


@dataclass
class Detection:
    """Single detected object in a frame."""
    label: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] pixel coordinates
    track_id: Optional[int] = None


class RetailTheftProcessor:
    """
    YOLO11-based processor detecting people, items, and tracking interactions.
    
    How it works:
    - Uses YOLO11n (nano) for speed — runs at ~30 FPS on most hardware
    - model.track() gives persistent IDs: Person #3 stays Person #3 across frames
    - TARGET_CLASSES are COCO dataset class IDs for theft-relevant objects
    - Interaction = person bounding box center within INTERACTION_DISTANCE of item center
    
    COCO class IDs used:
        0  = person
        24 = backpack
        25 = umbrella  
        26 = handbag
        28 = suitcase
        39 = bottle
    """

    TARGET_CLASSES = [0, 24, 25, 26, 28, 39]
    INTERACTION_DISTANCE = 150  # pixels — tune based on camera distance/resolution

    def __init__(self, model_path: str = "yolo11n.pt", conf_threshold: float = 0.45):
        """
        Args:
            model_path: Path to YOLO model weights. Downloads automatically on first run.
            conf_threshold: Minimum confidence to keep a detection (0.0-1.0).
                           Lower = more detections but more false positives.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.track_history: Dict[int, deque] = {}  # track_id -> position history
        self.interaction_memory: deque = deque(maxlen=200)

    def process(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame and return structured detection results.
        
        Args:
            frame: BGR image from OpenCV (numpy array, shape HxWx3)
            
        Returns:
            Dict with keys:
                - detections: all Detection objects
                - persons: only person detections
                - items: only item detections (bags, bottles, etc.)
                - interactions: list of person-item proximity events
                - frame_shape: (H, W, C) of the input frame
        """
        # model.track() enables multi-object tracking with persistent IDs
        results = self.model.track(
            frame,
            persist=True,           # Keep track IDs across frames
            classes=self.TARGET_CLASSES,
            conf=self.conf_threshold,
            verbose=False           # Suppress per-frame console output
        )

        detections = []
        persons = []
        items = []

        # results[0].boxes.id is None when no objects are tracked
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()

            for box, cls, conf, track_id in zip(boxes, classes, confs, track_ids):
                det = Detection(
                    label=self.model.names[int(cls)],
                    confidence=float(conf),
                    bbox=box.tolist(),
                    track_id=int(track_id)
                )
                detections.append(det)

                if int(cls) == 0:
                    persons.append(det)
                else:
                    items.append(det)

                # Store position history for trajectory analysis
                tid = int(track_id)
                if tid not in self.track_history:
                    self.track_history[tid] = deque(maxlen=50)
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                self.track_history[tid].append(center)

        interactions = self._analyze_interactions(persons, items)

        return {
            "detections": detections,
            "persons": persons,
            "items": items,
            "interactions": interactions,
            "frame_shape": frame.shape,
        }

    def _analyze_interactions(
        self, persons: List[Detection], items: List[Detection]
    ) -> List[Dict]:
        """
        Detect when persons are physically close to items.
        
        This is the core heuristic: if a person's center is within
        INTERACTION_DISTANCE pixels of an item's center, we flag it
        as an interaction. This triggers Gemini analysis.
        
        Note: This is a simple distance check. More sophisticated approaches
        could use IoU overlap or check if the item bbox is inside the person bbox.
        """
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

                if distance < self.INTERACTION_DISTANCE:
                    interaction = {
                        "person_id": person.track_id,
                        "item_type": item.label,
                        "item_id": item.track_id,
                        "distance": distance,
                        "person_bbox": person.bbox,
                        "item_bbox": item.bbox,
                    }
                    interactions.append(interaction)

                    self.interaction_memory.append(
                        {
                            "timestamp": time.time(),
                            "person_id": person.track_id,
                            "item": item.label,
                            "action": "nearby",
                            "distance": distance,
                        }
                    )

        return interactions

    def get_recent_interactions(self, person_id: int, seconds: int = 5) -> List[Dict]:
        """Get recent interactions for a specific person (for behavioral context)."""
        cutoff = time.time() - seconds
        return [
            i
            for i in self.interaction_memory
            if i["person_id"] == person_id and i["timestamp"] > cutoff
        ]

    def get_person_trajectory(self, track_id: int) -> List[tuple]:
        """Get movement path for a tracked person (useful for exit detection)."""
        return list(self.track_history.get(track_id, []))

    def draw_debug(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        Draw bounding boxes, labels, and interaction lines on frame.
        
        Color coding:
        - Green boxes: persons
        - Blue boxes: items
        - Red lines: active interactions (person near item)
        """
        annotated = frame.copy()

        for det in results["detections"]:
            x1, y1, x2, y2 = map(int, det.bbox)
            color = (0, 255, 0) if det.label == "person" else (255, 165, 0)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{det.label} {det.confidence:.2f} ID:{det.track_id}"
            # Background for text readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
            cv2.putText(
                annotated, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        # Draw interaction lines (red)
        for inter in results["interactions"]:
            p_center = (
                int((inter["person_bbox"][0] + inter["person_bbox"][2]) / 2),
                int((inter["person_bbox"][1] + inter["person_bbox"][3]) / 2),
            )
            i_center = (
                int((inter["item_bbox"][0] + inter["item_bbox"][2]) / 2),
                int((inter["item_bbox"][1] + inter["item_bbox"][3]) / 2),
            )
            cv2.line(annotated, p_center, i_center, (0, 0, 255), 2)
            # Distance label
            mid = ((p_center[0] + i_center[0]) // 2, (p_center[1] + i_center[1]) // 2)
            cv2.putText(
                annotated, f"{inter['distance']:.0f}px",
                mid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
            )

        # HUD: counts
        hud = f"Persons: {len(results['persons'])} | Items: {len(results['items'])} | Interactions: {len(results['interactions'])}"
        cv2.putText(annotated, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return annotated
