"""
Gemini-powered behavioral theft analysis.

This is the "brain" of RetailGuard. While YOLO tells us WHERE objects are,
Gemini tells us WHAT is happening — is a person concealing an item? Walking
toward an exit? Opening a product?

Architecture decisions:
- Runs at 2 FPS (not every frame) to control API costs
- Only called when YOLO detects an interaction (person near item)
- Uses structured JSON output for reliable parsing
- Temperature 0.1 for consistent, conservative judgments
- Requires >0.75 confidence to trigger alerts (reduce false positives)

The prompt engineering is critical here. We give Gemini:
1. The raw frame (JPEG image)
2. Structured context from YOLO (what objects, where, interactions)
3. Clear behavior definitions (concealment, unpaid exit, consumption)
4. Explicit instruction to be conservative (false positives are costly)
"""

import base64
import json
import asyncio
import time
from io import BytesIO
from typing import Callable, Optional, Dict, List

import cv2
import numpy as np
from PIL import Image


class GeminiTheftAnalyzer:
    """
    Uses Google Gemini to analyze video frames for theft behaviors.
    
    Behavior types detected:
    - CONCEALMENT: Hiding items in pockets, bags, clothing
    - UNPAID_EXIT: Moving toward store exit with merchandise
    - CONSUMPTION: Opening/using products before payment
    
    Usage:
        analyzer = GeminiTheftAnalyzer(api_key="your-key")
        result = await analyzer.analyze_behavior(frame, yolo_context)
        if result['theft_detected']:
            print(f"Theft: {result['behavior_type']}")
    """

    SYSTEM_PROMPT = """You are a retail security AI analyzing surveillance footage.
You must be CONSERVATIVE — false positives cause real harm to innocent people.
Only flag behavior as theft when you are highly confident (>0.75).
Consider normal shopping behaviors: examining items, comparing products, 
putting items back on shelves. These are NOT theft."""

    def __init__(self, api_key: str):
        """
        Args:
            api_key: Google AI Studio API key (get from aistudio.google.com/app/apikey)
        """
        from google import genai
        from google.genai import types
        
        self.client = genai.Client(api_key=api_key)
        self.types = types
        self.model = "gemini-2.0-flash"  # Fast + vision capable
        self.analysis_history: List[Dict] = []
        self._last_analysis_time = 0

    async def analyze_behavior(
        self,
        frame: np.ndarray,
        yolo_context: dict,
        callback: Optional[Callable] = None,
    ) -> dict:
        """
        Analyze a frame for theft behaviors using Gemini Vision.
        
        Args:
            frame: BGR image from OpenCV
            yolo_context: Output from RetailTheftProcessor.process()
            callback: Optional async function called when theft is detected
            
        Returns:
            Dict with keys:
                - theft_detected: bool
                - behavior_type: "concealment" | "unpaid_exit" | "consumption" | "none"
                - confidence: 0.0-1.0
                - suspect_description: human-readable description
                - reasoning: why this was flagged (or not)
        """
        from google.genai import types

        # Encode frame as JPEG bytes for the API
        img_bytes = self._frame_to_jpeg_bytes(frame)

        # Build text context from YOLO detections
        context_str = self._build_context(yolo_context)

        prompt = f"""Analyze this surveillance frame for THEFT BEHAVIORS.

OBJECT DETECTION CONTEXT (from YOLO):
{context_str}

Look for these specific behaviors:
1. CONCEALMENT: Person placing items into pockets, bags, or under clothing without heading to checkout
2. UNPAID_EXIT: Person moving toward store exit/door while carrying unpaid merchandise
3. CONSUMPTION: Person opening, eating, drinking, or using a product before paying

Respond with ONLY this JSON (no markdown, no code blocks):
{{
    "theft_detected": true or false,
    "behavior_type": "concealment" or "unpaid_exit" or "consumption" or "none",
    "confidence": 0.0 to 1.0,
    "suspect_description": "brief description of what person is doing",
    "reasoning": "explain your analysis"
}}

IMPORTANT RULES:
- Default to "none" unless you see clear evidence
- Normal shopping (browsing, examining items, carrying a basket) is NOT theft
- Only set theft_detected=true if confidence > 0.75
- When uncertain, err on the side of caution (no theft)"""

        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt),
                            types.Part.from_bytes(
                                mime_type="image/jpeg",
                                data=img_bytes,
                            ),
                        ],
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,  # Low temperature = consistent, conservative
                    system_instruction=self.SYSTEM_PROMPT,
                ),
            )

            # Parse the JSON response, stripping any markdown fencing
            raw_text = response.text.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[1]  # Remove ```json line
                raw_text = raw_text.rsplit("```", 1)[0]  # Remove closing ```

            result = json.loads(raw_text)
            result["timestamp"] = time.time()
            result["yolo_persons"] = len(yolo_context.get("persons", []))
            result["yolo_items"] = len(yolo_context.get("items", []))
            result["yolo_interactions"] = len(yolo_context.get("interactions", []))

            # Keep history for dashboard stats
            self.analysis_history.append(result)
            if len(self.analysis_history) > 500:
                self.analysis_history = self.analysis_history[-250:]

            # Fire callback if theft detected
            if result.get("theft_detected") and callback:
                await callback(result)

            return result

        except json.JSONDecodeError as e:
            print(f"[GeminiAnalyzer] JSON parse error: {e}")
            print(f"[GeminiAnalyzer] Raw response: {raw_text[:200]}")
            return self._error_result(f"JSON parse error: {e}")
        except Exception as e:
            print(f"[GeminiAnalyzer] Analysis error: {e}")
            return self._error_result(str(e))

    def _frame_to_jpeg_bytes(self, frame: np.ndarray) -> bytes:
        """Convert OpenCV BGR frame to JPEG bytes for Gemini API."""
        # Resize large frames to save bandwidth and cost
        h, w = frame.shape[:2]
        max_dim = 1024
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buffer.tobytes()

    def _build_context(self, yolo_context: dict) -> str:
        """Convert YOLO output to a text description for the Gemini prompt."""
        persons = yolo_context.get("persons", [])
        items = yolo_context.get("items", [])
        interactions = yolo_context.get("interactions", [])
        shape = yolo_context.get("frame_shape", (0, 0, 0))

        lines = [
            f"Frame dimensions: {shape[1]}x{shape[0]} pixels",
            f"People detected: {len(persons)}",
            f"Items detected: {len(items)}",
        ]

        if persons:
            lines.append("People:")
            for p in persons:
                x1, y1, x2, y2 = p.bbox
                lines.append(
                    f"  - Person ID {p.track_id} at [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] conf={p.confidence:.2f}"
                )

        if items:
            lines.append("Items:")
            for it in items:
                x1, y1, x2, y2 = it.bbox
                lines.append(
                    f"  - {it.label} ID {it.track_id} at [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] conf={it.confidence:.2f}"
                )

        if interactions:
            lines.append("Active person-item interactions:")
            for inter in interactions:
                lines.append(
                    f"  - Person {inter['person_id']} is {inter['distance']:.0f}px from {inter['item_type']} (ID {inter['item_id']})"
                )

        return "\n".join(lines)

    def _error_result(self, error_msg: str) -> dict:
        """Return a safe default result on error."""
        return {
            "theft_detected": False,
            "behavior_type": "error",
            "confidence": 0.0,
            "suspect_description": "",
            "reasoning": f"Analysis failed: {error_msg}",
            "timestamp": time.time(),
        }

    def get_recent_analyses(self, count: int = 10) -> List[Dict]:
        """Get the most recent analyses (for dashboard display)."""
        return self.analysis_history[-count:]
