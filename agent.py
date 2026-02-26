"""
RetailGuard AI — Vision Agents SDK Integration

This is the main entry point using Stream's Vision Agents SDK.
It creates a real-time video AI agent that:
  1. Receives video via Stream's WebRTC edge network (<30ms latency)
  2. Runs YOLO11 object detection on every frame (people + items)
  3. Uses Gemini Realtime to analyze behavior and detect theft
  4. Speaks alerts via ElevenLabs TTS
  5. Logs incidents to SQLite for the dashboard

Run with:
  python agent.py run                          # Start the agent server
  python agent.py run --call-type default --call-id retail-guard  # Join specific call

Architecture (Vision Agents SDK):
  Stream Edge (WebRTC) -> YOLO Processor -> Gemini Realtime LLM -> ElevenLabs TTS
                                                                -> SQLite + Dashboard
"""

import asyncio
import logging
from typing import Any, Dict
from datetime import datetime

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, elevenlabs, gemini, getstream, ultralytics

from processors.retail_processor import RetailTheftProcessor, TheftInteractionEvent
from alerts.alert_manager import AlertManager

load_dotenv()
logger = logging.getLogger(__name__)

# ── Agent Instructions ─────────────────────────────────────────────
INSTRUCTIONS = """You are RetailGuard AI, a real-time retail security surveillance assistant.

Your job is to monitor a store camera feed and detect theft behaviors:

1. CONCEALMENT: A person hiding merchandise in their pockets, bag, or clothing
2. UNPAID EXIT: A person moving toward the store exit while carrying unpaid items
3. CONSUMPTION: A person opening, eating, drinking, or using a product before paying

RESPONSE RULES:
- When you see suspicious behavior, announce it clearly and concisely
- State what you see, the behavior type, and recommend an action
- Example: "Alert - I see a person near the electronics section placing a small item into their jacket pocket. This appears to be concealment. Staff should check aisle 3."
- When nothing suspicious is happening, stay quiet - do NOT narrate normal shopping
- If asked about recent activity, describe what you've observed
- Keep responses under 3 sentences
- Be professional and factual, never accusatory without evidence
- Normal behaviors (browsing, examining items, carrying a basket) are NOT theft

The YOLO processor will highlight people and items with bounding boxes.
Focus on the spatial relationship between people and merchandise."""


# Global alert manager (shared between agent lifecycle)
alert_manager = AlertManager(db_path="incidents.db", clips_dir="clips")


async def create_agent(**kwargs) -> Agent:
    """Create the RetailGuard agent with YOLO processor + Gemini Realtime."""

    # Gemini Realtime: sends video frames directly to Gemini at 2 FPS
    # This is the "brain" that understands behavior from video
    llm = gemini.Realtime(fps=2)

    # Custom YOLO processor for retail theft detection
    # Detects people, bags, bottles, etc. and tracks interactions
    theft_processor = RetailTheftProcessor(
        model_path="yolo11n.pt",
        conf_threshold=0.45,
        interaction_distance=150,
    )

    agent = Agent(
        edge=getstream.Edge(),  # Stream's ultra-low-latency WebRTC edge network
        agent_user=User(
            name="RetailGuard AI",
            id="retailguard-agent",
        ),
        instructions=INSTRUCTIONS,
        llm=llm,
        processors=[theft_processor],  # YOLO runs on every frame
        tts=elevenlabs.TTS(model_id="eleven_flash_v2_5"),
        stt=deepgram.STT(eager_turn_detection=True),
    )

    # Merge processor events so we can subscribe to them
    agent.events.merge(theft_processor.events)

    # ── Register LLM Functions (tool calling) ──────────────────────
    @llm.register_function(
        description="Get current incident statistics including total incidents, today's count, and breakdown by behavior type."
    )
    async def get_incident_stats() -> Dict[str, Any]:
        stats = alert_manager.get_stats()
        return stats

    @llm.register_function(
        description="Search past incidents by behavior type or description. Use when asked about past theft events."
    )
    async def search_incidents(query: str = "", date: str = "") -> Dict[str, Any]:
        results = alert_manager.search_incidents(query=query or None, date_filter=date or None)
        return {
            "incidents": results[:10],
            "total_found": len(results),
        }

    @llm.register_function(
        description="Log a theft incident to the database when you observe suspicious behavior. Call this when you detect theft with high confidence."
    )
    async def log_theft_incident(
        behavior_type: str,
        confidence: float,
        description: str,
    ) -> Dict[str, Any]:
        analysis = {
            "theft_detected": True,
            "behavior_type": behavior_type,
            "confidence": confidence,
            "suspect_description": description,
            "reasoning": f"Detected by RetailGuard AI Gemini analysis at {datetime.now().isoformat()}",
        }
        # We don't have the raw frame in tool-call context, so log without screenshot
        evidence = {"screenshot": "", "metadata": ""}
        yolo_data = {"persons": 0, "items": 0, "interactions": []}
        incident_id = alert_manager._log_incident(analysis, evidence, yolo_data)
        logger.info(f"Incident #{incident_id} logged: {behavior_type} ({confidence:.0%})")
        return {
            "incident_id": incident_id,
            "status": "logged",
            "behavior_type": behavior_type,
        }

    # ── Subscribe to Processor Events ──────────────────────────────
    @agent.events.subscribe
    async def on_theft_interaction(event: TheftInteractionEvent):
        """Called when YOLO detects a person interacting with an item."""
        logger.info(
            f"⚠️ Interaction: Person #{event.person_id} near {event.item_type} "
            f"(distance: {event.distance:.0f}px)"
        )
        # The Gemini Realtime LLM will see the annotated frame with
        # bounding boxes and interaction lines, and decide if it's theft

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join a Stream call and start monitoring."""
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        # Initial greeting
        await agent.llm.simple_response(
            text="RetailGuard AI is now monitoring this camera feed. "
            "I'll alert you if I detect any suspicious behavior. "
            "You can ask me about recent activity or incident statistics."
        )
        # Run until the call ends
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
