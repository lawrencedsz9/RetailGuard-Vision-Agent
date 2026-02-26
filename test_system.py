"""
Quick test script — verifies each component works independently.

Run: python test_system.py

Tests:
1. YOLO model loads and can process a dummy frame
2. AlertManager can create DB and log a test incident
3. (Optional) Gemini API connectivity if key is set

This does NOT require a webcam.
"""

import sys
import os
import numpy as np
import time

# Ensure imports work from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_yolo():
    """Test that YOLO11 loads and runs on a dummy frame."""
    print("\n[TEST 1] YOLO11 Detection...")
    try:
        from processors.theft_processor import RetailTheftProcessor

        processor = RetailTheftProcessor(model_path="yolo11n.pt", conf_threshold=0.45)

        # Create a dummy frame (black image)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = processor.process(dummy_frame)

        assert "detections" in results
        assert "persons" in results
        assert "items" in results
        assert "interactions" in results
        assert "frame_shape" in results

        print(f"  YOLO loaded: {processor.model.model_name}")
        print(f"  Dummy frame processed: {len(results['detections'])} detections")
        print("  [PASS] YOLO works!\n")
        return True
    except Exception as e:
        print(f"  [FAIL] YOLO error: {e}\n")
        return False


def test_alert_manager():
    """Test SQLite logging and evidence saving."""
    print("[TEST 2] Alert Manager...")
    try:
        from alerts.alert_manager import AlertManager

        # Use a test DB file
        alerts = AlertManager(db_path="test_incidents.db", clips_dir="test_clips")

        # Verify DB was created
        assert os.path.exists("test_incidents.db"), "DB not created"

        # Log a fake incident
        fake_analysis = {
            "theft_detected": True,
            "behavior_type": "concealment",
            "confidence": 0.85,
            "suspect_description": "Test person hiding item",
            "reasoning": "Test incident for verification",
        }
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_yolo = {"persons": [], "items": [], "interactions": [], "frame_shape": (480, 640, 3)}

        evidence = alerts._save_evidence(fake_frame, fake_analysis, fake_yolo)
        incident_id = alerts._log_incident(fake_analysis, evidence, fake_yolo)

        print(f"  Test incident logged: ID #{incident_id}")

        # Verify retrieval
        results = alerts.search_incidents()
        assert len(results) > 0, "No incidents found"
        print(f"  Retrieved {len(results)} incident(s)")

        # Verify stats
        stats = alerts.get_stats()
        print(f"  Stats: {stats}")

        # Cleanup test files
        os.remove("test_incidents.db")
        import shutil
        if os.path.exists("test_clips"):
            shutil.rmtree("test_clips")

        print("  [PASS] Alert Manager works!\n")
        return True
    except Exception as e:
        print(f"  [FAIL] Alert Manager error: {e}\n")
        return False


def test_gemini():
    """Test Gemini API connectivity (requires API key in .env)."""
    print("[TEST 3] Gemini API...")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY", "")
        
        if not api_key or api_key == "your_gemini_api_key_here":
            print("  [SKIP] No GOOGLE_API_KEY in .env — set it to test Gemini")
            return True

        from google import genai
        client = genai.Client(api_key=api_key)

        # Simple test call
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Say 'RetailGuard AI is ready' in exactly those words.",
        )
        print(f"  Gemini response: {response.text.strip()}")
        print("  [PASS] Gemini API works!\n")
        return True
    except Exception as e:
        print(f"  [FAIL] Gemini error: {e}\n")
        return False


def test_webcam():
    """Test webcam access (optional, visual test)."""
    print("[TEST 4] Webcam Access...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  [SKIP] No webcam detected")
            return True
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            print(f"  Webcam frame captured: {frame.shape}")
            print("  [PASS] Webcam works!\n")
            return True
        else:
            print("  [SKIP] Webcam opened but couldn't capture frame")
            return True
    except Exception as e:
        print(f"  [FAIL] Webcam error: {e}\n")
        return False


def test_sdk_processor():
    """Test that the Vision Agents SDK processor loads correctly."""
    print("[TEST 5] Vision Agents SDK Processor...")
    try:
        from processors.retail_processor import RetailTheftProcessor, TheftInteractionEvent
        from vision_agents.core.events.base import PluginBaseEvent

        # Check inheritance
        assert issubclass(TheftInteractionEvent, PluginBaseEvent), "Event must extend PluginBaseEvent"
        print("  TheftInteractionEvent extends PluginBaseEvent ✓")

        # Check event type field
        evt = TheftInteractionEvent(plugin_name="retail_theft", person_id=1, item_type="bag", item_id=2, distance=50.0, timestamp="now")
        assert evt.type == "retail.theft_interaction", f"Event type mismatch: {evt.type}"
        print(f"  Event type: {evt.type} ✓")

        # Check processor has required attributes
        from vision_agents.core.processors.base_processor import VideoProcessorPublisher
        from vision_agents.core.warmup import Warmable
        assert issubclass(RetailTheftProcessor, VideoProcessorPublisher), "Processor must extend VideoProcessorPublisher"
        assert issubclass(RetailTheftProcessor, Warmable), "Processor must extend Warmable"
        print("  RetailTheftProcessor extends VideoProcessorPublisher + Warmable ✓")

        print("  [PASS] SDK Processor is properly configured!\n")
        return True
    except Exception as e:
        print(f"  [FAIL] SDK Processor error: {e}\n")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("  RetailGuard AI — System Test")
    print("=" * 60)

    results = {
        "YOLO11": test_yolo(),
        "Alert Manager": test_alert_manager(),
        "Gemini API": test_gemini(),
        "Webcam": test_webcam(),
        "SDK Processor": test_sdk_processor(),
    }

    print("=" * 60)
    print("  RESULTS:")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"    {name}: {status}")
    print("=" * 60)

    all_passed = all(results.values())
    if all_passed:
        print("\n  All tests passed! You're ready to go.")
        print("  Next steps:")
        print("    1. Set your API keys in .env")
        print("    2. Run: python main.py --local")
        print("    3. In another terminal: streamlit run dashboard/app.py")
    else:
        print("\n  Some tests failed. Check the errors above.")
    
    sys.exit(0 if all_passed else 1)
