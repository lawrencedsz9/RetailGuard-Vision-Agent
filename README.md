# RetailGuard AI

> A multimodal AI surveillance system built with the [Vision Agents SDK](https://github.com/GetStream/Vision-Agents) that detects and prevents retail theft in real time using computer vision, behavioral analysis, and voice alerts.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![YOLO](https://img.shields.io/badge/YOLO-v11-green)
![Gemini](https://img.shields.io/badge/Gemini-2.0_Flash-orange)
![Vision Agents](https://img.shields.io/badge/Vision_Agents-SDK-purple)

---

## Overview

RetailGuard AI is a 3-stage intelligent pipeline built on the **Vision Agents SDK** by Stream. It monitors retail environments through camera feeds via Stream's WebRTC Edge Network , detects suspicious behavior in real time, and responds with voice alerts — all while logging incidents to a searchable dashboard.



## Pipeline

| Stage | Component | Role |
|-------|-----------|------|
| 1 | **YOLO11n** | Detects people, bags, bottles, backpacks. Tracks with unique IDs across frames |
| 2 | **Gemini Flash** | Analyzes annotated frames for theft behavior — concealment, unpaid exit, consumption |
| 3 | **ElevenLabs TTS** | Speaks real-time voice alerts. Logs incidents with screenshots to SQLite |

## Features

- **Real-time object detection** — YOLO11 runs on every frame with bounding boxes and tracking IDs
- **Behavioral analysis** — Gemini reads the scene and identifies 3 theft types:
  - *Concealment* — hiding merchandise in bags, pockets, or clothing
  - *Unpaid Exit* — heading toward exits with unpaid items
  - *Consumption* — opening or using products before paying
- **Voice deterrence** — ElevenLabs speaks alerts on detection (e.g., "Attention: suspicious activity detected in aisle 3")
- **Incident dashboard** — Streamlit UI with stats, screenshots, search, and status management
- **Two operating modes:**
  - **SDK Mode** — Vision Agents SDK with Stream WebRTC for <30ms latency
  - **Local Mode** — Direct webcam feed for testing without API keys

## Setup

### 1. Clone & create environment

```bash
git clone https://github.com/your-username/retailguard-ai.git
cd retailguard-ai
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

Create a `.env` file in the project root:

```env
STREAM_API_KEY=your_stream_api_key
STREAM_API_SECRET=your_stream_api_secret
GOOGLE_API_KEY=your_google_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
```

## Usage

### SDK Mode (full pipeline)
```bash
python agent.py run
```

### Local Mode (webcam testing)
```bash
python main.py
```

### Dashboard (separate terminal)
```bash
streamlit run dashboard/app.py
```

### Run Tests
```bash
python test_system.py
```

## Project Structure

```
retailguard-ai/
├── agent.py                    # SDK entry point — Agent + Runner + System
├── main.py                     # Local webcam pipeline
├── processors/
│   ├── retail_processor.py     # YOLO processor for Vision Agents SDK
│   └── theft_processor.py      # YOLO processor for local mode
├── agents/
│   └── theft_analyzer.py       # Gemini behavioral analysis
├── alerts/
│   └── alert_manager.py        # ElevenLabs TTS + SQLite logging + screenshots
├── dashboard/
│   └── app.py                  # Streamlit incident dashboard
├── test_system.py              # System validation tests
├── requirements.txt            # Python dependencies
├── .env                        
└── .gitignore
```



Built for the **Vision Agents Hackathon 2026** by Stream
