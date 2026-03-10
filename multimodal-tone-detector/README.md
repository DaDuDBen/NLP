# Multimodal Tone Detector

Real-time and file-based multimodal tone detection system that fuses:
- **Speech + NLP** (Whisper transcription + HuggingFace irony/emotion models)
- **Facial expressions** (OpenCV frame capture + DeepFace emotion inference)

Final labels include:
- `Sarcastic`
- `Sincere/Positive`
- `Joking/Playful`
- `Neutral/Sincere`

## Project Structure

```text
multimodal-tone-detector/
├── main.py
├── pipeline/
│   ├── audio.py
│   ├── video.py
│   ├── fusion.py
│   └── nlp.py
├── dashboard/
│   └── app.py
├── utils/
│   └── sync.py
├── requirements.txt
└── README.md
```

## Setup

1. Ensure Python 3.10+ is installed.
2. Install system dependencies:
   - `ffmpeg`
   - webcam/microphone drivers
   - (optional) PortAudio headers for PyAudio build (`portaudio19-dev` on Debian/Ubuntu)
3. Install Python packages:

```bash
pip install -r requirements.txt
```

## Run (CLI)

### Live webcam + microphone mode

```bash
python main.py --mode live --duration 60
```

### Video file mode

```bash
python main.py --mode file --video /path/to/video.mp4
```

Supported file formats include `.mp4`, `.mov`, `.avi`.

## Run (Dashboard)

```bash
streamlit run dashboard/app.py
```

Dashboard features:
- Mode selector (Live Webcam vs Upload Video File)
- Live webcam preview with face/emotion overlay
- Rolling transcript updates
- Tone label + confidence bar with color coding
- Scrollable timeline log (`timestamp | transcript | facial emotion | final tone | confidence`)
- Timeline replay table for uploaded file mode

## Fusion Logic Summary

`pipeline/fusion.py` applies rule-based logic:
- Positive text + negative face -> `Sarcastic`
- Irony + neutral/smirk-like face -> `Sarcastic`
- Joy text + happy face -> `Sincere/Positive`
- Neutral text + exaggerated face (surprise/happy) -> `Joking/Playful`
- Neutral text + neutral face -> `Neutral/Sincere`

All outputs include confidence scores constrained to `[0.0, 1.0]`.

## Notes

- Models run locally (no external API calls required).
- Audio and video capture are threaded separately and synchronized by timestamps (±1 second matching).
- Graceful fallback behavior is included for silent audio and no-face frames.
