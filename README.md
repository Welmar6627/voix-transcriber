# VOIX — Free Audio Transcription (Groq Whisper)

Transcribes MP3, M4A, WAV, OGG, and FLAC files using Groq's free Whisper API.
Handles long files by auto-splitting. Auto-detects language. Export as TXT or PDF.

**100% free — no credit card required.**

---

## Setup

### 1. Get a free Groq API key
- Go to https://console.groq.com
- Sign up (free, no credit card)
- Click "API Keys" → "Create API Key"
- Copy it — starts with `gsk_...`

### 2. Install ffmpeg

**Windows:** Download from https://ffmpeg.org → add to PATH
**macOS:** `brew install ffmpeg`
**Ubuntu:** `sudo apt install ffmpeg`

### 3. Install Python packages

```bash
pip install -r requirements.txt
```

### 4. Run

```bash
python app.py
```

Open **http://localhost:5000**, paste your `gsk_...` key, drop a file, click Transcribe.

---

## Features

- **Free** — Groq's free tier supports Whisper-large-v3
- **Auto language detection** — works with 90+ languages
- **Long file support** — files are converted to compressed mono MP3 and split into 18-minute chunks automatically
- **Multiple formats** — MP3, M4A, WAV, OGG, FLAC
- **Export** — Download as TXT or formatted PDF
- **Private** — files deleted from disk immediately after transcription

## Groq Free Tier Limits

- 7,200 seconds of audio per day (~2 hours)
- 20 requests per minute
- No credit card needed

For more: https://console.groq.com/docs/rate-limits
