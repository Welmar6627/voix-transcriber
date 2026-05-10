import os
import json
import subprocess
import tempfile
import math
import requests
import uuid
import shutil
import sys
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

UPLOAD_FOLDER = Path(__file__).parent / "uploads"
OUTPUT_FOLDER = Path(__file__).parent / "outputs"
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

GROQ_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "whisper-large-v3"

# ── Locate ffmpeg on Windows or Unix ─────────────────────────
def find_ffmpeg():
    # Hardcoded Windows path (confirmed location)
    hardcoded = r"C:\ffmpeg\bin\ffmpeg.exe"
    hardcoded_probe = r"C:\ffmpeg\bin\ffprobe.exe"
    if Path(hardcoded).exists():
        return hardcoded, hardcoded_probe

    # Fallback: search PATH
    found = shutil.which("ffmpeg")
    if found:
        return found, shutil.which("ffprobe") or found

    return None, None

FFMPEG, FFPROBE = find_ffmpeg()

if not FFMPEG:
    print("\n⚠️  ffmpeg not found. Please set FFMPEG_PATH in this script or add ffmpeg to PATH.\n")
    # Fallback — will raise a clear error at runtime
    FFMPEG = "ffmpeg"
    FFPROBE = "ffprobe"
else:
    print(f"✓ ffmpeg found: {FFMPEG}")

# Groq limit: 25MB per file. We chunk at 20MB to be safe.
# At ~128kbps MP3, 20MB ≈ ~20 minutes of audio
CHUNK_SIZE_MB = 20
CHUNK_DURATION_SEC = 18 * 60  # 18 minutes per chunk (safe margin)

ALLOWED_EXTENSIONS = {'.mp3', '.m4a', '.wav', '.ogg', '.flac', '.webm'}


def get_audio_duration(filepath: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        [FFPROBE, "-v", "quiet", "-print_format", "json", "-show_format", str(filepath)],
        capture_output=True, text=True
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def get_file_size_mb(filepath: str) -> float:
    return os.path.getsize(filepath) / (1024 * 1024)


def convert_to_mp3(filepath: str) -> str:
    """Convert audio to MP3 for Groq compatibility."""
    out = str(UPLOAD_FOLDER / f"converted_{uuid.uuid4().hex}.mp3")
    subprocess.run(
        [FFMPEG, "-y", "-i", str(filepath), "-ar", "16000",
         "-ac", "1", "-b:a", "64k", out],
        capture_output=True
    )
    return out


def split_audio(filepath: str) -> list[str]:
    """Split audio into chunks using ffmpeg if needed. Returns list of chunk paths."""
    size_mb = get_file_size_mb(filepath)
    
    # First convert to compressed mono MP3 to reduce size
    converted = convert_to_mp3(filepath)
    converted_size = get_file_size_mb(converted)

    if converted_size <= CHUNK_SIZE_MB:
        return [converted]

    # Need to split
    duration = get_audio_duration(converted)
    num_chunks = math.ceil(duration / CHUNK_DURATION_SEC)
    chunks = []

    for i in range(num_chunks):
        start = i * CHUNK_DURATION_SEC
        chunk_path = str(UPLOAD_FOLDER / f"chunk_{uuid.uuid4().hex}_{i}.mp3")
        subprocess.run([
            FFMPEG, "-y", "-ss", str(start),
            "-t", str(CHUNK_DURATION_SEC),
            "-i", converted,
            "-ar", "16000", "-ac", "1", "-b:a", "64k",
            chunk_path
        ], capture_output=True)
        if Path(chunk_path).exists() and Path(chunk_path).stat().st_size > 1024:
            chunks.append(chunk_path)

    # Clean up converted file if it was chunked
    try:
        os.remove(converted)
    except Exception:
        pass

    return chunks


def transcribe_chunk(chunk_path: str, api_key: str,
                     chunk_index: int = 0, total_chunks: int = 1) -> dict:
    """Send one chunk to Groq Whisper API."""
    headers = {"Authorization": f"Bearer {api_key}"}

    with open(chunk_path, "rb") as f:
        files = {
            "file": (Path(chunk_path).name, f, "audio/mpeg"),
        }
        data = {
            "model": GROQ_MODEL,
            "response_format": "verbose_json",  # gives us language detection
            "temperature": 0,
        }
        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=300
        )

    response.raise_for_status()
    result = response.json()

    text = result.get("text", "").strip()
    language = result.get("language", "unknown")

    return {"text": text, "language": language}


def create_pdf(transcription: str, filename: str, language: str) -> str:
    """Create a styled PDF from transcription text."""
    output_path = str(OUTPUT_FOLDER / f"{uuid.uuid4().hex}.pdf")

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        rightMargin=2.5*cm, leftMargin=2.5*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'Title2', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=20,
        textColor=colors.HexColor('#0f0e17'),
        spaceAfter=6,
    )
    meta_style = ParagraphStyle(
        'Meta', parent=styles['Normal'],
        fontName='Helvetica', fontSize=10,
        textColor=colors.HexColor('#8a8478'),
        spaceAfter=20,
    )
    body_style = ParagraphStyle(
        'Body2', parent=styles['Normal'],
        fontName='Helvetica', fontSize=11,
        leading=19, textColor=colors.HexColor('#0f0e17'),
        spaceAfter=10,
    )

    story = []
    safe_name = filename.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    story.append(Paragraph(f"Transcription: {safe_name}", title_style))
    story.append(Paragraph(f"Detected Language: {language.title()}", meta_style))
    story.append(Spacer(1, 0.4*cm))

    for para in transcription.strip().split('\n\n'):
        if para.strip():
            safe = para.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br/>')
            story.append(Paragraph(safe, body_style))
            story.append(Spacer(1, 0.15*cm))

    doc.build(story)
    return output_path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    api_key = request.headers.get('X-API-Key', '').strip()
    if not api_key:
        return jsonify({"error": "Missing Groq API key"}), 401

    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files['audio']
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported format. Use MP3, M4A, WAV, OGG, FLAC."}), 400

    # Save upload
    upload_path = str(UPLOAD_FOLDER / f"{uuid.uuid4().hex}{ext}")
    file.save(upload_path)

    chunk_paths = []
    converted_paths = []

    try:
        chunk_paths = split_audio(upload_path)
        total_chunks = len(chunk_paths)

        transcriptions = []
        detected_language = "unknown"

        for i, chunk_path in enumerate(chunk_paths):
            result = transcribe_chunk(chunk_path, api_key, i, total_chunks)
            transcriptions.append(result["text"])
            if i == 0:
                detected_language = result["language"]

        full_text = "\n\n".join(transcriptions)

        return jsonify({
            "transcription": full_text,
            "language": detected_language.title(),
            "chunks": total_chunks,
            "filename": file.filename
        })

    except requests.HTTPError as e:
        try:
            msg = e.response.json().get("error", {}).get("message", str(e))
        except Exception:
            msg = str(e)
        return jsonify({"error": f"Groq API error: {msg}"}), 502

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        for p in [upload_path] + chunk_paths:
            try:
                os.remove(p)
            except Exception:
                pass


@app.route('/api/download/pdf', methods=['POST'])
def download_pdf():
    data = request.get_json()
    transcription = data.get("transcription", "")
    filename = data.get("filename", "audio")
    language = data.get("language", "Unknown")
    if not transcription:
        return jsonify({"error": "No transcription"}), 400
    pdf_path = create_pdf(transcription, filename, language)
    return send_file(pdf_path, as_attachment=True,
                     download_name=Path(filename).stem + "_transcription.pdf",
                     mimetype="application/pdf")


@app.route('/api/download/txt', methods=['POST'])
def download_txt():
    data = request.get_json()
    transcription = data.get("transcription", "")
    filename = data.get("filename", "audio")
    if not transcription:
        return jsonify({"error": "No transcription"}), 400
    txt_path = str(OUTPUT_FOLDER / f"{uuid.uuid4().hex}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcription)
    return send_file(txt_path, as_attachment=True,
                     download_name=Path(filename).stem + "_transcription.txt",
                     mimetype="text/plain")


if __name__ == '__main__':
    print("\n🎙️  VOIX — Groq Whisper Transcription Server")
    print("=" * 44)
    print("  Open: http://localhost:5000")
    print("  Free API key: https://console.groq.com")
    print("=" * 44 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
