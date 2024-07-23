from flask import Flask, render_template
from flask_socketio import SocketIO
from openai import OpenAI
import os
import tempfile
import yaml
import ssl
from faster_whisper import WhisperModel
import torch
import subprocess
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app)

# Load configuration
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Initialize OpenAI client with API key from config
client = OpenAI(api_key=config['openai']['api_key'])

# Initialize Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel("large-v3", device=device, compute_type="float16" if device == "cuda" else "int8")

def log(message):
    print(f"{datetime.now()} - {message}")

def convert_to_wav(input_file: str) -> str:
    log("convert_to_wav: Start")
    file_name, file_extension = os.path.splitext(input_file)
    wav_file = file_name + '.wav'
    
    if not os.path.exists(wav_file):
        log("convert_to_wav: Converting file")
        if file_extension.lower() in ['.mp3', '.m4a', '.mp4', '.webm']:
            subprocess.run(['ffmpeg', '-i', input_file, '-acodec', 'pcm_s16le', '-ar', '16000', wav_file], check=True)
        else:
            log("convert_to_wav: Unsupported file format")
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    log("convert_to_wav: End")
    return wav_file

def transcribe_audio(audio_file: str) -> str:
    log("transcribe_audio: Start")
    segments, _ = model.transcribe(audio_file, beam_size=5)
    result = " ".join([seg.text for seg in segments])
    log("transcribe_audio: End")
    return result

@app.route('/')
def index():
    log("index: Rendering index.html")
    return render_template('index.html')

@app.route('/large-v3')
def large_v3():
    log("large_v3: Rendering large_v3.html")
    return render_template('large_v3.html')

@socketio.on('audio_data')
def handle_audio_data(audio_data):
    log("handle_audio_data: Start")
    # 一時ファイルとして音声データを保存
    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_audio:
        temp_audio.write(audio_data)
        temp_audio_path = temp_audio.name
        log(f"handle_audio_data: Audio data saved to {temp_audio_path}")

    try:
        # OpenAI APIを使用して音声を文字起こし
        log("handle_audio_data: Transcribing audio using OpenAI API")
        with open(temp_audio_path, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        # 文字起こし結果をクライアントに送信
        log("handle_audio_data: Emitting transcription result")
        socketio.emit('transcription', {'text': transcript.text})
    finally:
        # 一時ファイルを削除
        log(f"handle_audio_data: Deleting temporary file {temp_audio_path}")
        os.unlink(temp_audio_path)
    log("handle_audio_data: End")

@socketio.on('audio_data_local')
def handle_audio_data_local(audio_data):
    log("handle_audio_data_local: Start")
    # 一時ファイルとして音声データを保存
    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_audio:
        temp_audio.write(audio_data)
        temp_audio_path = temp_audio.name
        log(f"handle_audio_data_local: Audio data saved to {temp_audio_path}")

    try:
        # WAVに変換
        log("handle_audio_data_local: Converting to WAV")
        wav_file = convert_to_wav(temp_audio_path)
        log(f"handle_audio_data_local: Conversion successful: {wav_file}")
    except subprocess.CalledProcessError as e:
        log(f"handle_audio_data_local: Conversion failed: {e}")
        wav_file = None  # Ensure wav_file is defined
    finally:
        log(f"handle_audio_data_local: Deleting temporary file {temp_audio_path}")
        os.unlink(temp_audio_path)
    
    if wav_file and os.path.exists(wav_file):
        # Faster Whisperモデルを使用してローカルで音声を文字起こし
        log("handle_audio_data_local: Transcribing audio using Faster Whisper model")
        transcript = transcribe_audio(wav_file)
        
        # 文字起こし結果をクライアントに送信
        log("handle_audio_data_local: Emitting transcription result")
        socketio.emit('transcription', {'text': transcript})
        if os.path.exists(wav_file):
            log(f"handle_audio_data_local: Deleting WAV file {wav_file}")
            os.unlink(wav_file)
    log("handle_audio_data_local: End")

if __name__ == '__main__':
    log("Starting server")
    ssl_context = ssl.create_default_context(purpose=ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(
        certfile='fullchain.pem',
        keyfile='privkey.pem'
    )
    
    socketio.run(app, host='0.0.0.0', port=5000, ssl_context=ssl_context, debug=True)
    log("Server started")
