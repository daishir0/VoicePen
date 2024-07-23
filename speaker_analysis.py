import argparse
import contextlib
import wave
import numpy as np
import os
import subprocess
import tempfile
import time
from pyannote.audio import Audio, Pipeline
from pyannote.core import Segment
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from sklearn.cluster import AgglomerativeClustering
import yaml
import torch

def log(message: str):
    print(message)

def convert_to_wav(input_file: str) -> str:
    log("convert_to_wav: Start")
    start_time = time.time()
    file_name, file_extension = os.path.splitext(input_file)
    wav_file = file_name + '.wav'
    
    if not os.path.exists(wav_file):
        log("convert_to_wav: Converting file")
        if file_extension.lower() in ['.mp3', '.m4a', '.mp4', '.webm']:
            try:
                subprocess.run(['ffmpeg', '-i', input_file, '-acodec', 'pcm_s16le', '-ar', '16000', wav_file], check=True)
            except subprocess.CalledProcessError as e:
                log(f"ffmpegエラー: {e.stderr.decode()}")
                raise
        else:
            log("convert_to_wav: Unsupported file format")
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    log(f"convert_to_wav: End (処理時間: {time.time() - start_time:.2f}秒)")
    return wav_file

def generate_speaker_embeddings(audio_file: str) -> np.ndarray:
    """
    音声ファイルから話者の埋め込みを計算します。
    """
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    audio = Audio()
    embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=device)
    # config.yamlからトークンを読み込む
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=config['huggingface']['use_auth_token'])
    
    with contextlib.closing(wave.open(audio_file, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    
    # 発話ごとにセグメントを作成
    diarization = pipeline(audio_file)
    
    embeddings = []
    for segment in diarization.get_timeline():
        waveform, _ = audio.crop(audio_file, segment)
        embedding = embedding_model(waveform[None])
        embeddings.append(embedding)
    
    log(f"generate_speaker_embeddings: End (処理時間: {time.time() - start_time:.2f}秒)")
    return np.vstack(embeddings), diarization

def clustering_embeddings(speaker_count: int, embeddings: np.ndarray) -> np.ndarray:
    """
    埋め込みデータをクラスタリングして、ラベルを返します。
    """
    start_time = time.time()
    clustering = AgglomerativeClustering(speaker_count).fit(embeddings)
    log(f"clustering_embeddings: End (処理時間: {time.time() - start_time:.2f}秒)")
    return clustering.labels_

def format_time(seconds: float) -> str:
    """
    秒数を HH:MM:SS 形式に変換します。
    """
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def main(audio_file: str, speaker_count: int):
    print(f"入力ファイル {audio_file} を処理中...")
    
    # 出力ファイル名を生成
    output_file = os.path.splitext(audio_file)[0] + ".txt"
    
    # 入力ファイルをWAVに変換
    temp_wav = convert_to_wav(audio_file)
    
    try:
        print("話者の埋め込みを生成中...")
        embeddings, speech_segments = generate_speaker_embeddings(temp_wav)
        
        print("クラスタリングを実行中...")
        labels = clustering_embeddings(speaker_count, embeddings)
        
        print("結果を書き込み中...")
        start_time = time.time()
        with open(output_file, 'w') as f:
            f.write("=== 話者分析結果 ===\n")
            for segment, label in zip(speech_segments.get_timeline(), labels):
                start_time_segment = format_time(segment.start)
                end_time_segment = format_time(segment.end)
                f.write(f"{start_time_segment} - {end_time_segment} : 話者{label + 1}\n")
        log(f"結果の書き込み: End (処理時間: {time.time() - start_time:.2f}秒)")
        
        print(f"分析が完了しました。結果は {output_file} に保存されました。")
    
    finally:
        # 一時ファイルを削除
        os.unlink(temp_wav)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="音声ファイルの話者分析を行います。")
    parser.add_argument("audio_file", help="分析対象の主音声ファイル（WAV, MP3, MP4, M4A形式）")
    parser.add_argument("-s", "--speaker-count", type=int, required=True, help="予想される話者数")
    
    args = parser.parse_args()
    
    main(args.audio_file, args.speaker_count)
