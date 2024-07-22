# VoicePen

## Overview
VoicePen is a web application that provides real-time audio transcription using both OpenAI's Whisper API and a local Faster Whisper model. It offers a user-friendly interface for recording audio and obtaining transcriptions, with options for cloud-based and on-premise processing.

## Installation
To install VoicePen, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/daishir0/VoicePen.git
   ```
2. Change to the project directory:
   ```
   cd VoicePen
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Set up your OpenAI API key in a `config.yaml` file:
   ```yaml
   openai:
     api_key: 'your_api_key_here'
   ```
5. Ensure you have FFmpeg installed on your system for audio conversion.

## Usage
1. Start the server:
   ```
   python app.py
   ```
2. Open a web browser and navigate to `https://localhost:5000`.
3. For OpenAI API transcription, use the default page.
4. For local transcription using Faster Whisper, navigate to `/large-v3`.
5. Click the "Start Recording" button to begin recording audio.
6. Click "Stop Recording" when finished.
7. The transcription will appear on the page.

## Notes
- The application uses SSL for secure connections. Make sure to have `fullchain.pem` and `privkey.pem` in the project directory.
- The local transcription uses the "large-v3" Whisper model, which requires significant computational resources.
- GPU acceleration is used if available, falling back to CPU if not.
- Supported input audio formats include MP3, M4A, MP4, and WebM.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---

# VoicePen

## 概要
VoicePenは、OpenAIのWhisper APIとローカルのFaster Whisperモデルを使用してリアルタイムの音声文字起こしを提供するWebアプリケーションです。音声を録音して文字起こしを取得するための使いやすいインターフェースを提供し、クラウドベースとオンプレミスの処理オプションを備えています。

## インストール方法
VoicePenをインストールするには、以下の手順に従ってください：

1. リポジトリをクローンします：
   ```
   git clone https://github.com/daishir0/VoicePen.git
   ```
2. プロジェクトディレクトリに移動します：
   ```
   cd VoicePen
   ```
3. 必要な依存関係をインストールします：
   ```
   pip install -r requirements.txt
   ```
4. `config.yaml`ファイルにOpenAI APIキーを設定します：
   ```yaml
   openai:
     api_key: 'your_api_key_here'
   ```
5. 音声変換用にFFmpegがシステムにインストールされていることを確認してください。

## 使い方
1. サーバーを起動します：
   ```
   python app.py
   ```
2. Webブラウザを開き、`https://localhost:5000`にアクセスします。
3. OpenAI APIによる文字起こしにはデフォルトページを使用します。
4. Faster Whisperを使用したローカルでの文字起こしには、`/large-v3`に移動します。
5. "録音開始"ボタンをクリックして音声の録音を開始します。
6. 録音が終わったら"録音停止"をクリックします。
7. 文字起こし結果がページに表示されます。

## 注意点
- アプリケーションはセキュアな接続のためにSSLを使用します。プロジェクトディレクトリに`fullchain.pem`と`privkey.pem`があることを確認してください。
- ローカルの文字起こしには"large-v3" Whisperモデルを使用しており、かなりの計算リソースを必要とします。
- GPUが利用可能な場合はGPUアクセラレーションを使用し、利用できない場合はCPUにフォールバックします。
- サポートされている入力音声フォーマットには、MP3、M4A、MP4、WebMが含まれます。

## ライセンス
このプロジェクトはMITライセンスの下でライセンスされています。詳細はLICENSEファイルを参照してください。
