# 🎙️ ボイスレコーダー & 声紋分析システム

10秒制限のボイスレコーダーで録音した音声の声紋分析を行うWebアプリケーションです。リアルタイム波形表示、高度な音声特徴量解析、直感的なUIを提供します。

## 📋 目次

- [主要機能](#主要機能)
- [クイックスタート](#クイックスタート)
- [詳細なセットアップ手順](#詳細なセットアップ手順)
- [実行方法](#実行方法)
- [使用方法](#使用方法)
- [技術仕様](#技術仕様)
- [ファイル構成](#ファイル構成)
- [エラーハンドリング](#エラーハンドリング)
- [トラブルシューティング](#トラブルシューティング)
- [パフォーマンス最適化](#パフォーマンス最適化)
- [開発者向け情報](#開発者向け情報)

## 🎯 主要機能

### 📹 音声録音機能
- **ブラウザベース録音**: MediaRecorder APIを使用した最大10秒間の高品質音声録音
- **リアルタイム波形表示**: 録音中の音声波形をリアルタイムで可視化
- **自動録音停止**: 10秒経過時の自動停止機能
- **録音状態表示**: 視覚的な録音状態インジケーター

### 🔊 音声再生・管理
- **インスタント再生**: 録音直後の即座な音声再生
- **音声ファイルダウンロード**: WAVファイルとしてのローカル保存
- **音声品質確認**: 再生前の音声品質チェック

### 🔬 高度な声紋分析
- **メルスペクトログラム**: 人間の聴覚に近い周波数解析
- **STFTスペクトログラム**: 短時間フーリエ変換による詳細な時間-周波数解析
- **基本周波数 (F0) 推定**: 音の高さ変化の詳細な追跡
- **音声波形表示**: 時間領域での音声信号表示
- **MFCC解析**: メル周波数ケプストラム係数による特徴量抽出
- **スペクトル重心**: 周波数分布の重心計算
- **ゼロクロッシングレート**: 音声の動的特性解析
- **スペクトラルロールオフ**: 高周波成分の分析

### 🎨 ユーザーインターフェース
- **レスポンシブデザイン**: モバイル・デスクトップ対応
- **ローディング表示**: 分析処理中の視覚的フィードバック
- **エラー表示**: 詳細なエラーメッセージとガイダンス
- **秘密の管理画面**: &マークをトリプルクリックでアクセス可能な高度な設定画面

## ⚡ クイックスタート

### 📋 前提条件
- **Python**: 3.8以上（3.9以上推奨）
- **オーディオデバイス**: マイク付きデバイス
- **ブラウザ**: Chrome 88+, Firefox 85+, Safari 14+, Edge 88+
- **システムメモリ**: 2GB以上の空きメモリ
- **ストレージ**: 500MB以上の空き容量

### 🚀 30秒でセットアップ

```bash
# 1. 依存関係のインストール
pip3 install -r requirements.txt

# 2. アプリケーション起動
python3 app.py

# 3. ブラウザでアクセス
# http://localhost:8080 を開く
```

## 🔧 詳細なセットアップ手順

### Step 1: Python環境の確認と準備

```bash
# Python バージョン確認（3.8以上が必要）
python3 --version

# pip の確認とアップグレード
pip3 --version
pip3 install --upgrade pip

# システム情報確認（オプション）
python3 -c "import sys; print(f'Python: {sys.version}'); import platform; print(f'OS: {platform.system()} {platform.release()}')"
```

#### ⚠️ Python バージョンチェック結果
- ✅ **Python 3.8-3.11**: 完全サポート
- ⚠️ **Python 3.12+**: 一部ライブラリで互換性問題の可能性
- ❌ **Python 3.7以下**: サポート対象外

### Step 2: プロジェクトの取得と配置

```bash
# プロジェクトディレクトリに移動
cd voice_analysis

# ファイル構成確認
ls -la
# 以下のファイルが存在することを確認:
# app.py, app_production.py, voice_recorder.html, requirements.txt
```

### Step 3: 仮想環境の作成（強く推奨）

```bash
# 仮想環境作成
python3 -m venv venv

# 仮想環境のアクティベート
# macOS/Linux:
source venv/bin/activate

# Windows (PowerShell):
venv\Scripts\Activate.ps1

# Windows (Command Prompt):
venv\Scripts\activate.bat

# 仮想環境が有効か確認
which python  # macOS/Linux
where python  # Windows
```

### Step 4: 依存関係の詳細インストール

```bash
# requirements.txtからの一括インストール（推奨）
pip install -r requirements.txt

# または個別インストール（依存関係の詳細理解用）
pip install Flask==2.3.3        # Webフレームワーク
pip install librosa==0.10.1      # 音声解析ライブラリ
pip install matplotlib==3.7.2    # グラフ描画
pip install numpy==1.24.3        # 数値計算
pip install scipy==1.11.2        # 科学計算

# インストール確認
pip list | grep -E "(Flask|librosa|matplotlib|numpy|scipy)"
```

#### 📦 依存関係の詳細説明
- **Flask**: HTTPサーバーとAPI提供
- **librosa**: 音声信号処理と特徴量抽出
- **matplotlib**: スペクトログラム画像生成
- **numpy**: 数値計算とデータ処理
- **scipy**: 信号処理と科学計算

### Step 5: 環境テストと動作確認

```bash
# Python ライブラリのインポートテスト
python3 -c "
import flask; print(f'✅ Flask: {flask.__version__}')
import librosa; print(f'✅ librosa: {librosa.__version__}')
import matplotlib; print(f'✅ matplotlib: {matplotlib.__version__}')
import numpy; print(f'✅ numpy: {numpy.__version__}')
import scipy; print(f'✅ scipy: {scipy.__version__}')
print('🎉 All dependencies loaded successfully!')
"
```

## 🔄 実行方法

### 🌟 開発環境での実行

```bash
# 開発用サーバー起動（詳細デバッグ情報付き）
python3 app.py

# 期待される出力:
# Voice Analysis Server Starting...
# 🎙️ ローカルアクセス: http://localhost:8080
# 🌐 ネットワークアクセス: http://192.168.x.x:8080
# 📊 Debug mode: ON
# 🔄 Auto-reload: ON
```

### 🏭 本番環境での実行

```bash
# 本番用サーバー起動（最適化済み）
python3 app_production.py

# 期待される出力:
# Voice Analysis Production Server Starting...
# 🎙️ ローカルアクセス: http://localhost:8080
# 🌐 ネットワークアクセス: http://192.168.x.x:8080
# 📊 Debug mode: OFF
# 🔒 Production optimizations: ON
```

### ⚙️ カスタムポート指定

```bash
# アプリケーション起動前にポート環境変数を設定
export PORT=8081  # macOS/Linux
set PORT=8081     # Windows

python3 app.py
```

### 🔧 高度な起動オプション

```bash
# バックグラウンド実行
nohup python3 app.py > server.log 2>&1 &

# プロセス確認
ps aux | grep python

# サーバー停止
pkill -f "python3 app.py"
```

## 📖 使用方法

### 👆 基本操作手順

1. **📱 ブラウザアクセス**: http://localhost:8080 にアクセス
2. **🎤 マイク許可**: ブラウザのマイクアクセス許可ダイアログで「許可」をクリック
3. **▶️ 録音開始**: 「録音開始」ボタンをクリック
4. **⏹️ 録音完了**: 10秒経過で自動停止、または手動で「停止」ボタンをクリック
5. **🔍 音声分析**: 「声紋分析」ボタンをクリックして詳細解析を実行
6. **📊 結果確認**: スペクトログラムと数値データを確認
7. **💾 ファイル保存**: 必要に応じて音声ファイルをダウンロード

### 🎛️ 高度な機能

#### 秘密の管理画面
1. メイン画面の「&」記号を素早く3回クリック
2. 管理画面にアクセスして以下の機能を利用:
   - サーバー状態監視
   - ログ表示
   - システム設定変更

#### リアルタイム波形表示
- 録音中に音声の波形がリアルタイムで表示
- 音量レベルと波形パターンを視覚的に確認
- 無音状態や音声品質の即座な判断が可能

## 🔬 技術仕様

### 🎵 音声処理仕様
- **サンプリングレート**: 44.1kHz (CD品質)
- **ビット深度**: 16bit
- **チャンネル**: モノラル
- **フォーマット**: WAV (非圧縮)
- **録音時間**: 最大10秒

### 📊 分析アルゴリズム詳細
- **窓関数**: Hann窓 (2048サンプル)
- **オーバーラップ**: 50%
- **FFTサイズ**: 2048ポイント
- **メルフィルタ**: 128バンド
- **F0推定**: YIN アルゴリズム

### 🖥️ システムアーキテクチャ
- **フロントエンド**: HTML5 + CSS3 + ES6 JavaScript
- **バックエンド**: Python 3.8+ Flask
- **音声処理**: librosa + scipy
- **データ可視化**: matplotlib
- **通信プロトコル**: HTTP/1.1 + WebSocket (リアルタイム機能)

### 🌐 ブラウザ互換性
| ブラウザ | 最小バージョン | 推奨バージョン | MediaRecorder API | WebAudio API |
|---------|---------------|---------------|------------------|--------------|
| Chrome  | 88+           | 120+          | ✅ 完全対応      | ✅ 完全対応  |
| Firefox | 85+           | 120+          | ✅ 完全対応      | ✅ 完全対応  |
| Safari  | 14+           | 17+           | ✅ 完全対応      | ⚠️ 一部制限  |
| Edge    | 88+           | 120+          | ✅ 完全対応      | ✅ 完全対応  |

## 📁 ファイル構成

```
voice_analysis/
├── 📄 app.py                 # 開発用Flaskサーバー（デバッグ機能付き）
├── 📄 app_production.py      # 本番用Flaskサーバー（最適化済み）
├── 🌐 voice_recorder.html    # メインWebアプリケーション
├── 📄 wsgi.py               # WSGI エントリーポイント
├── 📋 requirements.txt       # Python依存関係リスト
├── 📖 README.md             # このドキュメント
├── 📖 SETUP.md              # セットアップガイド
├── ⚙️ .htaccess             # Webサーバー設定
├── 🙈 .gitignore            # Git無視ファイル
└── 🎵 test_audio.wav        # テスト用音声サンプル
```

### 📄 主要ファイルの詳細

#### `app.py` (開発用)
- デバッグモード有効
- 詳細ログ出力
- ホットリロード機能
- 開発者向けエラー表示

#### `app_production.py` (本番用)
- 最適化済み実行
- エラーハンドリング強化
- ログレベル制御
- セキュリティ設定

#### `voice_recorder.html`
- レスポンシブWebUI
- MediaRecorder API実装
- リアルタイム波形表示
- エラーハンドリング

## ⚠️ エラーハンドリング

### 🔍 エラー分類と対処法

#### 📁 ファイル関連エラー

```bash
# エラー: ModuleNotFoundError: No module named 'librosa'
# 原因: librosaがインストールされていない
# 対処法:
pip install librosa

# エラー: Permission denied
# 原因: インストール権限不足
# 対処法:
pip install --user -r requirements.txt
# または
sudo pip install -r requirements.txt  # Linux/macOS
```

#### 🌐 ネットワーク関連エラー

```bash
# エラー: OSError: [Errno 48] Address already in use
# 原因: ポート8080が既に使用中
# 対処法1: 使用中プロセスの確認と終了
lsof -i :8080                    # macOS/Linux
netstat -ano | findstr :8080     # Windows
kill -9 <PID>                    # プロセス終了

# 対処法2: 別ポートでの起動
export PORT=8081
python3 app.py
```

#### 🎤 音声デバイスエラー

```bash
# エラー: NotAllowedError: Permission denied
# 原因: マイクアクセス許可が拒否された
# 対処法:
# 1. ブラウザのマイク設定を確認
# 2. プライバシー設定でマイクアクセスを許可
# 3. HTTPSアクセスを検討（一部ブラウザで必要）
```

#### 📊 音声処理エラー

```python
# app.py内でのエラーハンドリング例
def analyze_voice(audio_file_path):
    try:
        # ファイル存在確認
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_file_path}")

        # ファイルサイズ確認
        if os.path.getsize(audio_file_path) == 0:
            raise ValueError("音声ファイルが空です")

        # 音声読み込み（複数形式対応）
        try:
            y, sr = librosa.load(audio_file_path, sr=None)
        except Exception as e:
            # 代替読み込み方法
            y, sr = librosa.load(audio_file_path, sr=22050)

    except FileNotFoundError as e:
        return {"error": f"ファイルエラー: {str(e)}"}
    except ValueError as e:
        return {"error": f"データエラー: {str(e)}"}
    except Exception as e:
        return {"error": f"予期しないエラー: {str(e)}"}
```

### 🩺 システムヘルスチェック

```bash
# 総合システムチェックスクリプト
python3 -c "
import sys
import subprocess
import importlib

print('🔍 システムヘルスチェック開始...')
print(f'Python: {sys.version}')

# 必要ライブラリのチェック
required_libs = ['flask', 'librosa', 'matplotlib', 'numpy', 'scipy']
for lib in required_libs:
    try:
        mod = importlib.import_module(lib)
        print(f'✅ {lib}: {getattr(mod, \"__version__\", \"OK\")}')
    except ImportError:
        print(f'❌ {lib}: 未インストール')

print('🔍 ヘルスチェック完了')
"
```

## 🛠️ トラブルシューティング

### 🔧 よくある問題と解決策

#### 1. **音声ファイル形式エラー**

```bash
# エラー: LibsndfileError: Format not recognised / NoBackendError
# 原因: 音声ファイルの形式が認識できない、または音声処理バックエンドの問題
# 症状: 音声分析時に「音声ファイルの読み込みに失敗しました」エラー

# 解決策1: 音声処理ライブラリの追加インストール
pip install soundfile
pip install audioread[ffmpeg]

# 解決策2: FFmpegのインストール（システムレベル）
# macOS:
brew install ffmpeg

# Ubuntu/Debian:
sudo apt update
sudo apt install ffmpeg

# Windows:
# https://ffmpeg.org/download.html からダウンロードしてインストール

# 解決策3: 代替音声バックエンドのインストール
pip install pydub[mp3]
pip install mutagen

# 解決策4: すべての音声関連依存関係の再インストール
pip uninstall librosa soundfile audioread
pip install librosa soundfile audioread[ffmpeg]
```

#### 音声形式対応状況
| 形式 | 対応状況 | 必要な追加ライブラリ |
|------|----------|---------------------|
| WAV  | ✅ 標準対応 | なし |
| MP3  | ⚠️ 要追加設定 | audioread[ffmpeg] |
| M4A  | ⚠️ 要追加設定 | audioread[ffmpeg] |
| FLAC | ⚠️ 要追加設定 | audioread[ffmpeg] |
| WebM | ❌ 未対応 | ブラウザ設定変更が必要 |

#### ブラウザ録音形式の設定
```javascript
// voice_recorder.html内で録音形式を明示的に指定
const mediaRecorder = new MediaRecorder(stream, {
    mimeType: 'audio/wav'  // WAV形式を強制
});

// 対応形式の確認
console.log('対応形式:', MediaRecorder.isTypeSupported('audio/wav'));
console.log('対応形式:', MediaRecorder.isTypeSupported('audio/webm'));
```

#### 2. **インストール関連**

```bash
# 問題: pip install でエラーが発生
# 解決策1: pipのアップグレード
pip install --upgrade pip

# 解決策2: 依存関係を個別インストール
pip install wheel setuptools
pip install numpy  # 他のライブラリより先にnumpyをインストール
pip install -r requirements.txt

# 解決策3: conda使用（Anaconda環境）
conda install flask librosa matplotlib numpy scipy
```

#### 2. **音声録音問題**

```javascript
// 問題: マイクアクセスが機能しない
// 解決策: ブラウザ設定確認
// Chrome: chrome://settings/content/microphone
// Firefox: about:preferences#privacy

// HTTPSアクセスが必要な場合の対応
// 自己署名証明書での起動
python3 -c "
import ssl
context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.load_cert_chain('cert.pem', 'key.pem')
app.run(host='0.0.0.0', port=8443, ssl_context=context)
"
```

#### 3. **パフォーマンス問題**

```bash
# 問題: 分析処理が遅い
# 解決策1: Python最適化
python3 -O app.py  # 最適化モードで実行

# 解決策2: メモリ使用量確認
python3 -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'メモリ使用量: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"

# 解決策3: librosaの設定調整
# app.py内で以下を追加:
# import librosa
# librosa.cache.clear()  # キャッシュクリア
```

#### 4. **ブラウザ互換性問題**

| 問題 | 原因 | 解決策 |
|------|------|--------|
| 録音ボタンが反応しない | MediaRecorder API未対応 | Chrome 88+またはFirefox 85+を使用 |
| 音声が再生されない | WebAudio API制限 | ユーザーインタラクション後に再生実行 |
| 波形表示が崩れる | Canvas API制限 | ブラウザのハードウェアアクセラレーション確認 |

### 📊 デバッグモード

```bash
# 詳細デバッグ情報の有効化
export FLASK_DEBUG=1
export FLASK_ENV=development
python3 app.py

# ログレベル設定
export LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
python3 app.py
```

### 🔍 ログ分析

```bash
# リアルタイムログ監視
tail -f server.log

# エラーログの抽出
grep -i error server.log

# アクセスログの分析
grep "POST /analyze" server.log | wc -l  # 分析実行回数
```

## ⚡ パフォーマンス最適化

### 🚀 サーバー最適化

```python
# app_production.py での最適化設定
app.config.update(
    SEND_FILE_MAX_AGE_DEFAULT=31536000,  # 静的ファイルキャッシュ
    JSON_SORT_KEYS=False,                # JSON並び替え無効化
    JSONIFY_PRETTYPRINT_REGULAR=False    # JSON整形無効化
)

# 音声処理の最適化
import librosa
librosa.set_cache_level(40)  # キャッシュレベル調整
```

### 💾 メモリ使用量最適化

```python
# 大きな音声ファイル処理時のメモリ管理
def analyze_voice_optimized(audio_file_path):
    try:
        # チャンクサイズ指定での読み込み
        y, sr = librosa.load(audio_file_path, sr=22050, duration=10.0)

        # 不要な変数の明示的削除
        del audio_file_path

        # ガベージコレクション実行
        import gc
        gc.collect()

    finally:
        # メモリクリーンアップ
        plt.close('all')
```

### 🌐 ネットワーク最適化

```html
<!-- voice_recorder.html での最適化 -->
<script>
// 音声データの圧縮送信
const compressAudio = (audioBlob) => {
    // 必要に応じて音声圧縮ロジックを実装
    return audioBlob;
};

// 非同期分析リクエスト
const analyzeAudioAsync = async (audioBlob) => {
    const formData = new FormData();
    formData.append('audio', compressAudio(audioBlob));

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        return await response.json();
    } catch (error) {
        console.error('分析エラー:', error);
        throw error;
    }
};
</script>
```

## 👨‍💻 開発者向け情報

### 🔧 開発環境セットアップ

```bash
# 開発用依存関係の追加インストール
pip install pytest black flake8 mypy

# コード品質チェック
black *.py                    # コードフォーマット
flake8 *.py                   # リンティング
mypy *.py                     # 型チェック

# テスト実行
pytest tests/                 # ユニットテスト実行
```

### 🧪 テスト用音声ファイル生成

```python
# test_audio_generator.py
import numpy as np
import librosa
import soundfile as sf

def generate_test_audio():
    """テスト用音声ファイルを生成"""
    duration = 5.0  # 秒
    sr = 44100

    # 440Hz正弦波（A4音）
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    # ファイル保存
    sf.write('test_audio_generated.wav', audio, sr)
    print("テスト音声ファイルを生成しました: test_audio_generated.wav")

if __name__ == "__main__":
    generate_test_audio()
```

### 📊 API エンドポイント詳細

#### `POST /analyze`
```json
{
  "request": {
    "content-type": "multipart/form-data",
    "audio": "binary_audio_data"
  },
  "response": {
    "mel_spectrogram": "base64_image_data",
    "stft_spectrogram": "base64_image_data",
    "f0_plot": "base64_image_data",
    "waveform": "base64_image_data",
    "features": {
      "spectral_centroid": [数値配列],
      "mfcc": [数値配列],
      "zero_crossing_rate": [数値配列],
      "spectral_rolloff": [数値配列],
      "f0_mean": 数値,
      "f0_std": 数値
    },
    "metadata": {
      "duration": 数値,
      "sample_rate": 数値,
      "file_size": 数値
    }
  }
}
```

### 🔒 セキュリティ考慮事項

```python
# セキュリティ強化設定
from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

# レート制限設定
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# ファイルアップロード制限
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB制限

# セキュアヘッダー設定
@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response
```

### 📈 モニタリングと分析

```python
# アクセス統計の実装
import time
from collections import defaultdict

class AccessStats:
    def __init__(self):
        self.stats = defaultdict(int)
        self.start_time = time.time()

    def record_access(self, endpoint):
        self.stats[endpoint] += 1

    def get_stats(self):
        uptime = time.time() - self.start_time
        return {
            'uptime_hours': uptime / 3600,
            'total_requests': sum(self.stats.values()),
            'endpoints': dict(self.stats)
        }

# 使用例
stats = AccessStats()

@app.route('/stats')
def get_statistics():
    return jsonify(stats.get_stats())
```

## 📞 サポート・連絡先

### 🐛 バグレポート
- GitHubでissueを作成してください
- エラーメッセージ、実行環境、再現手順を含めてください

### 💡 機能要望
- GitHubのFeature Requestテンプレートを使用してください

### 📚 追加リソース
- [librosa公式ドキュメント](https://librosa.org/doc/latest/)
- [Flask公式ドキュメント](https://flask.palletsprojects.com/)
- [Web Audio API仕様](https://www.w3.org/TR/webaudio/)

---

**🎉 Voice Analysis Systemをお使いいただき、ありがとうございます！**