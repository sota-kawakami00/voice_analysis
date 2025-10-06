# ボイスレコーダー & 声紋分析システム

10秒制限のボイスレコーダーで録音した音声の声紋分析を行うWebアプリケーションです。

## 機能

- **音声録音**: ブラウザから最大10秒間の音声録音
- **音声再生**: 録音した音声の再生
- **音声ダウンロード**: 録音したファイルのダウンロード
- **声紋分析**: Pythonによる高度な音声分析
  - メルスペクトログラム
  - STFTスペクトログラム
  - 基本周波数 (F0) の推定
  - 音声波形表示
  - 各種音声特徴量の計算
- **リアルタイム波形表示**: 録音中の音声波形をリアルタイムで表示
- **ローディング表示**: 分析処理中のユーザーフレンドリーなUI
- **秘密の管理画面**: &マークをトリプルクリックでアクセス可能

## クイックスタート

### 前提条件
- Python 3.8以上
- マイク付きデバイス
- モダンブラウザ（Chrome, Firefox, Safari等）

### 1. 依存関係のインストール
```bash
pip3 install -r requirements.txt
```

### 2. アプリケーション起動
```bash
python3 app.py
```

### 3. ブラウザでアクセス
http://localhost:8080 を開く

## 詳細セットアップ

### Step 1: Python環境の確認
```bash
python3 --version  # Python 3.8以上が必要
pip3 --version
```

### Step 2: プロジェクトのクローン
```bash
git clone <repository-url>
cd voice_analysis
```

### Step 3: 仮想環境の作成（推奨）
```bash
# 仮想環境作成
python3 -m venv venv

# アクティベート
source venv/bin/activate  # macOS/Linux
# または
venv\Scripts\activate     # Windows
```

### Step 4: 依存関係インストール
```bash
# 一括インストール
pip install -r requirements.txt

# または個別インストール
pip install Flask==2.3.3
pip install librosa==0.10.1
pip install matplotlib==3.7.2
pip install numpy==1.24.3
pip install scipy==1.11.2
```

### Step 5: サーバー起動と動作確認
```bash
python3 app.py

# 以下のメッセージが表示されれば成功:
# Voice Analysis Server Starting...
# ローカルアクセス: http://localhost:8080
# ネットワークアクセス: http://[IP]:8080
```

## 使い方

1. 「録音開始」ボタンをクリック（マイクアクセスを許可）
2. 音声を録音（最大10秒で自動停止）
3. 「声紋分析」ボタンをクリックして分析実行
4. スペクトログラムと音声特徴量を確認

## 技術仕様

- **フロントエンド**: HTML5, CSS, JavaScript (MediaRecorder API)
- **バックエンド**: Python Flask
- **音声処理**: librosa, scipy
- **可視化**: matplotlib
- **分析内容**:
  - メルスペクトログラム
  - STFT スペクトログラム
  - 基本周波数 (F0)
  - スペクトル重心
  - MFCC
  - ゼロクロッシングレート
  - スペクトラルロールオフ

## ファイル構成

```
voice_analysis/
├── app.py                 # Flaskサーバー
├── voice_recorder.html    # フロントエンド
├── requirements.txt       # Python依存関係
├── SETUP.md              # セットアップガイド
└── README.md             # このファイル
```

## トラブルシューティング

### よくある問題

1. **ModuleNotFoundError: No module named 'librosa'**
   ```bash
   pip install librosa
   ```

2. **Permission denied エラー**
   ```bash
   pip install --user -r requirements.txt
   ```

3. **Port already in use**
   - 別のアプリケーションがポート8080を使用中
   - app.py内のポート番号を変更するか、使用中のプロセスを終了

4. **マイクアクセスが拒否される**
   - HTTPSが必要な場合があります
   - Chromeの場合: chrome://flags/#unsafely-treat-insecure-origin-as-secure

### システム要件
- Python 3.8+
- 2GB以上の空きメモリ
- マイク付きデバイス
- Chrome, Firefox, Safari等のモダンブラウザ

### ネットワーク内での使用
アプリケーション起動時に表示されるIPアドレス（例: http://192.168.1.100:8080）で同じネットワーク内の他のデバイスからアクセス可能です。

## 注意事項

- HTTPS環境でないとマイクアクセスが制限される場合があります
- 初回実行時、librosaの依存ライブラリの読み込みに時間がかかる場合があります