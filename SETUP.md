# 音声分析システム - ローカルセットアップガイド

## クイックスタート

### 1. 依存関係のインストール
```bash
pip3 install -r requirements.txt
```

### 2. アプリケーション起動
```bash
python3 app.py
```

### 3. アクセス
ブラウザで http://localhost:8080 を開く

## 詳細セットアップ

### Step 1: Python環境の確認
```bash
python3 --version  # Python 3.8以上が必要
pip3 --version
```

### Step 2: 仮想環境の作成（推奨）
```bash
# 仮想環境作成
python3 -m venv venv

# アクティベート
source venv/bin/activate  # macOS/Linux
# または
venv\Scripts\activate     # Windows
```

### Step 3: 依存関係インストール
```bash
pip install Flask==2.3.3
pip install librosa==0.10.1
pip install matplotlib==3.7.2
pip install numpy==1.24.3
pip install scipy==1.11.2

# または一括インストール
pip install -r requirements.txt
```

### Step 4: 動作確認
```bash
# サーバー起動
python3 app.py

# 以下のメッセージが表示されれば成功:
# Voice Analysis Server Starting...
# ローカルアクセス: http://localhost:8080
# ネットワークアクセス: http://[IP]:8080
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
アプリケーション起動時に表示されるIPアドレス（例: http://192.168.1.100:8080）で
同じネットワーク内の他のデバイスからアクセス可能です。

## 機能一覧
- ✅ 10秒間の音声録音
- ✅ リアルタイム波形表示
- ✅ 音声ファイルダウンロード
- ✅ 詳細な声紋分析（スペクトログラム、MFCC、F0等）
- ✅ ローディング表示付きUI
- ✅ 秘密の管理画面（&マークをトリプルクリック）