from flask import Flask, request, jsonify, send_file, render_template_string
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # バックエンド設定

# フォント設定（デフォルトのみ使用）
plt.rcParams['font.family'] = ['DejaVu Sans']
# 日本語フォント警告を回避
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
import io
import base64
from scipy import signal
import os
import tempfile

app = Flask(__name__)

# アップロードファイル用の一時ディレクトリ
UPLOAD_FOLDER = tempfile.mkdtemp()

def analyze_voice(audio_file_path):
    """
    音声ファイルを分析し、スペクトログラム画像をbase64で返す
    """
    try:
        print(f"[DEBUG] Starting analysis of: {audio_file_path}")

        # ファイル存在確認
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_file_path}")

        # ファイルサイズ確認
        file_size = os.path.getsize(audio_file_path)
        print(f"[DEBUG] File size: {file_size} bytes")

        if file_size == 0:
            raise ValueError("音声ファイルが空です")

        # 音声データの読み込み（複数の方法を試行）
        y = None
        sr = None
        load_method = ""

        try:
            print("[DEBUG] Trying librosa.load with default settings...")
            y, sr = librosa.load(audio_file_path, sr=None)
            load_method = "librosa (default)"
        except Exception as e1:
            print(f"[DEBUG] First load attempt failed: {e1}")
            try:
                print("[DEBUG] Trying librosa.load with sr=22050...")
                y, sr = librosa.load(audio_file_path, sr=22050)
                load_method = "librosa (22050Hz)"
            except Exception as e2:
                print(f"[DEBUG] Second load attempt failed: {e2}")
                try:
                    print("[DEBUG] Trying librosa.load with sr=16000...")
                    y, sr = librosa.load(audio_file_path, sr=16000)
                    load_method = "librosa (16000Hz)"
                except Exception as e3:
                    print(f"[DEBUG] All load attempts failed: {e3}")
                    raise ValueError(f"音声ファイルの読み込みに失敗しました: {e3}")

        print(f"[DEBUG] Audio loaded successfully using {load_method}")
        print(f"[DEBUG] Sample rate: {sr} Hz, Duration: {len(y)/sr:.2f}s, Samples: {len(y)}")

        # データの検証
        if len(y) == 0:
            raise ValueError("音声データが空です")
        if sr <= 0:
            raise ValueError("無効なサンプリングレートです")

        # 音声データの統計
        print(f"[DEBUG] Audio stats - Min: {np.min(y):.6f}, Max: {np.max(y):.6f}, Mean: {np.mean(y):.6f}")

        # スペクトログラムと声紋の生成
        plt.figure(figsize=(14, 12))

        # 音声波形
        plt.subplot(3, 2, 1)
        times = np.linspace(0, len(y) / sr, len(y))
        plt.plot(times, y, color='blue', linewidth=0.8)
        plt.title('Voice Waveform', fontsize=12)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)

        # メルスペクトログラム（声紋として表示）
        plt.subplot(3, 2, 2)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
        plt.title('Voice Print (Mel Spectrogram)', fontsize=12)
        plt.colorbar(format='%+2.0f dB')

        # STFT スペクトログラム
        plt.subplot(3, 2, 3)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', cmap='viridis')
        plt.title('STFT Spectrogram', fontsize=12)
        plt.colorbar(format='%+2.0f dB')

        # 基本周波数 (F0) の推定
        plt.subplot(3, 2, 4)
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            times = librosa.frames_to_time(range(len(f0)), sr=sr)
            # 有声音のみ表示
            voiced_f0 = np.where(voiced_flag, f0, np.nan)
            plt.plot(times, voiced_f0, 'o-', color='red', linewidth=2, markersize=3)
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Pitch (F0) Tracking', fontsize=12)
            plt.grid(True, alpha=0.3)
        except Exception as e:
            plt.text(0.5, 0.5, 'F0 estimation failed', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Pitch (F0) Tracking', fontsize=12)

        # スペクトル重心の時系列
        plt.subplot(3, 2, 5)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames, sr=sr)
        plt.plot(t, spectral_centroids, color='green', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Hz')
        plt.title('Spectral Centroid', fontsize=12)
        plt.grid(True, alpha=0.3)

        # ゼロクロッシングレート
        plt.subplot(3, 2, 6)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        frames = range(len(zcr))
        t = librosa.frames_to_time(frames, sr=sr)
        plt.plot(t, zcr, color='orange', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Rate')
        plt.title('Zero Crossing Rate', fontsize=12)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # 画像をbase64エンコード
        try:
            print("[DEBUG] Generating spectrogram image...")
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_size = len(img_buffer.getvalue())
            print(f"[DEBUG] Image generated successfully: {img_size} bytes")
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            print("[DEBUG] Image encoded to base64 successfully")
        except Exception as img_error:
            plt.close()
            print(f"[ERROR] Image generation failed: {img_error}")
            raise ValueError(f"スペクトログラム画像の生成に失敗しました: {img_error}")

        # 音声特徴量の計算
        try:
            print("[DEBUG] Extracting audio features...")
            features = extract_audio_features(y, sr)
            print("[DEBUG] Feature extraction completed successfully")
        except Exception as feat_error:
            print(f"[ERROR] Feature extraction failed: {feat_error}")
            raise ValueError(f"音声特徴量の抽出に失敗しました: {feat_error}")

        print("[DEBUG] Analysis completed successfully")
        return {
            'success': True,
            'spectrogram': img_base64,
            'features': features,
            'debug_info': {
                'load_method': load_method,
                'sample_rate': sr,
                'duration': f"{len(y)/sr:.2f}s",
                'samples': len(y)
            }
        }

    except Exception as e:
        print(f"[ERROR] Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }

def extract_audio_features(y, sr):
    """
    音声から特徴量を抽出
    """
    try:
        # 基本統計
        duration = len(y) / sr

        # スペクトル重心
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

        # MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # ゼロクロッシングレート
        zcr = librosa.feature.zero_crossing_rate(y)[0]

        # スペクトラルロールオフ
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

        # 基本周波数
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_clean = f0[~np.isnan(f0)]

        features = {
            '継続時間': f'{duration:.2f}秒',
            'サンプリングレート': f'{sr} Hz',
            '平均スペクトル重心': f'{np.mean(spectral_centroids):.2f} Hz',
            '平均ゼロクロッシングレート': f'{np.mean(zcr):.4f}',
            '平均スペクトラルロールオフ': f'{np.mean(spectral_rolloff):.2f} Hz',
            '基本周波数 (平均)': f'{np.mean(f0_clean):.2f} Hz' if len(f0_clean) > 0 else 'N/A',
            '基本周波数 (標準偏差)': f'{np.std(f0_clean):.2f} Hz' if len(f0_clean) > 0 else 'N/A',
            'MFCCの次元': f'{mfccs.shape[0]} x {mfccs.shape[1]}'
        }

        return features

    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def index():
    """
    メインページ
    """
    return send_file('voice_recorder.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    音声分析エンドポイント
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': '音声ファイルが見つかりません'})

        audio_file = request.files['audio']

        if audio_file.filename == '':
            return jsonify({'success': False, 'error': 'ファイルが選択されていません'})

        # 一時ファイルとして保存
        temp_path = os.path.join(UPLOAD_FOLDER, 'temp_audio.wav')
        audio_file.save(temp_path)

        # 音声分析実行
        result = analyze_voice(temp_path)

        # 一時ファイル削除
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"Voice Analysis Server Starting...")
    print(f"ローカルアクセス: http://localhost:8080")
    print(f"ネットワークアクセス: http://{local_ip}:8080")
    app.run(debug=True, host='0.0.0.0', port=8080)