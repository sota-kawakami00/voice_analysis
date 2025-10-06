#!/usr/bin/env python3
"""
WSGI entry point for production deployment
お名前.comサーバー対応のWSGIアプリケーション
"""

import sys
import os

# 現在のディレクトリをPythonパスに追加
sys.path.insert(0, os.path.dirname(__file__))

# 本番環境用のアプリケーションをインポート
from app_production import app

# WSGIアプリケーションとして公開
application = app

if __name__ == "__main__":
    application.run()