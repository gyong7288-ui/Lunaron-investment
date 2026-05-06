FROM python:3.11-slim

WORKDIR /app

# 依存関係インストール
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリ全体をコピー
COPY . .

# HF Spaces は 7860 ポートを使う
EXPOSE 7860

# 起動コマンド（frontend ディレクトリは backend/main.py から相対パスで参照）
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
