# LUNARON Investment Dashboard v3.0 (Ollama版)

**完全無料・ローカルLLM** (Gemma3 via Ollama) ＋ yfinance を活用した  
個人用クオンツ・ダッシュボードです。APIキー不要！

---

## ⚠️ よくあるエラーと解決方法

### ① `InvalidGitRepositoryError` が出る

**原因**: uvicorn を間違ったディレクトリから起動すると、  
プロジェクトの `main.py` ではなく site-packages 内の別の `main.py` が読まれてしまいます。

**解決策**: 必ず `backend/` フォルダに移動してから起動してください。

```powershell
# ❌ NG（プロジェクトルートから起動）
cd C:\dev\lunaron
uvicorn main:app --reload --port 8000   ← これがエラーの原因

# ✅ OK（backend フォルダに入ってから起動）
cd C:\dev\lunaron\backend
uvicorn main:app --reload --port 8000
```

### ② `pip install -r requirements.txt` で赤いエラーが出る

**原因**: 旧バージョンは `anthropic` パッケージが必要でしたが、  
バージョンの競合や不要な依存が大量に入っていました。

**解決策**: 今回の修正版では `anthropic` を完全に削除しました。  
仮想環境を作り直してインストールしてください（下記セットアップ参照）。

---

## 📁 ディレクトリ構成

```
lunaron/
├── backend/
│   ├── main.py            # FastAPI バックエンド（Ollama版）
│   └── requirements.txt   # Python 依存パッケージ
├── frontend/
│   └── index.html         # React フロントエンド（CDN版）
└── README.md
```

---

## 🚀 セットアップ手順

### STEP 1: Ollama をインストール

https://ollama.com からインストーラーをダウンロードして実行。

### STEP 2: Gemma3 モデルをダウンロード

```powershell
ollama pull gemma3
```

> 軽量版を使う場合（VRAM が少ない PC 向け）:
> ```powershell
> ollama pull gemma3:1b   # 約 800MB
> ollama pull gemma2:2b   # 約 1.6GB（旧世代）
> ```
> `main.py` の `OLLAMA_MODEL = "gemma3"` を `"gemma3:1b"` などに変更してください。

### STEP 3: Ollama を起動

```powershell
ollama serve
```

> インストール後は自動起動している場合もあります。  
> タスクトレイに Ollama のアイコンがあれば起動済みです。

### STEP 4: Python 仮想環境を作成

```powershell
cd C:\dev\lunaron\backend   # ← backend フォルダに移動（重要！）
python -m venv venv
venv\Scripts\activate
```

### STEP 5: パッケージをインストール

```powershell
pip install -r requirements.txt
```

`anthropic` が含まれていないので、赤いエラーは出なくなります。

### STEP 6: バックエンドを起動

```powershell
# backend/ フォルダ内にいることを確認してから:
uvicorn main:app --reload --port 8000
```

確認: http://localhost:8000 → `{"status":"ok","app":"Lunaron Investment API v3 (Ollama)"}` が表示されればOK。

Ollama の状態確認: http://localhost:8000/api/ollama/status

### STEP 7: フロントエンドを開く

```powershell
cd ..\frontend
python -m http.server 3000
```

→ http://localhost:3000 にアクセス

---

## 🔧 機能一覧

| タブ | 機能 |
|------|------|
| チャート分析 | RSI / MACD / ボリンジャーバンド / シャープレシオ |
| ポートフォリオ | 銘柄・数量・取得単価を入力 → 損益計算・AI診断 |
| GBM予測 | 幾何ブラウン運動モデルで7日間の価格帯を確率的に予測 |
| AI分析 | Ollama + Gemma3 による完全無料・ローカルLLM分析 |

---

## 📊 対応銘柄

| ID   | 名前        | yfinance シンボル |
|------|-------------|-------------------|
| SPY  | S&P 500     | SPY               |
| QQQ  | NASDAQ 100  | QQQ               |
| BTC  | Bitcoin     | BTC-USD           |
| GOLD | Gold        | GLD               |
| NVDA | NVIDIA      | NVDA              |
| TSLA | Tesla       | TSLA              |
| AAPL | Apple       | AAPL              |
| ETH  | Ethereum    | ETH-USD           |

---

## 💡 モデル変更方法

`backend/main.py` の先頭付近を編集するだけです:

```python
OLLAMA_MODEL = "gemma3"      # デフォルト（高品質、約5GB）
# OLLAMA_MODEL = "gemma3:1b" # 超軽量（低スペックPC向け）
# OLLAMA_MODEL = "gemma2"    # Gemma2（旧世代）
# OLLAMA_MODEL = "llama3.2"  # LLaMA 3.2 も使えます
```

---

## ⚠️ 注意事項

- 本ツールは**教育・研究目的**のものです。
- 投資判断はご自身の責任で行ってください。
- yfinance のデータは遅延がある場合があります。
- **Not financial advice.**
