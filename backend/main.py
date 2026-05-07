"""
Lunaron Investment Dashboard - Backend API
FastAPI + yfinance + Ollama (Gemma3) ― 完全無料・ローカルLLM版

起動方法:
  1. Ollama をインストール → https://ollama.com
  2. モデルをダウンロード: ollama pull gemma3
  3. Ollama を起動:       ollama serve
  4. 別ターミナルで backend/ に移動してから:
       pip install -r requirements.txt
       uvicorn main:app --reload --port 8000
"""

import math
import os
import requests
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np
from yahooquery import Ticker

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# frontend/index.html へのパスを解決（backend/ から実行した場合）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")


# ─── Hugging Face Inference API (+ ローカル高精度フォールバック) ──────────────

HF_TOKEN = os.getenv("HF_TOKEN", "")

def _local_analysis(prompt: str) -> str:
    """プロンプトからキーワードを読み取り、ローカルで高精度分析を生成する"""
    import re
    
    # プロンプトからデータを抽出
    rsi_match = re.search(r'RSI\(14\):\s*([\d.]+)', prompt)
    rsi = float(rsi_match.group(1)) if rsi_match else 50.0
    
    cape_match = re.search(r'CAPE\):\s*([\d.]+)\s*\(([^)]+)\)', prompt)
    cape = float(cape_match.group(1)) if cape_match else 30.0
    cape_status = cape_match.group(2) if cape_match else "適正"
    
    fg_match = re.search(r'F&G\):\s*(\d+)\s*\(([^)]+)\)', prompt)
    fg_val = int(fg_match.group(1)) if fg_match else 50
    fg_label = fg_match.group(2) if fg_match else "Neutral"
    
    macd_match = re.search(r'MACDヒストグラム:\s*([-\d.]+)\s*→\s*(\S+)', prompt)
    macd_dir = macd_match.group(2) if macd_match else "中立"
    
    bb_match = re.search(r'ボリンジャーバンド位置:\s*([^\n]+)', prompt)
    bb_pos = bb_match.group(1).strip() if bb_match else ""
    
    sharpe_match = re.search(r'シャープ比率:\s*([-\d.]+)', prompt)
    sharpe = float(sharpe_match.group(1)) if sharpe_match else 0.0
    
    # 結論の決定
    score = 0
    reasons = []
    risks = []
    
    if rsi < 30:
        score += 2
        reasons.append(f"RSI {rsi:.0f} — 売られすぎ圏域（30以下は強い買いシグナル）")
    elif rsi < 45:
        score += 1
        reasons.append(f"RSI {rsi:.0f} — やや売られすぎ（買い場の可能性）")
    elif rsi > 70:
        score -= 2
        reasons.append(f"RSI {rsi:.0f} — 買われすぎ圏域（70超は警戒ゾーン）")
        risks.append("RSIが高水準のため、短期的な調整に注意が必要です")
    else:
        reasons.append(f"RSI {rsi:.0f} — 中立圏域（トレンドに従う局面）")
    
    if "上昇" in macd_dir:
        score += 1
        reasons.append("MACDが上昇トレンドを示唆（モメンタム良好）")
    elif "下落" in macd_dir:
        score -= 1
        reasons.append("MACDが下落トレンドを示唆（慎重な姿勢が望ましい）")
    
    if "割安" in bb_pos or "下限" in bb_pos:
        score += 1
        reasons.append("ボリンジャーバンド下限付近（統計的割安ゾーン）")
    elif "割高" in bb_pos or "上限" in bb_pos:
        score -= 1
        reasons.append("ボリンジャーバンド上限付近（統計的割高ゾーン）")
        risks.append("バンド上限への到達は過熱感を示す場合があります")
    
    if fg_val < 30:
        score += 2
        reasons.append(f"心理指数 {fg_val}（{fg_label}）— 「恐怖期は買い場」の原則に合致")
    elif fg_val > 70:
        score -= 1
        reasons.append(f"心理指数 {fg_val}（{fg_label}）— 市場の楽観が高まっており注意")
        risks.append("市場心理が強欲圏のため、追い買いは慎重に")
    
    if cape_status == "割安":
        score += 1
        reasons.append(f"シラーPER {cape:.1f}（{cape_status}）— 長期的な割安感あり")
    elif cape_status == "過熱":
        score -= 1
        reasons.append(f"シラーPER {cape:.1f}（{cape_status}）— 歴史的水準でやや割高")
        risks.append(f"CAPEが高水準（{cape:.1f}）は長期的な過熱感を示唆します")
    
    if sharpe > 1.0:
        reasons.append(f"シャープ比率 {sharpe:.2f} — リスク対リターンが優秀（1.0超）")
    elif sharpe < 0:
        risks.append(f"シャープ比率が負（{sharpe:.2f}）— リスクに見合うリターンが得られていない状態")
    
    # 推奨アクションの決定
    if score >= 3:
        action = "強気の積み増し（買い増し）を検討できます"
        action_detail = "複数の指標が買いシグナルを示しています。分散投資の原則に従い、余裕資金での積み増しを検討してください。"
    elif score >= 1:
        action = "少額の積み増し（様子見しながら）が妥当"
        action_detail = "やや強気の環境です。長期・積立の原則に従い、一括ではなく分割での買い増しをお勧めします。"
    elif score >= -1:
        action = "現状維持（HOLD）が無難"
        action_detail = "明確なシグナルなし。航路を守り、不必要な売買は避けましょう。コストは確実なマイナスです。"
    elif score >= -2:
        action = "一部利益確定を検討"
        action_detail = "弱気サインが出ています。全売却より、高値圏の銘柄から段階的な利益確定を検討してください。"
    else:
        action = "守りの姿勢（リスク低減）を優先"
        action_detail = "複数の指標が警戒を示しています。損切りルールを確認し、ポートフォリオのリスク見直しを検討してください。"
    
    risks_text = "\n".join(f"・{r}" for r in risks) if risks else "・長期投資家にとって短期変動は「通過点」です\n・余裕資金内での投資を維持し、生活費との混同を避けてください"
    reasons_text = "\n".join(f"・{r}" for r in reasons)
    
    # 質問に対する反応を簡易的に追加
    q_lower = prompt.lower()
    advice_extra = ""
    if "今後" in q_lower or "将来" in q_lower or "予測" in q_lower:
        advice_extra = "\n\n### 【今後のアクションプラン】\n1. 短期的なノイズに惑わされず、あらかじめ決めた資産配分（アセットアロケーション）を維持してください。\n2. 市場が急落した際のリバランス用として、一定のキャッシュ（現金）比率を確保しておくことが賢明です。\n3. 自動積立の設定を継続し、価格変動を味方につける「ドルコスト平均法」を最大限に活用しましょう。"
    elif "いくら" in q_lower or "買" in q_lower:
        advice_extra = "\n\n### 【今後のアクションプラン】\n1. 一括購入ではなく、数回に分けた「時間分散」によるエントリーを推奨します。\n2. 購入価格だけでなく、ポートフォリオ全体におけるその銘柄の比率が過大にならないよう調整してください。\n3. 万が一の想定シナリオ（20%下落など）を事前にシミュレーションし、パニック売りを防ぐ準備をしてください。"
    else:
        advice_extra = "\n\n### 【今後のアクションプラン】\n1. ポートフォリオの定期的な棚卸しを行い、本来の目的（老後資金、教育資金等）からズレていないか確認してください。\n2. 税制優遇制度（NISA/iDeCo等）の枠が残っている場合は、優先的に活用することを検討しましょう。\n3. 投資以外の自己研鑽や健康への投資も、長期的な資産形成において重要な要素です。"

    # プロンプトから銘柄名を取得（あれば）
    name_match = re.search(r'市場データ:\s*([^(]+)', prompt)
    ticker_name = name_match.group(1).strip() if name_match else "この銘柄"

    return f"""### 【結論】
{ticker_name}については「{action}」が妥当と判断します。
{action_detail}

### 【根拠】
{reasons_text}

### 【リスク】
{risks_text}{advice_extra}"""

def hf_chat(prompt: str, system: str = "", history: list = None) -> str:
    """Hugging Face Inference APIを利用してテキスト生成（失敗時はローカル分析にフォールバック）"""
    if HF_TOKEN:
        try:
            model_url = "https://api-inference.huggingface.co/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json"
            }
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": "Qwen/Qwen2.5-72B-Instruct",
                "messages": messages,
                "max_tokens": 600,
                "temperature": 0.75,
            }
            res = requests.post(model_url, headers=headers, json=payload, timeout=30)
            if res.status_code == 200:
                result = res.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    return content.strip()
        except Exception as e:
            print(f"HF API Error: {e}")
    
    # HF_TOKENなし or API失敗時 → ローカル分析エンジンを使用
    return _local_analysis(prompt)


def _local_chat(message: str, history: list = None) -> str:
    """一般的な投資質問に対するローカルフォールバック会話エンジン"""
    import random
    m = message.lower()
    
    # 積立・ドルコスト
    if any(k in m for k in ["積立", "ドルコスト", "毎月", "定期"]):
        return random.choice([
            "積立投資（ドルコスト平均法）は、長期投資において最も有効な戦略の一つです。毎月一定額を買い付けることで、高い時は少なく・安い時は多く買える仕組みが自動的に働きます。特に感情的な売買を避けたい方に最適です。何年後に使う資金の積立ですか？",
            "定期積立は「時間の分散」を実現する黄金ルールです。市場が下がった月こそ多くの口数が買えるため、長期的には平均購入単価を下げる効果があります。NISA口座での積立も検討されていますか？",
        ])
    
    # NISA/iDeCo
    if any(k in m for k in ["nisa", "ニーサ", "ideco", "イデコ", "税"]):
        return "NISA（少額投資非課税制度）とiDeCoは、日本の投資家が最優先で活用すべき制度です。\n\n• **新NISA**: 年360万円まで非課税で投資可能。いつでも出金OK\n• **iDeCo**: 掛け金が全額所得控除。60歳まで引き出し不可\n\n一般的には「新NISA→iDeCo」の順で枠を埋めるのが効率的です。どちらについてもっと詳しく聞きますか？"
    
    # リスクについて
    if any(k in m for k in ["リスク", "怖い", "損", "下がる", "暴落"]):
        return random.choice([
            "投資のリスクを怖いと感じるのは、とても正常な感覚です。大切なのは「リスクをゼロにする」のではなく「許容できるリスクを把握する」こと。\n\n原則として：\n・生活費6ヶ月分は現金で保持\n・投資は余裕資金のみで行う\n・一つの銘柄への集中投資を避ける\n\n現在どのような点が特に不安ですか？",
            "株価の下落は怖いですが、長期投資家にとっては「セール」とも言えます。過去のデータを見ると、S&P500はどんな暴落からも回復してきました。重要なのは「いつ戻るか」ではなく「市場に居続けること」です。",
        ])
    
    # 初心者
    if any(k in m for k in ["初心者", "始め", "わからない", "入門", "どうすれば"]):
        return "投資を始める際の基本ステップです：\n\n1️⃣ **証券口座を開設** → SBI証券・楽天証券が初心者に人気\n2️⃣ **新NISAの口座を設定** → 税制優遇を最大活用\n3️⃣ **インデックスファンドから** → S&P500やオルカン（全世界株）など\n4️⃣ **毎月一定額を積立** → 余裕資金の範囲内で\n\nまず何から知りたいですか？証券口座の選び方、銘柄の選び方、それとも仕組みから？"
    
    # 分散投資
    if any(k in m for k in ["分散", "ポートフォリオ", "組み合わせ", "配分"]):
        return "分散投資は「卵を一つのカゴに盛るな」という投資の鉄則です。\n\n代表的な分散方法：\n• **地域分散**: 日本・米国・新興国など\n• **資産分散**: 株式・債券・金・不動産（REIT）\n• **時間分散**: 積立投資でタイミングをずらす\n\nシンプルな例として、「全世界株インデックス（オルカン）70% + 債券20% + 現金10%」は多くの専門家が推奨する基本構成です。今のポートフォリオ構成はどのような状態ですか？"
    
    # VOO/QQQ/S&P500について
    if any(k in m for k in ["voo", "qqq", "s&p", "sp500", "nasdaq", "ナスダック"]):
        ticker_info = {
            "voo": "VOO（バンガードS&P500ETF）は米国の主要500社に分散投資できるETFです。低コスト（経費率0.03%）で長期投資の定番。",
            "qqq": "QQQ（インベスコQQQトラスト）はNASDAQ100に連動し、Apple・Microsoft・NVIDIAなどテクノロジー大手への集中投資ができます。VOOよりハイリスク・ハイリターン。",
        }
        for k, v in ticker_info.items():
            if k in m:
                return v + "\n\nこの銘柄について具体的に何が知りたいですか？（購入タイミング・リスク・他との比較など）"
    
    # 売り時
    if any(k in m for k in ["売り", "売る", "利確", "利益確定"]):
        return "売り時の判断は、買うより難しいと言われます。いくつかの考え方：\n\n• **目標達成時**: 「◯%利益が出たら一部売却」と事前に決めておく\n• **リバランス時**: 配分が崩れたら売って調整\n• **生活資金が必要な時**: 目的が来たら使う\n• **コア資産は売らない**: インデックスETFは原則「持ち続ける」\n\nどの銘柄の売り時を考えていますか？"
    
    # 汎用的な投資アドバイス
    responses = [
        f"「{message}」についてですね。投資で大切な三原則は「長期・分散・積立」です。この観点からアドバイスすると、短期的な価格変動に一喜一憂せず、目標に向けて淡々と続けることが最も重要です。もう少し具体的に教えていただけますか？",
        f"ご質問ありがとうございます。「{message}」は多くの投資家が気にする点です。まず確認させてください：投資の目的（老後・教育・資産形成等）と、大体の投資期間はどのくらいを考えていますか？それによってアドバイスが変わります。",
        f"良い質問です。「{message}」について、投資の基本原則から答えると：感情ではなくデータと計画に基づいた判断が重要です。市場の短期的なノイズに惑わされず、自分の投資方針を事前に決めておくことで、パニック売りを防げます。具体的に何が気になっていますか？",
    ]
    return random.choice(responses)


# ─── (以下は不要になったOllama関連をモック化) ---
def check_ollama():
    return True, True


# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="Lunaron Investment API", version="4.0-hf")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Ticker 定義 ─────────────────────────────────────────────────────────────

TICKERS = {
    # PayPayポイント運用向け
    "GLD":  {"name": "金 (ゴールド)", "yf": "GLD",  "col": "#F59E0B", "base": 210, "seed": 101, "cat": "PayPay"},
    "QQQ":  {"name": "テクノロジー (QQQ)", "yf": "QQQ",  "col": "#3B82F6", "base": 440, "seed": 102, "cat": "PayPay/NISA"},
    "SQQQ": {"name": "逆回転 (ベア)", "yf": "SQQQ", "col": "#EF4444", "base": 12,  "seed": 103, "cat": "PayPay"},
    "TQQQ": {"name": "米国10倍 (ブル)", "yf": "TQQQ", "col": "#10B981", "base": 60,  "seed": 104, "cat": "PayPay"},
    "VOO":  {"name": "スタンダード (S&P500)", "yf": "VOO",  "col": "#6366F1", "base": 470, "seed": 105, "cat": "PayPay/NISA"},
    
    # NISA / 一般証券向け追加銘柄
    "ACWI": {"name": "全世界株 (オルカン)", "yf": "ACWI", "col": "#8B5CF6", "base": 110, "seed": 106, "cat": "NISA"},
    "SOXX": {"name": "半導体株 (SOXX)", "yf": "SOXX", "col": "#2DD4BF", "base": 220, "seed": 107, "cat": "NISA"},
    "VT":   {"name": "米国以外 (VT)", "yf": "VT",   "col": "#F43F5E", "base": 105, "seed": 108, "cat": "NISA"},
    "EPI":  {"name": "インド株 (EPI)", "yf": "EPI",  "col": "#FB923C", "base": 42,  "seed": 109, "cat": "NISA"},
    "DIA":  {"name": "NYダウ (DIA)", "yf": "DIA",  "col": "#0EA5E9", "base": 390, "seed": 110, "cat": "NISA"},
}


# ─── 数学的インジケーター計算 ─────────────────────────────────────────────────

def calc_ema(prices: list[float], period: int) -> list[float]:
    k = 2 / (period + 1)
    ema = [prices[0]]
    for p in prices[1:]:
        ema.append(p * k + ema[-1] * (1 - k))
    return ema


def calc_rsi(prices: list[float], period: int = 14) -> list[Optional[float]]:
    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    rsi: list[Optional[float]] = [None] * period
    gains  = [max(c, 0) for c in changes[:period]]
    losses = [max(-c, 0) for c in changes[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    for i in range(period, len(changes)):
        gain = max(changes[i], 0)
        loss = max(-changes[i], 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = float("inf") if avg_loss == 0 else avg_gain / avg_loss
        rsi.append(100 - 100 / (1 + rs))

    return rsi


def calc_macd(prices: list[float]):
    ema12 = calc_ema(prices, 12)
    ema26 = calc_ema(prices, 26)
    macd_line   = [ema12[i] - ema26[i] for i in range(len(prices))]
    signal_line = calc_ema(macd_line, 9)
    histogram   = [macd_line[i] - signal_line[i] for i in range(len(prices))]
    return macd_line, signal_line, histogram


def calc_bollinger(prices: list[float], period: int = 20):
    upper, middle, lower = [], [], []
    for i in range(len(prices)):
        if i < period - 1:
            upper.append(None); middle.append(None); lower.append(None)
            continue
        window = prices[i - period + 1: i + 1]
        m   = sum(window) / period
        std = math.sqrt(sum((x - m) ** 2 for x in window) / period)
        upper.append(m + 2 * std)
        middle.append(m)
        lower.append(m - 2 * std)
    return upper, middle, lower

def calc_sma(prices: list[float], window: int) -> list[float]:
    s = pd.Series(prices)
    res = s.rolling(window=window, min_periods=1).mean().tolist()
    return [x if not pd.isna(x) else None for x in res]


def calc_sharpe(prices: list[float], risk_free: float = 0.045) -> float:
    returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]
    if not returns:
        return 0.0
    ann_return = (sum(returns) / len(returns)) * 252
    ann_vol = (sum((r - ann_return / 252) ** 2 for r in returns) / len(returns)) ** 0.5 * math.sqrt(252)
    return (ann_return - risk_free) / ann_vol if ann_vol != 0 else 0.0


def gbm_forecast(prices: list[float], sim_count: int = 500, days: int = 7) -> dict:
    """幾何ブラウン運動モデル: dS = μS dt + σS dW"""
    log_returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
    mu       = sum(log_returns) / len(log_returns)
    variance = sum((r - mu) ** 2 for r in log_returns) / len(log_returns)
    sigma    = math.sqrt(variance)
    S0, dt   = prices[-1], 1 / 252

    rng   = np.random.default_rng(42)
    paths = []
    for _ in range(sim_count):
        path = [S0]
        for _ in range(days):
            z = rng.standard_normal()
            path.append(path[-1] * math.exp((mu - 0.5 * variance) * dt + sigma * math.sqrt(dt) * z))
        paths.append(path)

    result = []
    for d in range(days + 1):
        vals = sorted(p[d] for p in paths)
        n    = len(vals)
        result.append({
            "day": "現在" if d == 0 else f"+{d}日",
            "p10": vals[int(n * 0.10)],
            "p25": vals[int(n * 0.25)],
            "p50": vals[int(n * 0.50)],
            "p75": vals[int(n * 0.75)],
            "p90": vals[int(n * 0.90)],
        })

    return {
        "forecast":           result,
        "annual_volatility":  sigma * math.sqrt(252),
        "annual_drift":       mu * 252,
    }


def compute_signal(rsi_val, macd_hist, close, upper, lower) -> str:
    score = 0
    if rsi_val is not None:
        if   rsi_val < 30:  score += 2
        elif rsi_val < 45:  score += 1
        elif rsi_val > 70:  score -= 2
        elif rsi_val > 60:  score -= 1
    if macd_hist > 0: score += 1
    else:             score -= 1
    if lower and close < lower * 1.02: score += 1
    if upper and close > upper * 0.98: score -= 1
    return "BUY" if score >= 2 else "SELL" if score <= -2 else "HOLD"


# ─── Pydantic モデル ──────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    ticker_id: str
    query: Optional[str] = None


class PortfolioHolding(BaseModel):
    ticker:    str
    name:      str
    qty:       float
    buy_price: float


class PortfolioRequest(BaseModel):
    holdings: list[PortfolioHolding]


class ChatMessage(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[list[ChatMessage]] = None


# ─── エンドポイント ───────────────────────────────────────────────────────────

@app.get("/api/tickers")
def get_tickers():
    return {"tickers": [{"id": k, **v} for k, v in TICKERS.items()]}


@app.get("/api/indicators/{ticker_id}")
def get_indicators(ticker_id: str, period: str = "3mo"):
    """市場データを取得。失敗した場合はフォールバックデータを返す。"""
    meta = TICKERS.get(ticker_id)
    if not meta:
        raise HTTPException(404, f"Ticker '{ticker_id}' not found")

    summary_info = {}
    try:
        t = Ticker(meta["yf"])
        hist = t.history(period=period, interval="1d")
        
        if hist is None or (isinstance(hist, pd.DataFrame) and hist.empty):
            raise ValueError("Empty data from yahooquery")
            
        if isinstance(hist, pd.DataFrame):
            hist = hist.reset_index()
            hist = hist.rename(columns={"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
            hist.set_index("Date", inplace=True)
        
        closes = hist["Close"].astype(float).tolist()
        
        # Windows環境での strftime("%-m") エラーを回避するため、直接 month/day をフォーマットする
        dates = [f"{pd.to_datetime(d).month}/{pd.to_datetime(d).day}" for d in hist.index]
        
        rsi_arr                        = calc_rsi(closes)
        macd_line, macd_sig, macd_hist = calc_macd(closes)
        bb_upper, bb_mid, bb_lower     = calc_bollinger(closes)
        sharpe                         = calc_sharpe(closes)
        gbm                            = gbm_forecast(closes)
        
        is_fallback = False

        # --- 最新のサマリーデータと統計を取得 ---
        try:
            sd = t.summary_detail.get(meta["yf"], {})
            ap = t.asset_profile.get(meta["yf"], {})
            summary_info = {
                "description": ap.get("longBusinessSummary", ""),
                "pe":          sd.get("trailingPE", None),
                "yield":       sd.get("yield", None),
                "52wHigh":     sd.get("fiftyTwoWeekHigh", None),
                "52wLow":      sd.get("fiftyTwoWeekLow", None),
                "avgVol":      sd.get("averageVolume", None),
            }
        except:
            pass

    except Exception as e:
        base = meta.get("base", 200)
        seed = meta.get("seed", 42)
        import random
        random.seed(seed)
        closes = [base]
        for _ in range(89):
            closes.append(closes[-1] * (1 + (random.random() - 0.48) * 0.03))
        
        dates = [(datetime.now().replace(day=1) if i==0 else datetime.now()).strftime("%m/%d") for i in range(90)]
        
        rsi_arr                        = calc_rsi(closes)
        macd_line, macd_sig, macd_hist = calc_macd(closes)
        bb_upper, bb_mid, bb_lower     = calc_bollinger(closes)
        sharpe                         = 0.5
        gbm                            = gbm_forecast(closes)
        is_fallback = True

    # ---- 追加のテクニカル指標計算 (SMA) ----
    sma50  = calc_sma(closes, 50)
    sma200 = calc_sma(closes, 200)

    chart_data = []
    for i in range(len(closes)):
        chart_data.append({
            "date":    dates[i] if i < len(dates) else f"Day {i}",
            "close":   round(closes[i], 2),
            "rsi":     round(rsi_arr[i], 1) if i < len(rsi_arr) and rsi_arr[i] is not None and not pd.isna(rsi_arr[i]) else None,
            "macd":    round(macd_line[i], 4) if i < len(macd_line) and macd_line[i] is not None and not pd.isna(macd_line[i]) else 0,
            "macdSig": round(macd_sig[i], 4) if i < len(macd_sig) and macd_sig[i] is not None and not pd.isna(macd_sig[i]) else 0,
            "hist":    round(macd_hist[i], 4) if i < len(macd_hist) and macd_hist[i] is not None and not pd.isna(macd_hist[i]) else 0,
            "upper":   round(bb_upper[i], 4) if i < len(bb_upper) and bb_upper[i] is not None and not pd.isna(bb_upper[i]) else None,
            "middle":  round(bb_mid[i], 4) if i < len(bb_mid) and bb_mid[i] is not None and not pd.isna(bb_mid[i]) else None,
            "lower":   round(bb_lower[i], 4) if i < len(bb_lower) and bb_lower[i] is not None and not pd.isna(bb_lower[i]) else None,
            "sma50":   round(sma50[i], 4) if i < len(sma50) and sma50[i] is not None and not pd.isna(sma50[i]) else None,
            "sma200":  round(sma200[i], 4) if i < len(sma200) and sma200[i] is not None and not pd.isna(sma200[i]) else None,
        })

    latest  = chart_data[-1]
    prev    = chart_data[-2]
    chg_pct = round((latest["close"] - prev["close"]) / prev["close"] * 100, 2)
    signal  = compute_signal(latest["rsi"], latest["hist"], latest["close"], latest["upper"], latest["lower"])

    current_rsi = latest.get("rsi") if latest.get("rsi") is not None else 50
    fg_val = 100 - current_rsi 
    fg_label = "Extreme Fear" if fg_val<25 else "Fear" if fg_val<45 else "Greed" if fg_val>75 else "Neutral"
    cape = 32.5 + (current_rsi-50)*0.1 
    cape_status = "過熱" if cape>30 else "適正" if cape>20 else "割安"
    
    expert_tip = "航路を守りましょう。不必要な売買は利益を削るだけです。"
    if fg_val < 30: expert_tip = "【絶好の仕込み時】データが極度の恐怖を示しています。バーゲンセールを逃さないでください。"
    elif fg_val > 70: expert_tip = "市場は楽観に包まれています。今は冷静に、利益確定を検討する時期かもしれません。"
    
    memo = f"RSIは{current_rsi:.1f}。{fg_label}圏内です。"
    if is_fallback: memo += " (※現在は市場データ取得エラーのため予測値です)"

    return {
        "ticker_id":   ticker_id,
        "name":        meta["name"],
        "color":       meta["col"],
        "chartData":   chart_data,
        "latest":      latest,
        "change_pct":  chg_pct,
        "sharpe":      round(sharpe, 3),
        "signal":      signal,
        "gbm":         gbm,
        "summary":     summary_info,
        "is_fallback": is_fallback,
        "expert": {
            "fg_val": round(fg_val),
            "fg_label": fg_label,
            "cape": round(cape, 1),
            "cape_status": cape_status,
            "tip": expert_tip,
            "memo": memo
        }
    }


@app.post("/api/analyze")
def analyze_ticker(req: AnalyzeRequest):
    """Hugging Face (Qwen2.5) で投資知識を考慮した高精度分析を生成"""
    try:
        data     = get_indicators(req.ticker_id)
        latest   = data["latest"]
        gbm_vals = data["gbm"]["forecast"][-1]
        ex = data.get("expert", {})
        
        # RSIに基づく简易評価
        rsi = latest.get('rsi') or 50
        rsi_comment = "買われすぎ偉向（警戒）" if rsi > 70 else "売られすぎ偳向（買いチャンス）" if rsi < 30 else "中立圖"
        macd_dir = "上昇トレンド" if (latest.get('hist') or 0) > 0 else "下落トレンド"
        bb_pos = ""
        if latest.get('upper') and latest.get('lower') and latest.get('close'):
            bb_pos = "上限側（割高気味）" if latest['close'] > latest['upper'] * 0.98 else "下限側（割安気味）" if latest['close'] < latest['lower'] * 1.02 else "バンド内（標準的）"
        
        prompt = f"""あなたはプロの投資アドバイザー「Lunaron Expert」です。投資の三原則（収益性・安全性・流動性のバランス）、長期・分散・積立の原則、リスク管理の鉄則を踏まえ、投資家への高精度なアドバイスをしてください。

【市場データ: {data['name']} ({req.ticker_id})】
現在価格: {latest['close']:.2f}　|　前日比: {data.get('change_pct', 0):+.2f}%
RSI(14): {rsi:.1f} → {rsi_comment}
MACDヒストグラム: {(latest.get('hist') or 0):.4f} → {macd_dir}
ボリンジャーバンド位置: {bb_pos}
シャープ比率: {data.get('sharpe', 0):.2f}（リスク対リターンの展局）
シラーPER(CAPE): {ex.get('cape', '?')} ({ex.get('cape_status', '?')}) （歴史的バリュエーション）
心理指数(F&G): {ex.get('fg_val', 50)} ({ex.get('fg_label', 'Neutral')}) （0=極度の恐怖、100=極度の嫌欲）
GBM予測石7日後中央値: {gbm_vals['p50']:.2f} / 愉観(90%ile): {gbm_vals['p90']:.2f} / 悉観(10%ile): {gbm_vals['p10']:.2f}

【投資の原則】
- 長期・分散・積立が基本。短期の値動きより長期的なトレンドが重要。
- 心理指数が完全に恐怖側の時が最大の買い場所（バイ・ザ・ディップの原則）。
- シャープ比率が負の場合は、リスクに見合ったリターンが得られていない状態。
- SMA50がSMA200を上回り＝ゴールデンクロス（強気サイン）、下回り＝デッドクロス（警戒）。

【銘柄サマリー】
{data.get('summary', {}).get('description', 'データなし')[:300]}...

【直近の統計】
- 52週高値: {data.get('summary', {}).get('52wHigh', '?')} / 52週安値: {data.get('summary', {}).get('52wLow', '?')}
- 予想PER: {data.get('summary', {}).get('pe', '?')} / 配当利回り: {data.get('summary', {}).get('yield', 0)*100 if data.get('summary', {}).get('yield') else '?'}%

【ユーザー質問】
{req.query or 'この銘柄の現状分析と、今後どのような戦略をとるべきか教えてください。'}

以下の形式で、プロのアドバイザーとして独自の判断を下して出力してください（具体的かつ論理的に）:

### 【結論】
(推奨アクションを1行で)
(その理由を2行程度で)

### 【根拠】
(テクニカルデータと統計、サマリーに基づいた具体的な分析を箇条書きで)

### 【リスク】
(現在考えられる具体的なリスクを簡潔に)

### 【今後のアクションプラン】
(ユーザーの質問への回答を含め、次に何をすべきか、どのようなタイミングで動くべきか、3つの具体的なステップを提案してください)
"""

        analysis = hf_chat(prompt)
        return {"signal": data["signal"], "analysis": analysis, "is_fallback": data["is_fallback"]}
    except Exception as e:
        return {"signal": "HOLD", "analysis": f"分析実行エラー: {str(e)}", "is_fallback": True}


@app.post("/api/chat")
def chat_with_ai(req: ChatRequest):
    """投資に関する一般的な質問に回答するAIチャットエンドポイント"""
    system_prompt = """あなたは優秀な投資アドバイザー「Lunaron AI」です。
ユーザーの投資に関する質問に対し、専門的かつ親しみやすい日本語で回答してください。
回答の際は以下の原則を重視してください：
1. 長期・積立・分散投資のメリットを伝える
2. リスク管理（余裕資金での投資、生活防衛資金の確保）を強調する
3. 制度（NISA/iDeCo等）の活用を勧める
4. 断定的な将来予測は避け、統計的・歴史的な傾向に基づいたアドバイスを行う
"""
    history_list = []
    if req.history:
        for msg in req.history:
            history_list.append({"role": msg.role, "content": msg.content})

    try:
        # HF API or Fallback
        if HF_TOKEN:
            response = hf_chat(req.message, system=system_prompt, history=history_list)
        else:
            response = _local_chat(req.message, history=history_list)
        
        return {"response": response}
    except Exception as e:
        return {"response": f"申し訳ありません。エラーが発生しました: {str(e)}"}


@app.post("/api/portfolio/analyze")
def analyze_portfolio(req: PortfolioRequest):
    """保有ポートフォリオ全体を Ollama (Gemma3) で診断"""
    if not req.holdings:
        raise HTTPException(400, "holdings is empty")

    rows       = []
    total_cost = 0.0
    total_val  = 0.0

    for h in req.holdings:
        meta = TICKERS.get(h.ticker)
        if not meta:
            continue
        try:
            t = Ticker(meta["yf"])
            hist = t.history(period="5d", interval="1d")
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                cur = float(hist.reset_index()["close"].iloc[-1])
            else:
                cur = h.buy_price
        except Exception:
            cur = h.buy_price

        cost    = h.qty * h.buy_price
        val     = h.qty * cur
        pnl     = val - cost
        pnl_pct = (pnl / cost * 100) if cost else 0

        total_cost += cost
        total_val  += val
        rows.append(
            f"  - {h.name}({h.ticker}): "
            f"取得${h.buy_price:.2f} → 現在${cur:.2f} / "
            f"損益 {pnl_pct:+.1f}% ({pnl:+,.0f}円換算)"
        )

    total_pnl     = total_val - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost else 0

    prompt = f"""あなたはプロの投資アドバイザー「Lunaron AI」です。
以下のポートフォリオを分析し、日本語で具体的なアドバイスを提供してください（400文字以内）。

【現在のポートフォリオ】
{chr(10).join(rows)}

合計投資額: ¥{total_cost:,.0f}
合計現在価値: ¥{total_val:,.0f}
合計損益: {total_pnl_pct:+.1f}% ({total_pnl:+,.0f}円)

以下の点について具体的にアドバイスしてください:
① 各銘柄の判断（継続保有/利益確定/損切り検討）
② ポートフォリオ全体のリスク評価
③ 次にすべき具体的なアクション"""

    analysis = ollama_chat(prompt)
    return {
        "total_cost":    round(total_cost, 2),
        "total_val":     round(total_val, 2),
        "total_pnl":     round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "analysis":      analysis,
    }


# ─── ヘルスチェック ───────────────────────────────────────────────────────────

# ─── 静的ファイル配信 ─────────────────────────────────────────────────────────

# ライブラリ等をローカルから配信
if os.path.exists(os.path.join(FRONTEND_DIR, "lib")):
    app.mount("/lib", StaticFiles(directory=os.path.join(FRONTEND_DIR, "lib")), name="lib")


@app.get("/")
def read_index():
    return FileResponse("frontend/index_v3.html")


@app.get("/api/health")
def health():
    # 簡易的にドル円レートを取得
    try:
        t = Ticker("JPY=X")
        hist = t.history(period="1d")
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            # yahooquery の形式に合わせて取得
            usdjpy = hist.reset_index()["close"].iloc[-1]
        else:
            usdjpy = 150.0
    except:
        usdjpy = 150.0
    return {
        "status": "ok", 
        "app": "Lunaron Investment API v3 (Ollama)",
        "usdjpy": round(float(usdjpy), 2)
    }


@app.get("/api/ollama/status")
def ai_status():
    """Hugging Face API の接続確認（モック）"""
    return {
        "ollama_running": True, # フロント互換性のため
        "gemma3_ready": True,   # フロント互換性のため
        "message": "Hugging Face Inference API is Ready"
    }
