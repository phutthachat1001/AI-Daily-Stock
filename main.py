# main.py
# -*- coding: utf-8 -*-
import os
import re
import json
import math
import yaml
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import yfinance as yf
import feedparser
import google.generativeai as genai

# ------------------ Utils ------------------
def load_config(path="config.yml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def fmt_pct(x, digits=2):
    if pd.isna(x):
        return "-"
    return f"{x*100:.{digits}f}%"

def fmt_price(x, digits=2):
    if pd.isna(x):
        return "-"
    return f"{x:.{digits}f}"

def epoch_to_iso(ts):
    try:
        return datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""

# ------------------ Indicators ------------------
def sma(series, window=20):
    return series.rolling(window).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def pct_change(series, periods=1):
    return series.pct_change(periods=periods)

# ------------------ Data Fetch ------------------
def fetch_history(ticker, lookback_days=260):
    df = yf.download(ticker, period="400d", interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.tail(lookback_days)
    return df

# ------------------ Company names (for news queries) ------------------
COMPANY_NAME = {
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "ALAB": "Astera Labs",
    "PLTR": "Palantir",
    "TSM": "Taiwan Semiconductor",
    "AMD": "Advanced Micro Devices",
    "RKLB": "Rocket Lab",
}

# ------------------ News fetchers ------------------
def fetch_news_google(query, days=2, max_items=3, lang="en-US", country="US"):
    """
    Google News RSS (ผ่าน feedparser)
    ตัวอย่าง query: "Tesla TSLA stock when:2d"
    """
    try:
        q = f"{query} stock OR shares when:{days}d"
        url = f"https://news.google.com/rss/search?q={quote_plus(q)}&hl={lang}&gl={country}&ceid={country}:{lang.split('-')[-1]}"
        feed = feedparser.parse(url)
        items = []
        for e in feed.entries[: max_items]:
            items.append({
                "title": e.get("title", "").strip(),
                "link": e.get("link", "").strip(),
                "published": e.get("published", ""),
                "source": (getattr(e, "source", {}).get("title") if hasattr(e, "source") else "") or e.get("author", "")
            })
        return items
    except Exception:
        return []

def fetch_news_yfinance(ticker, days=2, max_items=3):
    """
    Fallback: ข่าวจาก yfinance.Ticker(t).news (ถ้ามี)
    """
    try:
        news = yf.Ticker(ticker).news
        if not news:
            return []
        cutoff = time.time() - days*86400
        items = []
        for n in news:
            tpub = n.get("providerPublishTime")
            if tpub and tpub >= cutoff:
                items.append({
                    "title": n.get("title",""),
                    "link": n.get("link",""),
                    "published": epoch_to_iso(tpub),
                    "source": n.get("publisher","")
                })
            if len(items) >= max_items:
                break
        return items[:max_items]
    except Exception:
        return []

def fetch_news_for_ticker(ticker, company, cfg_news):
    if not cfg_news.get("enable", True):
        return []
    days = int(cfg_news.get("lookback_days", 2))
    per_ticker = int(cfg_news.get("per_ticker", 3))

    # 1) Google News first
    items = fetch_news_google(f"{company} {ticker}", days=days, max_items=per_ticker)
    # 2) Fallback to Yahoo news
    if len(items) < per_ticker:
        extra = fetch_news_yfinance(ticker, days=days, max_items=(per_ticker - len(items)))
        items.extend(extra)
    return items[:per_ticker]

# ------------------ Feature Engineering ------------------
def build_features(ticker, df):
    if df.empty or len(df) < 50:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    out = {}
    out["ticker"] = ticker
    out["last_date"] = df.index[-1].strftime("%Y-%m-%d")
    out["price"] = float(close.iloc[-1])

    out["sma20"] = float(sma(close, 20).iloc[-1])
    out["sma50"] = float(sma(close, 50).iloc[-1])
    out["sma200"] = float(sma(close, 200).iloc[-1]) if len(df) >= 200 else np.nan

    rsi_series = rsi(close, 14)
    out["rsi14"] = float(rsi_series.iloc[-1])

    macd_line, signal_line, hist = macd(close, 12, 26, 9)
    out["macd"] = float(macd_line.iloc[-1])
    out["macd_signal"] = float(signal_line.iloc[-1])
    out["macd_hist"] = float(hist.iloc[-1])

    out["chg_1d"] = float(pct_change(close, 1).iloc[-1])
    out["chg_5d"] = float(pct_change(close, 5).iloc[-1])
    out["chg_20d"] = float(pct_change(close, 20).iloc[-1])

    lookback_252 = close.tail(252) if len(close) >= 252 else close
    out["high_52w"] = float(lookback_252.max())
    out["low_52w"] = float(lookback_252.min())
    out["off_high_52w_pct"] = float((out["price"] / out["high_52w"]) - 1.0) if out["high_52w"] else np.nan
    out["above_low_52w_pct"] = float((out["price"] / out["low_52w"]) - 1.0) if out["low_52w"] else np.nan

    def trend_label(x):
        return "Uptrend" if x else "Down/Sideways"

    out["trend_sma"] = trend_label(out["price"] > out["sma50"] > (out["sma200"] if not math.isnan(out["sma200"]) else -1e9))
    out["rsi_state"] = "Overbought" if out["rsi14"] >= 70 else ("Oversold" if out["rsi14"] <= 30 else "Neutral")
    out["macd_state"] = "Bullish" if out["macd"] > out["macd_signal"] else "Bearish"

    return out

# ------------------ Gemini Prompt ------------------
def build_ai_prompt(config, market_overview, features_list, news_map):
    tz = config.get("timezone", "Asia/Bangkok")
    today = datetime.now(ZoneInfo(tz)).strftime("%Y-%m-%d")
    risk = config.get("risk_management", {})
    dsl = risk.get("default_stop_loss_pct", 0.03)
    dtp = risk.get("default_take_profit_pct", 0.06)

    sys = f"""
You are an equity analyst. Based on technical metrics and provided headlines, output STRICT JSON only.
Schema:
{{
  "date": "YYYY-MM-DD",
  "tickers": [
    {{
      "ticker": "TSLA",
      "stance": "Buy|Sell|Hold",
      "confidence": 0-100,
      "entry_rule": "text",
      "entry_price_range": "e.g., 240-245 or '-'",
      "stop_loss": "price or percent string",
      "take_profit": "price or percent string",
      "timeframe": "days|weeks",
      "reasoning_bullets": ["…", "…"],
      "drop_news_summary": "ถ้าราคาติดลบ 1d ให้สรุป 2-3 bullet ว่า ‘ข่าว/ปัจจัย’ ใดน่าจะทำให้ราคาลดลง โดยอ้างอิงหัวข้อข่าวที่ให้มา; ถ้าไม่พบให้ใส่ '-'",
      "news_refs": ["หัวข้อข่าวที่ใช้อ้างอิง", "..."]
    }}
  ],
  "notes": "optional"
}}

Rules:
- Decide ONLY from given metrics/news. Avoid hallucination.
- Thai language for bullets/summary. Keep concise and concrete.
- If 1d change >= 0 → set drop_news_summary to "-".
- Output pure JSON (no Markdown).
- Reasoning uses indicators e.g., SMA/RSI/MACD, 52w levels, trend.
- If uncertain SL/TP, use defaults: SL {dsl:.2%}, TP {dtp:.2%}.
"""

    # market overview lines (compact)
    overview_lines = []
    for k, arr in market_overview.items():
        if not arr:
            continue
        for item in arr:
            overview_lines.append(
                f"{k}:{item['ticker']} p={fmt_price(item['price'])} 1d={fmt_pct(item['chg_1d'])} rsi={item['rsi14']:.1f} trend={item['trend_sma']}"
            )

    # features lines
    feat_lines = []
    for f in features_list:
        if not f:
            continue
        feat_lines.append(
            f"{f['ticker']} price={fmt_price(f['price'])} 1d={fmt_pct(f['chg_1d'])} 5d={fmt_pct(f['chg_5d'])} 20d={fmt_pct(f['chg_20d'])} "
            f"SMA20/50/200={fmt_price(f['sma20'])}/{fmt_price(f['sma50'])}/{fmt_price(f['sma200'])} "
            f"RSI14={f['rsi14']:.1f}({f['rsi_state']}) MACD={f['macd']:.3f}/{f['macd_signal']:.3f}({f['macd_state']}) "
            f"52wH/L={fmt_price(f['high_52w'])}/{fmt_price(f['low_52w'])} offHigh={fmt_pct(f['off_high_52w_pct'])}"
        )

    # news section: include only titles + links compactly to save tokens
    news_lines = []
    for t, items in news_map.items():
        if not items:
            continue
        for i, it in enumerate(items, 1):
            title = it.get("title", "")
            link = it.get("link", "")
            news_lines.append(f"{t} NEWS{i}: {title} | {link}")

    user = f"""
วันที่: {today}

ภาพรวมตลาด (ย่อ):
- {'; '.join(overview_lines) if overview_lines else '-'}

หุ้นที่จะวิเคราะห์:
- {'\n- '.join(feat_lines) if feat_lines else '-'}

หัวข้อข่าวต่อหุ้น (ใช้เป็นบริบทสำหรับ 'drop_news_summary'):
- {'\n- '.join(news_lines) if news_lines else '-'}

งานของคุณ:
- ให้คำแนะนำ Buy/Sell/Hold + แผนเข้า-ออก
- และสำหรับหุ้นที่ 1d<0 ให้สรุปข่าว/ปัจจัยที่น่าจะทำให้ราคาลดลง (อ้างอิงหัวข้อข่าวด้านบน)
- ส่งออกเป็น JSON ตามสคีมา (ห้ามมีข้อความอื่น)
""".strip()

    return sys.strip(), user.strip()

def call_gemini(model_name, system_prompt, user_prompt):
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY")
    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content([{"role":"user","parts":[system_prompt + "\n\n" + user_prompt]}])
    text = (resp.text or "").strip().strip("`")
    # Try to extract JSON if model wrapped anything
    try:
        return json.loads(text)
    except Exception:
        # fallback: find first {...} block
        m = re.search(r"\{.*\}\s*$", text, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise

# ------------------ Report ------------------
def render_report(date_str, overview, features, ai_json, news_map):
    rec_map = {t["ticker"]: t for t in ai_json.get("tickers", [])}

    md = []
    md.append(f"# Daily AI Stock Insight — {date_str}\n")
    md.append("> *รายงานอัตโนมัติจาก GitHub Actions + Gemini — เพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน*\n")

    # Market overview
    md.append("## ภาพรวมตลาด\n")
    def render_group(name, arr):
        if not arr:
            return
        md.append(f"**{name}**")
        for it in arr:
            md.append(f"- `{it['ticker']}` ราคา {fmt_price(it['price'])} | 1d {fmt_pct(it['chg_1d'])} | RSI {it['rsi14']:.1f} | แนวโน้ม {it['trend_sma']}")
    render_group("ดัชนี", overview.get("indices"))
    render_group("สินค้าโภคภัณฑ์", overview.get("commodities"))
    render_group("อัตราแลกเปลี่ยน", overview.get("fx"))
    md.append("")

    # Summary table
    md.append("## สรุปสัญญาณต่อหุ้น\n")
    md.append("| Ticker | Price | 1d | Trend | RSI14 | Stance | Confidence |")
    md.append("|---|---:|---:|---|---:|---|---:|")
    for f in features:
        r = rec_map.get(f["ticker"], {})
        md.append(f"| `{f['ticker']}` | {fmt_price(f['price'])} | {fmt_pct(f['chg_1d'])} | {f['trend_sma']} | {f['rsi14']:.1f} | {r.get('stance','-')} | {r.get('confidence','-')} |")
    md.append("")

    # Details per ticker
    md.append("## แผนการเข้า-ออกต่อหุ้น (รายละเอียด)\n")
    for f in features:
        r = rec_map.get(f["ticker"], {})
        md.append(f"### {f['ticker']}")
        md.append(f"- ราคา: {fmt_price(f['price'])} | 1d {fmt_pct(f['chg_1d'])} | 5d {fmt_pct(f['chg_5d'])} | 20d {fmt_pct(f['chg_20d'])}")
        md.append(f"- SMA20/50/200: {fmt_price(f['sma20'])} / {fmt_price(f['sma50'])} / {fmt_price(f['sma200'])}")
        md.append(f"- RSI14: {f['rsi14']:.1f} ({f['rsi_state']}) | MACD: {f['macd']:.3f}/{f['macd_signal']:.3f} ({f['macd_state']})")
        md.append(f"- 52w: H {fmt_price(f['high_52w'])} / L {fmt_price(f['low_52w'])} | ห่าง 52wH: {fmt_pct(f['off_high_52w_pct'])}")
        md.append(f"- **ข้อแนะนำ (Gemini):** {r.get('stance','-')} | ความเชื่อมั่น: {r.get('confidence','-')}")
        md.append(f"  - Entry: {r.get('entry_rule','-')} | ช่วงราคาเข้า: {r.get('entry_price_range','-')}")
        md.append(f"  - Stop Loss: {r.get('stop_loss','-')} | Take Profit: {r.get('take_profit','-')} | Timeframe: {r.get('timeframe','-')}")
        bullets = r.get("reasoning_bullets", [])
        if bullets:
            md.append("  - เหตุผล:")
            for b in bullets:
                md.append(f"    - {b}")

        # News section (only show if price fell or there are news)
        drop_summary = r.get("drop_news_summary", "-")
        if drop_summary and drop_summary != "-":
            md.append("\n**ข่าว/ปัจจัยที่น่าจะทำให้ราคาลดลง (สรุป):**")
            for line in drop_summary.split("\n"):
                md.append(f"- {line.strip()}")

        # show raw headlines with links
        items = news_map.get(f["ticker"], [])
        if items:
            md.append("\n**หัวข้อข่าวล่าสุด:**")
            for it in items:
                title = it.get("title","").strip()
                link = it.get("link","").strip()
                src = it.get("source","")
                pub = it.get("published","")
                md.append(f"- [{title}]({link}) — {src} ({pub})")
        md.append("")

    notes = ai_json.get("notes","")
    if notes:
        md.append("## หมายเหตุจากโมเดล\n")
        md.append(notes)
        md.append("")

    md.append("---")
    md.append("**หมายเหตุ/ข้อจำกัด:** รายงานนี้สร้างโดยอัลกอริทึมและ LLM เพื่อการศึกษาเท่านั้น มิใช่คำแนะนำการลงทุน ความผิดพลาดอาจเกิดขึ้นได้ ควรตรวจสอบข้อมูลก่อนตัดสินใจ")
    return "\n".join(md)

# ------------------ Overview helper ------------------
def to_overview_block(tickers, lookback_days):
    arr = []
    for t in tickers:
        df = fetch_history(t, lookback_days)
        if df.empty: 
            continue
        f = build_features(t, df)
        if f:
            arr.append(f)
    return arr

# ------------------ Main ------------------
def main():
    config = load_config("config.yml")
    tz = config.get("timezone", "Asia/Bangkok")
    now = datetime.now(ZoneInfo(tz))

    if config.get("skip_if_weekend", True) and now.weekday() >= 5:
        print("Weekend — skipped.")
        return

    lookback = int(config.get("lookback_days", 260))

    tickers = config.get("tickers", [])
    indices = config.get("indices", [])
    commodities = config.get("commodities", [])
    fx = config.get("fx", [])
    cfg_news = config.get("news", {"enable": True, "lookback_days": 2, "per_ticker": 3})

    # Market overview
    overview = {
        "indices": to_overview_block(indices, lookback),
        "commodities": to_overview_block(commodities, lookback),
        "fx": to_overview_block(fx, lookback),
    }

    # Stock features
    features = []
    for t in tickers:
        df = fetch_history(t, lookback)
        f = build_features(t, df)
        if f:
            features.append(f)

    if not features:
        raise RuntimeError("No feature data built; check tickers or network")

    # Fetch news per ticker (compact)
    news_map = {}
    if cfg_news.get("enable", True):
        for f in features:
            t = f["ticker"]
            company = COMPANY_NAME.get(t, t)
            news_map[t] = fetch_news_for_ticker(t, company, cfg_news)

    # Build AI prompt and call Gemini (1 batch call)
    sys, usr = build_ai_prompt(config, overview, features, news_map)
    model_name = config.get("gemini_model", "gemini-2.5-flash")
    ai_json = call_gemini(model_name, sys, usr)

    report_date = now.strftime("%Y-%m-%d")
    md = render_report(report_date, overview, features, ai_json, news_map)

    ensure_dir("reports")
    with open(f"reports/{report_date}.md", "w", encoding="utf-8") as f:
        f.write(md)
    with open("reports/latest.md", "w", encoding="utf-8") as f:
        f.write(md)

    print("Report generated:", report_date)

if __name__ == "__main__":
    main()

