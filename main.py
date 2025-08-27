import os
import re
import json
import math
import time
import yaml
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests  # เผื่อเช็คเน็ต/ใช้ภายหลัง

# ---- Matplotlib (headless สำหรับ GitHub Actions) ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
except Exception:
    gspread = None
    ServiceAccountCredentials = None

# ---- Gemini ----
try:
    import google.generativeai as genai
except Exception:
    genai = None


# ======================= Config / Utils =======================
def load_config(path="config.yml"):
    default = {
        "timezone": "Asia/Bangkok",
        "skip_if_weekend": True,
        "lookback_days": 260,
        "gemini_model": "gemini-2.5-flash",
        "news": {"enable": True, "lookback_days": 2, "per_ticker": 3},
        "risk_management": {"default_stop_loss_pct": 0.03, "default_take_profit_pct": 0.06},
        "charts": {"enable": True},
        "tickers": ["TSLA","NVDA","AAPL","MSFT","AMZN","ALAB","PLTR","TSM","AMD","RKLB"],
        "indices": ["SPY","QQQ"],
        "commodities": ["USO","BNO"],
        "fx": ["UUP","FXE"],
        "per_call_delay_sec": 0.2,
        "google_news_locale": {"hl": "en-US", "gl": "US", "ceid": "US:en"}
    }
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        for k, v in default.items():
            if k not in user:
                user[k] = v
        return user
    return default

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

def to_scalar(x):
    """
    ดึงค่า scalar อย่างปลอดภัย: รองรับ numpy scalar / pandas Series 1 element / python float
    ตัด FutureWarning: 'Calling float on a single element Series'
    """
    try:
        # pandas / numpy scalar มักจะมี .item()
        return x.item()
    except AttributeError:
        try:
            return float(x)
        except Exception:
            return np.nan
    except Exception:
        try:
            return float(x)
        except Exception:
            return np.nan


# ======================= Technical Indicators =======================
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


# ======================= Yahoo Finance =======================
def yahoo_candles(symbol: str, lookback_days=400):
    """
    ดึงแท่งเทียนจาก Yahoo Finance ผ่าน yfinance
    คืน DataFrame: Open, High, Low, Close, Volume ; index=datetime
    """
    try:
        import yfinance as yf
    except Exception as e:
        print("[Yahoo] yfinance not installed:", e)
        return pd.DataFrame()

    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=int(lookback_days * 1.4))  # เผื่อวันหยุด
        df = yf.download(
            symbol,
            start=start.date().isoformat(),
            end=end.date().isoformat(),
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=True,
        )
        if df is None or df.empty:
            print(f"[Yahoo] no data for {symbol}")
            return pd.DataFrame()
        # normalize columns
        df = df.rename(columns={
            "Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"
        })
        df.index = pd.to_datetime(df.index)
        df = df[["Open","High","Low","Close","Volume"]].dropna(how="any")
        return df.sort_index()
    except Exception as e:
        print(f"[Yahoo] error {symbol}: {e}")
        return pd.DataFrame()


# ======================= Google News RSS =======================
def google_news_company(query_text: str, lookback_days=2, limit=3, locale=None):
    """ดึงข่าวด้วย Google News RSS โดยใช้คำค้น (บริษัท/ติ๊กเกอร์)"""
    try:
        import feedparser
    except Exception as e:
        print("[News] feedparser not installed:", e)
        return []

    if not query_text:
        return []

    locale = locale or {"hl": "en-US", "gl": "US", "ceid": "US:en"}
    q = quote_plus(f'{query_text} when:{lookback_days}d')
    url = f"https://news.google.com/rss/search?q={q}&hl={locale['hl']}&gl={locale['gl']}&ceid={locale['ceid']}"

    try:
        feed = feedparser.parse(url)
        items = []
        for e in feed.entries[:limit]:
            title = getattr(e, "title", "").strip()
            link  = getattr(e, "link", "").strip()
            pub   = getattr(e, "published", "")
            src   = getattr(getattr(e, "source", {}), "title", "") or "Google News"
            items.append({"title": title, "link": link, "published": pub, "source": src})
        return items
    except Exception as e:
        print(f"[News] error for query={query_text}: {e}")
        return []


# ======================= Company Names (ข่าว) =======================
COMPANY_NAME = {
    "TSLA": "Tesla", "NVDA": "Nvidia", "AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon",
    "ALAB": "Astera Labs", "PLTR": "Palantir", "TSM": "Taiwan Semiconductor",
    "AMD": "Advanced Micro Devices", "RKLB": "Rocket Lab",
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq-100 ETF",
    "USO": "United States Oil Fund", "BNO": "United States Brent Oil Fund",
    "UUP": "Invesco US Dollar Index", "FXE": "Invesco Euro Currency Trust",
}


# ======================= Feature Engineering =======================
def build_features(ticker, df):
    if df.empty or len(df) < 50:
        return None

    close = df["Close"]
    out = {"ticker": ticker}
    out["last_date"] = df.index[-1].strftime("%Y-%m-%d")
    out["price"] = to_scalar(close.iloc[-1])

    s20 = sma(close, 20)
    s50 = sma(close, 50)
    s200 = sma(close, 200) if len(df) >= 200 else pd.Series(dtype=float)

    out["sma20"] = to_scalar(s20.iloc[-1])
    out["sma50"] = to_scalar(s50.iloc[-1])
    out["sma200"] = to_scalar(s200.iloc[-1]) if len(df) >= 200 else np.nan

    rsi_series = rsi(close, 14)
    out["rsi14"] = to_scalar(rsi_series.iloc[-1])

    macd_line, signal_line, hist = macd(close, 12, 26, 9)
    out["macd"] = to_scalar(macd_line.iloc[-1])
    out["macd_signal"] = to_scalar(signal_line.iloc[-1])
    out["macd_hist"] = to_scalar(hist.iloc[-1])

    out["chg_1d"] = to_scalar(pct_change(close, 1).iloc[-1])
    out["chg_5d"] = to_scalar(pct_change(close, 5).iloc[-1])
    out["chg_20d"] = to_scalar(pct_change(close, 20).iloc[-1])

    lookback_252 = close.tail(252) if len(close) >= 252 else close
    out["high_52w"] = to_scalar(lookback_252.max())
    out["low_52w"] = to_scalar(lookback_252.min())
    out["off_high_52w_pct"] = float((out["price"]/out["high_52w"]) - 1.0) if out["high_52w"] else np.nan
    out["above_low_52w_pct"] = float((out["price"]/out["low_52w"]) - 1.0) if out["low_52w"] else np.nan

    def trend_label(x):
        return "Uptrend" if x else "Down/Sideways"
    out["trend_sma"] = trend_label(out["price"] > out["sma50"] > (out["sma200"] if not math.isnan(out["sma200"]) else -1e9))
    out["rsi_state"] = "Overbought" if out["rsi14"] >= 70 else ("Oversold" if out["rsi14"] <= 30 else "Neutral")
    out["macd_state"] = "Bullish" if out["macd"] > out["macd_signal"] else "Bearish"

    return out


# ======================= Gemini Prompt =======================
def build_ai_prompt(config, market_overview, features_list, news_map):
    tz = config.get("timezone", "Asia/Bangkok")
    today = datetime.now(ZoneInfo(tz)).strftime("%Y-%m-%d")
    risk = config.get("risk_management", {})
    dsl = risk.get("default_stop_loss_pct", 0.03)
    dtp = risk.get("default_take_profit_pct", 0.06)

    overview_joined = "; ".join(
        f"{k}:{item['ticker']} p={fmt_price(item['price'])} 1d={fmt_pct(item['chg_1d'])} rsi={item['rsi14']:.1f} trend={item['trend_sma']}"
        for k, arr in market_overview.items() if arr
        for item in arr
    ) or "-"

    feat_lines = []
    for f in features_list:
        feat_lines.append(
            f"{f['ticker']} price={fmt_price(f['price'])} 1d={fmt_pct(f['chg_1d'])} 5d={fmt_pct(f['chg_5d'])} 20d={fmt_pct(f['chg_20d'])} "
            f"SMA20/50/200={fmt_price(f['sma20'])}/{fmt_price(f['sma50'])}/{fmt_price(f['sma200'])} "
            f"RSI14={f['rsi14']:.1f}({f['rsi_state']}) MACD={f['macd']:.3f}/{f['macd_signal']:.3f}({f['macd_state']}) "
            f"52wH/L={fmt_price(f['high_52w'])}/{fmt_price(f['low_52w'])} offHigh={fmt_pct(f['off_high_52w_pct'])}"
        )
    feat_block = "- " + "\n- ".join(feat_lines) if feat_lines else "-"

    news_lines = []
    for t, items in news_map.items():
        if not items:
            continue
        for i, it in enumerate(items, 1):
            news_lines.append(f"{t} NEWS{i}: {it.get('title','')} | {it.get('link','')}")
    news_block = "- " + "\n- ".join(news_lines) if news_lines else "-"

    sys = (
        "You are a stock analyst. Based on the provided technical metrics and headlines, output STRICT JSON only.\n\n"
        "Schema:\n"
        "{\n"
        '  "date": "YYYY-MM-DD",\n'
        '  "tickers": [\n'
        "    {\n"
        '      "ticker": "TSLA",\n'
        '      "stance": "Buy|Sell|Hold",\n'
        '      "confidence": 0-100,\n'
        '      "entry_rule": "short rule in Thai",\n'
        '      "entry_price_range": "e.g., 240-245 or \'-\'",\n'
        '      "stop_loss": "price or percent string",\n'
        '      "take_profit": "price or percent string",\n'
        '      "timeframe": "days|weeks",\n'
        '      "reasoning_bullets": ["สั้นๆ 2-4 ข้ออ้างอิงตัวเลข indicator"],\n'
        '      "positive_factors": ["2-4 ข่าว/ปัจจัยที่สนับสนุนราคาขึ้น (ไทย) หรือ \'-\'"],\n'
        '      "negative_factors": ["2-4 ข่าว/ปัจจัยที่กดดันราคาลง (ไทย) หรือ \'-\'"],\n'
        '      "news_refs": ["หัวข้อข่าวที่ใช้อ้างอิง"]\n'
        "    }\n"
        "  ],\n"
        '  "notes": "optional"\n'
        "}\n\n"
        "Rules:\n"
        "- ใช้ข้อมูล indicators (SMA/RSI/MACD/52w) + หัวข้อข่าวที่ให้มาเท่านั้น\n"
        "- แยก positive_factors และ negative_factors อย่างละ ≥2 ข้อ ถ้าไม่พบให้ใช้ \"-\"\n"
        "- ใช้ภาษาไทย กระชับ ชัดเจน, หลีกเลี่ยงการแต่งข้อมูลเอง\n"
        f"- ถ้าไม่แน่ใจ SL/TP ให้ใช้ defaults: SL {dsl:.2%}, TP {dtp:.2%}\n"
        "- Output เป็น JSON ล้วน ไม่มี Markdown/โค้ดบล็อก\n"
    )

    user = (
        f"วันที่: {today}\n\n"
        "ภาพรวมตลาด (ย่อ):\n"
        f"- {overview_joined}\n\n"
        "หุ้นที่จะวิเคราะห์:\n"
        f"{feat_block}\n\n"
        "หัวข้อข่าวต่อหุ้น (สำหรับ positive/negative_factors):\n"
        f"{news_block}\n\n"
        "งานของคุณ:\n"
        "- ให้คำแนะนำ Buy/Sell/Hold + แผนเข้า-ออก\n"
        "- แยกปัจจัยบวก/ลบ หลายข้อ โดยอ้างอิงหัวข้อข่าวด้านบนและตัวเลขอินดิเคเตอร์\n"
        "- ส่งออก JSON ตามสคีม (ห้ามมีข้อความอื่น)\n"
    )
    return sys, user

def call_gemini(model_name, system_prompt, user_prompt):
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY")
    if genai is None:
        raise RuntimeError("google-generativeai not installed")
    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content([{"role":"user","parts":[system_prompt + "\n\n" + user_prompt]}])
    text = (resp.text or "").strip().strip("`")
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}\s*$", text, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


# ======================= Charts =======================
def plot_stock(ticker, df, out_path):
    if plt is None or df.empty:
        return
    try:
        fig = plt.figure(figsize=(10, 4.5))
        ax = plt.gca()
        ax.plot(df.index, df["Close"], label="Close")
        ax.plot(df.index, df["Close"].rolling(20).mean(), label="SMA20")
        ax.plot(df.index, df["Close"].rolling(50).mean(), label="SMA50")
        if len(df) >= 200:
            ax.plot(df.index, df["Close"].rolling(200).mean(), label="SMA200")
        ax.set_title(f"{ticker} — Price & SMA")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    except Exception:
        pass


# ======================= Google Sheet =======================
def connect_google_sheet(json_keyfile: str, spreadsheet_id: str):
    if gspread is None or ServiceAccountCredentials is None:
        raise RuntimeError("gspread/oauth2client not installed")
    scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(json_keyfile, scope)
    client = gspread.authorize(creds)
    sh = client.open_by_key(spreadsheet_id)
    try:
        ws = sh.worksheet("Data")
    except gspread.WorksheetNotFound:
        ws = sh.sheet1
    return ws

def export_to_sheet(ws, date_str, features, ai_json):
    rec_map = {t["ticker"]: t for t in ai_json.get("tickers", [])}
    rows = []
    for f in features:
        r = rec_map.get(f["ticker"], {})
        rows.append([
            date_str,
            f["ticker"],
            f["price"],
            r.get("stance","-"),
            r.get("confidence","-"),
            f.get("rsi14",""),
            f.get("sma50",""),
            "; ".join(r.get("positive_factors", []) or []),
            "; ".join(r.get("negative_factors", []) or []),
        ])
    ws.append_rows(rows, value_input_option="RAW")


# ======================= Report =======================
def render_report(date_str, overview, features, ai_json, news_map):
    rec_map = {t["ticker"]: t for t in ai_json.get("tickers", [])}

    md = []
    md.append(f"# Daily AI Stock Insight — {date_str}\n")
    md.append("> *รายงานอัตโนมัติจาก GitHub Actions + Yahoo Finance + Google News + Gemini — เพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน*\n")

    md.append("## ภาพรวมตลาด\n")
    def render_group(name, arr):
        if not arr: return
        md.append(f"**{name}**")
        for it in arr:
            md.append(f"- `{it['ticker']}` ราคา {fmt_price(it['price'])} | 1d {fmt_pct(it['chg_1d'])} | RSI {it['rsi14']:.1f} | แนวโน้ม {it['trend_sma']}")
    render_group("ดัชนี/ETF", overview.get("indices"))
    render_group("สินค้าโภคภัณฑ์ (ETF)", overview.get("commodities"))
    render_group("FX (ETF)", overview.get("fx"))
    md.append("")

    md.append("## สรุปสัญญาณต่อหุ้น\n")
    md.append("| Ticker | Price | 1d | Trend | RSI14 | Stance | Confidence |")
    md.append("|---|---:|---:|---|---:|---|---:|")
    for f in features:
        r = rec_map.get(f["ticker"], {})
        md.append(f"| `{f['ticker']}` | {fmt_price(f['price'])} | {fmt_pct(f['chg_1d'])} | {f['trend_sma']} | {f['rsi14']:.1f} | {r.get('stance','-')} | {r.get('confidence','-')} |")
    md.append("")

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
            md.append("  - เหตุผล (เทคนิค):")
            for b in bullets: md.append(f"    - {b}")

        pos = r.get("positive_factors", [])
        neg = r.get("negative_factors", [])
        if pos and pos != "-":
            md.append("\n**ปัจจัยบวก (Positive Factors):**")
            for b in pos: md.append(f"- {b}")
        if neg and neg != "-":
            md.append("**ปัจจัยลบ (Negative Factors):**")
            for b in neg: md.append(f"- {b}")

        items = news_map.get(f["ticker"], [])
        if items:
            md.append("\n**หัวข้อข่าวล่าสุด:**")
            for it in items:
                title = it.get("title","").strip()
                link = it.get("link","").strip()
                src = it.get("source","")
                pub = it.get("published","")
                if link: md.append(f"- [{title}]({link}) — {src} ({pub})")
                else:    md.append(f"- {title} — {src} ({pub})")
        md.append("")

    notes = ai_json.get("notes","")
    if notes:
        md.append("## หมายเหตุจากโมเดล\n")
        md.append(notes); md.append("")

    md.append("---")
    md.append("**หมายเหตุ/ข้อจำกัด:** รายงานนี้สร้างโดยอัลกอริทึมและ LLM เพื่อการศึกษาเท่านั้น มิใช่คำแนะนำการลงทุน")
    return "\n".join(md)


# ======================= Blocks =======================
def to_overview_block(tickers, lookback_days, delay_sec=0.0):
    arr = []
    for t in tickers:
        df = yahoo_candles(t, lookback_days)
        if df.empty:
            print(f"[overview] skip {t}: no data")
        else:
            f = build_features(t, df)
            if f: arr.append(f)
        if delay_sec > 0:
            time.sleep(delay_sec)
    return arr


# ======================= Main =======================
def main():
    config = load_config("config.yml")
    tz = config.get("timezone", "Asia/Bangkok")
    now = datetime.now(ZoneInfo(tz))
    if config.get("skip_if_weekend", True) and now.weekday() >= 5:
        print("Weekend — skipped."); return

    lookback = int(config.get("lookback_days", 260))
    tickers = config.get("tickers")
    indices = config.get("indices")
    commodities = config.get("commodities")
    fx = config.get("fx")
    cfg_news = config.get("news", {"enable": True, "lookback_days": 2, "per_ticker": 3})
    delay = float(config.get("per_call_delay_sec", 0.0))
    locale_news = config.get("google_news_locale", {"hl":"en-US","gl":"US","ceid":"US:en"})

    # 1) ภาพรวมตลาด
    overview = {
        "indices": to_overview_block(indices, lookback, delay_sec=delay),
        "commodities": to_overview_block(commodities, lookback, delay_sec=delay),
        "fx": to_overview_block(fx, lookback, delay_sec=delay),
    }

    # 2) คุณลักษณะ/ตัวชี้วัดของหุ้นหลัก
    features = []
    hist_cache = {}
    for t in tickers:
        df = yahoo_candles(t, lookback)
        hist_cache[t] = df
        f = build_features(t, df) if not df.empty else None
        if f: features.append(f)
        if delay > 0:
            time.sleep(delay)

    if not features:
        # เขียนรายงานสั้น ๆ เพื่อไม่ให้ workflow ล้ม
        ensure_dir("reports")
        report_date = now.strftime("%Y-%m-%d")
        msg = ("# Daily AI Stock Insight — {d}\n\n"
               "ไม่สามารถโหลดข้อมูลจาก Yahoo Finance ได้ในรอบนี้ (เน็ตขัดข้อง/สัญลักษณ์ผิดพลาด).").format(d=report_date)
        open(f"reports/{report_date}.md","w",encoding="utf-8").write(msg)
        open("reports/latest.md","w",encoding="utf-8").write(msg)
        print("Report generated (empty)."); return

    # 3) ข่าวจาก Google News RSS
    news_map = {}
    if cfg_news.get("enable", True):
        per_ticker = int(cfg_news.get("per_ticker", 3))
        lb_days = int(cfg_news.get("lookback_days", 2))
        for f in features:
            t = f["ticker"]
            q = COMPANY_NAME.get(t, t)
            items = google_news_company(q, lookback_days=lb_days, limit=per_ticker, locale=locale_news)
            news_map[t] = items
            if delay > 0:
                time.sleep(delay)

    # 4) สร้าง prompt และเรียก Gemini
    sys, usr = build_ai_prompt(config, overview, features, news_map)
    model_name = config.get("gemini_model", "gemini-2.5-flash")
    ai_json = call_gemini(model_name, sys, usr)

    # 5) วาดกราฟ + ออกรายงาน
    report_date = now.strftime("%Y-%m-%d")
    ensure_dir("reports")

    if config.get("charts", {}).get("enable", True) and plt is not None:
        for t in tickers:
            df = hist_cache.get(t)
            if df is None or df.empty: continue
            out_png = f"reports/{report_date}_{t}.png"
            plot_stock(t, df, out_png)

    md = render_report(report_date, overview, features, ai_json, news_map)
    with open(f"reports/{report_date}.md","w",encoding="utf-8") as f: f.write(md)
    with open("reports/latest.md","w",encoding="utf-8") as f: f.write(md)
    print("Report generated:", report_date)

    # 6) ส่งข้อมูลเข้า Google Sheet (ออปชัน)
    sheet_id = os.getenv("SHEET_ID", "").strip()
    if sheet_id and os.path.exists("gcp_service_account.json"):
        try:
            ws = connect_google_sheet("gcp_service_account.json", sheet_id)
            export_to_sheet(ws, report_date, features, ai_json)
            print("Exported to Google Sheet")
        except Exception as e:
            print("Google Sheet export failed:", e)
    else:
        print("Skip Google Sheet export (missing SHEET_ID or key file).")


if __name__ == "__main__":
    main()
