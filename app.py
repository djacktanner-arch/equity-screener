# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import os
import tempfile
import time
from sec_edgar_downloader import Downloader
import openai
from typing import Tuple
from pathlib import Path
import pdfkit  # optional; or use streamlit download of HTML

# -------- CONFIG ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "PLEASE_SET_OPENAI_KEY_IN_STREAMLIT_SECRETS"
USER_AGENT = "EquityScreenerDemo/1.0 your_email@example.com"
openai.api_key = OPENAI_API_KEY
# sec-edgar-downloader stores files in ./sec_filings by default
dl = Downloader("sec_filings")
# ---------------------------

st.set_page_config(layout="wide", page_title="AI Equity Screener MVP")

st.title("AI Equity Screener — MVP")

with st.sidebar:
    st.header("Screener Inputs")
    tickers_input = st.text_input("Tickers (comma-separated)", "AAPL, MSFT, TSLA")
    max_results = st.number_input("Max tickers to analyze (AI summaries)", min_value=1, max_value=10, value=3)
    score_weights = {
        "pe": st.slider("Weight: P/E (lower better)", 0, 100, 25),
        "rev_growth": st.slider("Weight: Revenue growth", 0, 100, 25),
        "debt_equity": st.slider("Weight: Debt/Equity (lower better)", 0, 100, 25),
        "ai_risk": st.slider("Weight: AI risk flags (penalty)", 0, 100, 25),
    }
    if st.button("Run Screener"):
        run = True
    else:
        run = False

def fetch_yfinance(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = t.info
    # quick fundamental picks (handle missing gracefully)
    pe = info.get("trailingPE") or info.get("forwardPE") or None
    rev = info.get("totalRevenue") or None
    debt_to_equity = None
    try:
        bs = t.quarterly_balance_sheet if hasattr(t, "quarterly_balance_sheet") else None
    except Exception:
        bs = None
    # fallback simple
    return {"ticker": ticker, "info": info, "pe": pe, "totalRevenue": rev, "debtEquity": info.get("debtToEquity")}

def download_latest_filings(ticker: str, count:int=2) -> Tuple[str,str]:
    """Downloads latest 10-K or 10-Q and returns path to the first relevant filing text file."""
    # try to download 10-K then 10-Q
    try:
        dl.get("10-K", ticker, amount=count)
        # find most recent 10-K under sec_filings/company/<ticker>/10-K
    except Exception:
        pass
    try:
        dl.get("10-Q", ticker, amount=count)
    except Exception:
        pass
    # search downloaded dir for the most recent 10-K/10-Q file
    base = Path("sec_filings") / ticker.upper()
    if not base.exists():
        return None
    # walk files for .txt filings
    files = list(base.rglob("*.txt"))
    # prefer 10-K filenames
    files_sorted = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files_sorted[0]) if files_sorted else None

def call_gpt_summarize(filing_text: str, ticker: str) -> dict:
    system = ("You are a concise financial research assistant. Summarize the filing, list top 3-5 risks, "
              "and include short evidence citations (section heading or short text snippet). Be factual and include a one-sentence disclaimer.")
    user = (f"Filing for {ticker} below:\n\n{filing_text[:3500]}\n\n"
            "TASK: 1) Produce a 250-350 word plain-language summary of business and key drivers. "
            "2) List up to 5 red flags (one sentence each) with a 20-60 char supporting excerpt. "
            "3) Provide a one-sentence final note (no advice). Output as JSON with keys: summary, risks (list of {flag, evidence}).")
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini" if hasattr(openai, "ChatCompletion") else "gpt-4o-mini",
            messages=[{"role":"system","content":system}, {"role":"user","content":user}],
            temperature=0.2,
            max_tokens=800,
        )
        text = resp["choices"][0]["message"]["content"]
    except Exception as e:
        text = f"ERROR calling GPT: {e}"
    return text

def simple_score(row, weights):
    # normalize metrics (very simple)
    pe = row.get("pe") or 1e6
    rev = row.get("totalRevenue") or 0
    debt = row.get("debtEquity") or 1e6
    # inverse for pe & debt
    pe_score = 1 / max(pe, 1)
    rev_score = (rev / 1e9) if rev else 0
    debt_score = 1 / (1 + max(debt, 0))
    ai_penalty = row.get("ai_risk_count", 0)
    raw = (weights["pe"] * pe_score + weights["rev_growth"] * rev_score + weights["debt_equity"] * debt_score) - weights["ai_risk"] * ai_penalty
    return raw

if run:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    results = []
    st.info(f"Fetching data for {len(tickers)} tickers...")
    for tck in tickers:
        with st.spinner(f"Fetching {tck}"):
            y = fetch_yfinance(tck)
            filing_path = None
            try:
                filing_path = download_latest_filings(tck, count=1)
            except Exception as e:
                st.warning(f"EDGAR download issue for {tck}: {e}")
            ai_text = None
            ai_risk_count = 0
            if filing_path:
                with open(filing_path, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
                # trim long text passed to GPT
                ai_text = call_gpt_summarize(txt, tck)
                # crude risk count heuristic
                ai_risk_count = ai_text.lower().count("risk") if ai_text else 0
            results.append({
                "ticker": tck,
                "pe": y.get("pe"),
                "totalRevenue": y.get("totalRevenue"),
                "debtEquity": y.get("debtEquity"),
                "ai_summary": ai_text,
                "ai_risk_count": ai_risk_count,
                "filing_path": filing_path
            })
            time.sleep(0.3)  # be polite
    # scoring
    for r in results:
        r["score_raw"] = simple_score(r, score_weights)
    df = pd.DataFrame(results).sort_values("score_raw", ascending=False).reset_index(drop=True)
    st.subheader("Screener Results")
    st.dataframe(df[["ticker","pe","totalRevenue","debtEquity","score_raw"]].fillna("N/A"))
    # show top N with AI summaries
    st.subheader(f"Top {max_results} AI Summaries")
    for i, row in df.head(max_results).iterrows():
        st.markdown(f"### {row['ticker']} — score {row['score_raw']:.4f}")
        if row["ai_summary"]:
            st.code(row["ai_summary"][:12000])
        else:
            st.write("No filing summary available.")
        # export button for PDF (render HTML)
        if st.button(f"Export report for {row['ticker']}"):
            html = f"<h1>{row['ticker']} — AI Summary</h1><pre>{row['ai_summary']}</pre>"
            tmpf = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
            tmpf.write(html.encode("utf-8"))
            tmpf.flush()
            pdf_path = tmpf.name.replace(".html", ".pdf")
            try:
                pdfkit.from_file(tmpf.name, pdf_path)
                with open(pdf_path, "rb") as pf:
                    st.download_button(label="Download PDF", data=pf, file_name=f"{row['ticker']}_report.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"PDF export failed: {e}. You can copy the summary manually.")
