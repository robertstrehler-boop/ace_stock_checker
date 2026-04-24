import streamlit as st
import streamlit.components.v1 as st_components
import pandas as pd
import numpy as np
import yfinance as yf
import json
import os
from datetime import datetime
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

WATCHLIST_FILE  = "watchlist.json"
PORTFOLIO_FILE  = "portfolio.json"
PORTFOLIO_NAMES = ["Quiet Compounder", "Hidden Champions"]

# Bekannte ETFs: region, stil, defensiver Anteil (geschätzt)
ETF_MAP = {
    "IWDA.AS":  {"name": "iShares MSCI World",          "region": "Global",   "style": "Blend",  "def_pct": 0.33},
    "VWCE.DE":  {"name": "Vanguard FTSE All-World",      "region": "Global",   "style": "Blend",  "def_pct": 0.33},
    "SXR8.DE":  {"name": "iShares Core S&P 500",         "region": "USA",      "style": "Blend",  "def_pct": 0.28},
    "XMME.DE":  {"name": "Xtrackers MSCI EM",            "region": "Emerging", "style": "Growth", "def_pct": 0.20},
    "EXH1.DE":  {"name": "iShares STOXX Europe 600",     "region": "Europe",   "style": "Blend",  "def_pct": 0.40},
    "EUNL.DE":  {"name": "iShares Core MSCI World",      "region": "Global",   "style": "Blend",  "def_pct": 0.33},
    "IUSQ.DE":  {"name": "iShares MSCI ACWI",            "region": "Global",   "style": "Blend",  "def_pct": 0.33},
    "DBXD.DE":  {"name": "Xtrackers DAX",                "region": "Germany",  "style": "Blend",  "def_pct": 0.35},
    "EXS1.DE":  {"name": "iShares Core DAX",             "region": "Germany",  "style": "Blend",  "def_pct": 0.35},
    "QDVE.DE":  {"name": "iShares S&P 500 IT",           "region": "USA",      "style": "Growth", "def_pct": 0.05},
    "XDWD.DE":  {"name": "Xtrackers MSCI World",         "region": "Global",   "style": "Blend",  "def_pct": 0.33},
    "MEUD.PA":  {"name": "Amundi MSCI Europe",           "region": "Europe",   "style": "Blend",  "def_pct": 0.40},
    "SPYD.DE":  {"name": "SPDR S&P 500",                 "region": "USA",      "style": "Blend",  "def_pct": 0.28},
    "HMWO.L":   {"name": "HSBC MSCI World",              "region": "Global",   "style": "Blend",  "def_pct": 0.33},
    "VGWL.DE":  {"name": "Vanguard FTSE All-World",      "region": "Global",   "style": "Blend",  "def_pct": 0.33},
    "IS3N.DE":  {"name": "iShares Core MSCI EM IMI",     "region": "Emerging", "style": "Blend",  "def_pct": 0.22},
    "EXXT.DE":  {"name": "iShares NASDAQ-100",           "region": "USA",      "style": "Growth", "def_pct": 0.05},
    "SXRV.DE":  {"name": "iShares Core MSCI World",      "region": "Global",   "style": "Blend",  "def_pct": 0.33},
}

# Portfolio-Charakter-Profile
PORTFOLIO_PROFILES = {
    "Quiet Compounder": {
        "desc": "Stabile Compounder mit niedrigem Beta, verlässlichen Margen und Dividenden",
        "beta_max": 1.1, "beta_ideal": 0.85,
        "div_yield_min": 0.01,
        "prefer_sectors": {"Consumer Defensive", "Healthcare", "Industrials", "Financial Services"},
        "avoid_sectors":  {"Energy", "Basic Materials"},
    },
    "Hidden Champions": {
        "desc": "Nischenspieler mit Wachstumspotenzial, kleinere bis mittlere Marktkapitalisierung",
        "beta_max": 1.5, "beta_ideal": 1.1,
        "div_yield_min": 0.0,
        "prefer_sectors": {"Technology", "Industrials", "Healthcare", "Consumer Cyclical"},
        "avoid_sectors":  {"Utilities"},
    },
}

DEFENSIVE_SECTORS = {"Healthcare", "Consumer Defensive", "Utilities"}
GROWTH_SECTORS    = {"Technology", "Consumer Cyclical", "Communication Services", "Basic Materials"}

st.set_page_config(page_title="Velox", page_icon="⚡", layout="wide")

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def to_float(x):
    if x is None: return None
    if isinstance(x, (int, float)):
        try:
            return None if (isinstance(x, float) and np.isnan(x)) else float(x)
        except Exception: return None
    s = str(x).strip()
    if not s: return None
    s = s.replace("€","").replace("%","").replace(" ","").replace("Mrd","").replace("Mio","").strip()
    if "," in s and "." in s: s = s.replace(".","").replace(",",".")
    elif "," in s: s = s.replace(",",".")
    try: return float(s)
    except Exception: return None

def safe_float(x):
    try:
        if x is None: return None
        if isinstance(x, float) and np.isnan(x): return None
        return float(x)
    except Exception: return None

def clip_score(x): return float(np.clip(x, 1, 10))
def percent(x): return f"{x*100:.1f}%"

def bucket(score):
    if score >= 8.0: return "stark"
    if score >= 7.0: return "gut"
    if score >= 6.0: return "ok"
    if score >= 5.0: return "unruhig"
    return "nicht sauber"

def compute_position_weight(portfolio_total, position_value):
    if not portfolio_total or portfolio_total <= 0: return None
    if not position_value or position_value <= 0: return None
    return position_value / portfolio_total

def current_price_from_df(df):
    if df is None or df.empty or "Close" not in df.columns: return None
    v = df["Close"].dropna()
    return float(v.iloc[-1]) if len(v) > 0 else None

def lower_text(*parts):
    return " ".join(str(p).lower() for p in parts if p is not None)

def fmt_v(v, fallback=""):
    if v is None: return fallback
    return f"{v:.2f}".replace(".", ",") if isinstance(v, float) else str(v).replace(".", ",")

# ══════════════════════════════════════════════════════════════════════════════
# Watchlist — Snapshot-basiert
# Struktur: [{ticker, name, mode, notes, snapshots:[{saved_at, scores, triggers, risks, ...}]}]
# ══════════════════════════════════════════════════════════════════════════════
def load_watchlist():
    if not os.path.exists(WATCHLIST_FILE): return []
    try:
        raw = json.load(open(WATCHLIST_FILE, "r", encoding="utf-8"))
        migrated = []
        for item in raw:
            if "snapshots" not in item:
                snap = {k: v for k, v in item.items() if k not in ("ticker","name","mode","notes")}
                snap.setdefault("saved_at", item.get("saved_at",""))
                migrated.append({"ticker": item.get("ticker",""), "name": item.get("name",""),
                                  "mode": item.get("mode",""), "notes": item.get("notes",""),
                                  "snapshots": [snap]})
            else:
                migrated.append(item)
        return migrated
    except Exception: return []

def _wl_key(ticker, mode): return (ticker.upper(), mode)

def save_snapshot_to_watchlist(entry, notes=""):
    wl = load_watchlist()
    key = _wl_key(entry["ticker"], entry["mode"])
    snapshot = {k: entry.get(k) for k in
                ("saved_at","fund_score","timing_score","story_score","total_score",
                 "action","triggers","risks","metrics","red_flags")}
    for item in wl:
        if _wl_key(item["ticker"], item["mode"]) == key:
            item["snapshots"].append(snapshot)
            item["name"] = entry.get("name") or item["name"]
            if notes: item["notes"] = notes
            break
    else:
        wl.append({"ticker": entry["ticker"], "name": entry.get("name", entry["ticker"]),
                   "mode": entry["mode"], "notes": notes, "snapshots": [snapshot]})
    json.dump(wl, open(WATCHLIST_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def update_watchlist_notes(ticker, mode, notes):
    wl = load_watchlist()
    key = _wl_key(ticker, mode)
    for item in wl:
        if _wl_key(item["ticker"], item["mode"]) == key:
            item["notes"] = notes; break
    json.dump(wl, open(WATCHLIST_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def remove_from_watchlist(ticker, mode):
    wl = [x for x in load_watchlist() if _wl_key(x["ticker"], x["mode"]) != _wl_key(ticker, mode)]
    json.dump(wl, open(WATCHLIST_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# Portfolio — Screenshot-Import + manuelle Pflege
# Struktur: { "Quiet Compounder": { "positions": [...] }, "Hidden Champions": {...} }
# ══════════════════════════════════════════════════════════════════════════════
# Portfolio-Analyse: Datenabruf, Scoring, AI-Narrative
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_position_meta(ticker: str) -> dict:
    """Holt Sektor, Beta, KGV, Dividendenrendite, Marktkapitalisierung von Yahoo."""
    if not ticker: return {}
    tkr = ticker.upper().strip()
    # Bekannte ETFs direkt aus Mapping
    if tkr in ETF_MAP:
        m = ETF_MAP[tkr]
        return {"is_etf": True, "sector": "ETF — " + m["region"],
                "region": m["region"], "style": m["style"],
                "def_pct": m["def_pct"], "beta": 1.0, "pe": None,
                "div_yield": None, "market_cap": None, "country": m["region"]}
    try:
        info = yf.Ticker(tkr).info
        qt   = (info.get("quoteType") or "").upper()
        is_etf = qt in ("ETF", "MUTUALFUND")
        return {
            "is_etf":     is_etf,
            "sector":     info.get("sector", ""),
            "industry":   info.get("industry", ""),
            "country":    info.get("country", ""),
            "region":     info.get("country", ""),
            "beta":       safe_float(info.get("beta")),
            "pe":         safe_float(info.get("trailingPE")),
            "div_yield":  safe_float(info.get("dividendYield")),
            "market_cap": safe_float(info.get("marketCap")),
            "style":      "ETF" if is_etf else "",
            "def_pct":    None,
        }
    except Exception:
        return {}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_benchmark_return(ticker="IWDA.AS", period="1y") -> float | None:
    """12-Monats-Rendite des MSCI World als Benchmark."""
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty: return None
        start, end = hist["Close"].iloc[0], hist["Close"].iloc[-1]
        return (end - start) / start * 100
    except Exception:
        return None

def score_portfolio(positions: list, pname: str, metas: dict) -> dict:
    """
    Berechnet drei Scores (1–10) und Zusatzdaten für ein Portfolio.
    metas = {ticker: fetch_position_meta(ticker)}
    """
    total_cv = sum(p.get("current_value") or 0 for p in positions)
    if total_cv == 0:
        return {}

    # ── Gewichte ──────────────────────────────────────────────────────────────
    weights = {}
    for p in positions:
        tkr = (p.get("ticker") or "").upper()
        cv  = p.get("current_value") or 0
        weights[tkr] = cv / total_cv

    # ── 1. AUSGEWOGENHEIT ─────────────────────────────────────────────────────
    # a) Klumpenrisiko: größte Position
    max_wgt  = max(weights.values()) if weights else 0
    conc_score = 10 if max_wgt <= 0.10 else \
                  9 if max_wgt <= 0.15 else \
                  8 if max_wgt <= 0.20 else \
                  6 if max_wgt <= 0.30 else \
                  4 if max_wgt <= 0.40 else 2

    # b) Sektorvielfalt
    sectors = {}
    for p in positions:
        tkr = (p.get("ticker") or "").upper()
        m   = metas.get(tkr, {})
        sec = m.get("sector") or "Unbekannt"
        sectors[sec] = sectors.get(sec, 0) + (weights.get(tkr, 0))
    n_sectors  = len([s for s in sectors if not s.startswith("ETF")])
    sec_score  = min(10, max(1, n_sectors * 1.8))

    # c) Anzahl Positionen
    n = len(positions)
    n_score = 10 if 8 <= n <= 20 else \
               8 if 5 <= n <= 25  else \
               5 if 3 <= n        else 2

    balance_score = clip_score((conc_score * 0.45 + sec_score * 0.35 + n_score * 0.20))

    # ── 2. STABILITÄT ─────────────────────────────────────────────────────────
    beta_weighted = 0.0
    def_weight    = 0.0
    etf_weight    = 0.0
    div_weighted  = 0.0
    beta_count    = 0

    for p in positions:
        tkr = (p.get("ticker") or "").upper()
        w   = weights.get(tkr, 0)
        m   = metas.get(tkr, {})

        beta = m.get("beta")
        if beta and 0 < beta < 5:
            beta_weighted += beta * w; beta_count += 1

        if m.get("is_etf"):
            etf_weight += w
            def_weight += w * (m.get("def_pct") or 0.33)
        else:
            sec = m.get("sector") or ""
            if sec in DEFENSIVE_SECTORS:
                def_weight += w

        div = m.get("div_yield")
        if div and div > 0:
            div_weighted += div * w

    avg_beta   = beta_weighted if beta_count else 1.0
    beta_score = 10 if avg_beta <= 0.70 else \
                  8 if avg_beta <= 0.85 else \
                  7 if avg_beta <= 1.00 else \
                  5 if avg_beta <= 1.20 else \
                  3 if avg_beta <= 1.40 else 1

    def_score  = min(10, def_weight * 14)       # 70% defensiv = 10
    etf_score  = min(10, etf_weight * 14)
    div_score  = min(10, div_weighted * 250)     # 4% Portfoliorendite = 10

    stab_score = clip_score(beta_score * 0.35 + def_score * 0.30 +
                             etf_score * 0.20  + div_score * 0.15)

    # ── 3. WACHSTUMSPOTENZIAL ─────────────────────────────────────────────────
    growth_weight = 0.0
    smallmid_wgt  = 0.0
    pe_vals       = []

    for p in positions:
        tkr = (p.get("ticker") or "").upper()
        w   = weights.get(tkr, 0)
        m   = metas.get(tkr, {})
        sec = m.get("sector") or ""

        if sec in GROWTH_SECTORS:
            growth_weight += w
        mc = m.get("market_cap")
        if mc and mc < 10e9:               # < 10 Mrd = Small/Mid
            smallmid_wgt += w
        pe = m.get("pe")
        if pe and 0 < pe < 80:
            pe_vals.append(pe)

    growth_sec_score = min(10, growth_weight * 13)
    smallmid_score   = min(10, smallmid_wgt  * 14)
    # Niedriges PE = mehr Upside, sehr hohes PE = riskanter
    avg_pe     = sum(pe_vals) / len(pe_vals) if pe_vals else 20
    pe_score   = 9 if avg_pe < 15 else 7 if avg_pe < 22 else 6 if avg_pe < 30 else 4 if avg_pe < 50 else 2

    growth_score = clip_score(growth_sec_score * 0.50 + smallmid_score * 0.25 + pe_score * 0.25)

    # ── Gesamt-Score ──────────────────────────────────────────────────────────
    total_score = clip_score(balance_score * 0.35 + stab_score * 0.35 + growth_score * 0.30)

    # ── Portfolio-Charakter-Fit ───────────────────────────────────────────────
    profile    = PORTFOLIO_PROFILES.get(pname, {})
    fit_notes  = []
    if profile:
        if avg_beta > profile.get("beta_max", 99):
            fit_notes.append(f"Beta {avg_beta:.2f} ist zu hoch für dieses Portfolio-Profil")
        prefer = profile.get("prefer_sectors", set())
        prefer_wgt = sum(weights.get(t,0) for t,m in metas.items()
                         if (m.get("sector") or "") in prefer)
        if prefer_wgt < 0.25 and prefer:
            fit_notes.append(f"Wenig Gewicht in Kernsektoren ({', '.join(list(prefer)[:3])})")
        avoid = profile.get("avoid_sectors", set())
        avoid_wgt = sum(weights.get(t,0) for t,m in metas.items()
                        if (m.get("sector") or "") in avoid)
        if avoid_wgt > 0.20:
            fit_notes.append(f"Erhöhter Anteil in Sektoren, die wenig zum Profil passen")

    return {
        "balance_score":  round(balance_score,  1),
        "stab_score":     round(stab_score,      1),
        "growth_score":   round(growth_score,    1),
        "total_score":    round(total_score,     1),
        "avg_beta":       round(avg_beta,        2),
        "def_weight":     round(def_weight,      3),
        "etf_weight":     round(etf_weight,      3),
        "div_yield_pf":   round(div_weighted,    4),
        "avg_pe":         round(avg_pe,          1),
        "max_wgt":        round(max_wgt,         3),
        "n_positions":    n,
        "sectors":        sectors,
        "fit_notes":      fit_notes,
        "growth_weight":  round(growth_weight,   3),
    }

def generate_portfolio_narrative(pname: str, scores: dict, positions: list,
                                  metas: dict, benchmark_ret: float | None,
                                  portfolio_ret: float, api_key: str) -> str:
    """Generiert eine AI-Analyse des Portfolios auf Deutsch via OpenAI."""
    if not api_key: return ""
    try:
        client = OpenAI(api_key=api_key)
        profile_desc = PORTFOLIO_PROFILES.get(pname, {}).get("desc", "")

        # Positionen kompakt zusammenfassen
        pos_lines = []
        total_cv = sum(p.get("current_value") or 0 for p in positions)
        for p in sorted(positions, key=lambda x: -(x.get("current_value") or 0))[:12]:
            tkr  = (p.get("ticker") or "—").upper()
            cv   = p.get("current_value") or 0
            wgt  = cv / total_cv * 100 if total_cv else 0
            perf = p.get("perf_since_buy_pct") or 0
            m    = metas.get(tkr, {})
            sec  = m.get("sector") or ("ETF" if m.get("is_etf") else "unbekannt")
            beta = m.get("beta")
            beta_str = f"Beta {beta:.2f}" if beta else ""
            pos_lines.append(f"- {p.get('name','?')} ({tkr}): {wgt:.1f}% | {sec} | {perf:+.1f}% | {beta_str}")

        bench_line = f"MSCI World (1J): {benchmark_ret:+.1f}%" if benchmark_ret is not None else ""
        fit_notes  = "; ".join(scores.get("fit_notes", [])) or "keine besonderen Abweichungen"

        prompt = f"""Du bist ein erfahrener, unabhängiger Portfolio-Analyst. Analysiere das folgende Portfolio präzise und ehrlich auf Deutsch.

Portfolio: "{pname}"
Strategie-Profil: {profile_desc}

SCORES (1–10):
- Ausgewogenheit: {scores['balance_score']}
- Stabilität: {scores['stab_score']}
- Wachstumspotenzial: {scores['growth_score']}
- Gesamt: {scores['total_score']}

KENNZAHLEN:
- {scores['n_positions']} Positionen | größte Position: {scores['max_wgt']*100:.1f}%
- Ø Portfolio-Beta: {scores['avg_beta']} | Defensiver Anteil: {scores['def_weight']*100:.0f}%
- ETF-Anteil: {scores['etf_weight']*100:.0f}% | Ø KGV: {scores['avg_pe']}
- Portfolio-Performance (seit Kauf, gewichtet): {portfolio_ret:+.1f}%
- {bench_line}
- Profil-Fit-Hinweise: {fit_notes}

TOP-POSITIONEN:
{chr(10).join(pos_lines)}

Schreibe eine Analyse in DREI klar getrennten Abschnitten:
1. **Was gut läuft** — Stärken des Portfolios, was zur Strategie passt (2–3 Sätze)
2. **Was kritisch zu prüfen ist** — konkrete Schwachstellen, Klumpenrisiken, Strategie-Abweichungen (2–3 Sätze)
3. **Drei konkrete Empfehlungen** — spezifische, umsetzbare Handlungsschritte (nummerierte Liste)

Ton: direkt, sachkundig, ohne Floskeln. Kein "Als KI..." oder ähnliches. Maximal 300 Wörter gesamt."""

        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[{"role": "user", "content": prompt}]
        )
        return resp.output_text.strip()
    except Exception as e:
        return f"Analyse konnte nicht erstellt werden: {e}"

# ══════════════════════════════════════════════════════════════════════════════
# Position: { name, ticker, current_value, perf_since_buy_pct, shares, avg_price,
#              invested, last_updated, notes }
# ══════════════════════════════════════════════════════════════════════════════
def load_portfolio():
    if not os.path.exists(PORTFOLIO_FILE): return {n: {"positions": []} for n in PORTFOLIO_NAMES}
    try:
        data = json.load(open(PORTFOLIO_FILE, "r", encoding="utf-8"))
        for n in PORTFOLIO_NAMES:
            data.setdefault(n, {"positions": []})
        return data
    except Exception:
        return {n: {"positions": []} for n in PORTFOLIO_NAMES}

def save_portfolio(data):
    json.dump(data, open(PORTFOLIO_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def pf_display_name(port_data: dict, internal_name: str) -> str:
    """Gibt den benutzerdefinierten Portfolio-Namen zurück, oder den internen Namen."""
    return (port_data.get("_meta", {})
                     .get("custom_names", {})
                     .get(internal_name, internal_name))

def pf_set_display_name(port_data: dict, internal_name: str, display: str) -> dict:
    """Setzt einen benutzerdefinierten Portfolio-Namen in port_data und speichert."""
    port_data.setdefault("_meta", {}).setdefault("custom_names", {})[internal_name] = display.strip() or internal_name
    save_portfolio(port_data)
    return port_data

def _normalize_ticker(t: str) -> str:
    """Entfernt Börsen-Suffix: ASML.AS → ASML, RGL.DE → RGL, MSFT.DE → MSFT."""
    t = (t or "").upper().strip()
    if "." in t:
        t = t.split(".")[0]
    return t

def _normalize_name(n: str) -> str:
    """Bereinigt Namen für Vergleich: Kleinbuchstaben, ohne Sonderzeichen."""
    import re
    n = (n or "").lower().strip()
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    return re.sub(r"\s+", " ", n).strip()

# Generische Unternehmens-Suffixe die NICHT als Match-Kriterium taugen
_NAME_STOPWORDS = {
    "holding", "holdings", "group", "corp", "corporation", "company",
    "incorporated", "limited", "aktiengesellschaft", "gesellschaft",
    "international", "global", "technologies", "technology", "solutions",
    "services", "systems", "industries", "enterprises", "partners",
    "capital", "fund", "trust", "bank", "financials", "finance",
    "gmbh", "plc", "inc", "ltd", "nv", "bv", "asa", "oyj", "sab",
    "sarl", "sas", "spa", "kgaa", "management", "investments",
    # Sektor-Begriffe die als alleiniges Match falsch wären
    "semiconductor", "semiconductors", "pharma", "pharmaceutical",
    "energy", "electric", "electronics", "digital", "software",
    "media", "communications", "resources", "materials", "chemicals",
    "automotive", "aerospace", "healthcare", "medical", "insurance",
    "real", "estate", "logistics", "transport", "retail", "consumer",
}

def _name_match(a: str, b: str) -> bool:
    """True wenn Namen signifikant überlappen — generische Worte werden ignoriert."""
    if not a or not b: return False
    wa = set(w for w in _normalize_name(a).split()
             if len(w) >= 4 and w not in _NAME_STOPWORDS)
    wb = set(w for w in _normalize_name(b).split()
             if len(w) >= 4 and w not in _NAME_STOPWORDS)
    # Beide Sets müssen nicht-leer sein und mind. 1 echtes Wort teilen
    return bool(wa and wb and wa & wb)

def find_in_portfolio(ticker, isin: str = "", name_hint: str = ""):
    """Gibt (portfolio_name, position_dict, match_reason) zurück.
    4-stufiges Matching:
      1. Exakter Ticker-Vergleich          (RGLD == RGLD)
      2. Normalisiert ohne Suffix          (RGL.DE → RGL == RGL aus RGLD? nein, aber RGL == RGL)
      3. ISIN                              (zuverlässigster Key)
      4. Name-Overlap                      (Royal Gold Corp ≈ Royal Gold)
    Gibt immer ein 3-Tuple zurück: (pname, pos_dict, reason_str)
    """
    if not ticker and not isin and not name_hint: return None, None, ""
    port = load_portfolio()
    tk_norm = _normalize_ticker(ticker)

    candidates = []  # (priority, pname, pos, reason)
    for pname, pdata in port.items():
        for pos in pdata.get("positions", []):
            pos_tk   = (pos.get("ticker") or "").upper().strip()
            pos_norm = _normalize_ticker(pos_tk)
            pos_isin = (pos.get("isin") or "").upper().strip()
            pos_name = pos.get("name") or ""

            if ticker and pos_tk == ticker.upper():
                candidates.append((1, pname, pos, "Ticker exakt"))
            elif ticker and pos_norm == tk_norm and tk_norm:
                candidates.append((2, pname, pos, f"Ticker ({pos_tk})"))
            elif isin and pos_isin and pos_isin == isin.upper():
                candidates.append((3, pname, pos, "ISIN"))
            elif name_hint and _name_match(pos_name, name_hint):
                candidates.append((4, pname, pos, f"Name ({pos_name})"))

    if not candidates: return None, None, ""
    candidates.sort(key=lambda x: x[0])
    _, pname, pos, reason = candidates[0]
    _d = calc_position_derived(pos)
    return pname, {**pos, **_d, "perf_since_buy_pct": _d["pl_pct"]}, reason

def calc_position_derived(pos: dict) -> dict:
    """Berechnet current_value, invested, pl_abs, pl_pct aus gespeicherten Feldern.

    Priorität für invested:
      1. shares × avg_price  (wenn beide vorhanden)
      2. invested_csv         (aus TR-CSV-Import)
      3. invested             (alter Feldname / Legacy)
    Priorität für current_value:
      1. shares × current_price
      2. current_value        (Legacy)
    """
    shares = safe_float(pos.get("shares")) or 0
    avg    = safe_float(pos.get("avg_price")) or 0
    cp     = safe_float(pos.get("current_price")) or 0

    if shares and avg:
        inv = round(shares * avg, 2)
    else:
        inv = (safe_float(pos.get("invested_csv"))
               or safe_float(pos.get("invested")) or 0)

    if shares and cp:
        cv = round(shares * cp, 2)
    else:
        cv = safe_float(pos.get("current_value")) or 0

    pl  = round(cv - inv, 2)
    pct = round(pl / inv * 100, 2) if inv > 0 else 0
    return {"invested": inv, "current_value": cv, "pl_abs": pl, "pl_pct": pct}

def refresh_portfolio_prices(port_data: dict) -> dict:
    """Holt aktuelle Kurse von Yahoo für alle Positionen mit Ticker.
    Konvertiert automatisch USD/GBP → EUR."""
    today = datetime.now().strftime("%Y-%m-%d")
    fx_cache: dict = {}
    for pdata in port_data.values():
        for pos in pdata.get("positions", []):
            tkr = (pos.get("ticker") or "").strip()
            if not tkr:
                continue
            try:
                info = yf.Ticker(tkr).info or {}
                cp = safe_float(info.get("regularMarketPrice")
                                or info.get("currentPrice"))
                if not cp:
                    continue
                ccy = (info.get("currency") or "EUR").upper()
                # Pence → Pound Korrektur (LSE-Ticker in GBp)
                if ccy == "GBP" and cp > 100:
                    cp = cp / 100
                    ccy = "GBP"
                # Währungskorrektur → EUR
                if ccy != "EUR":
                    if ccy not in fx_cache:
                        fx_cache[ccy] = _get_fx_rate(ccy)
                    cp = round(cp * fx_cache[ccy], 4)
                pos["current_price"]     = cp
                pos["last_price_update"] = today
                pos["price_currency"]    = ccy   # für Debugging speichern
            except Exception:
                pass
    return port_data

def apply_buy(pos: dict, shares: float, price: float, fees: float = 1.0) -> dict:
    """Nachkauf: Anteile erhöhen, Ø-Kurs neu berechnen (Durchschnittskostenmethode)."""
    old_s = safe_float(pos.get("shares")) or 0
    old_a = safe_float(pos.get("avg_price")) or 0
    new_s = old_s + shares
    # Gebühren auf Einstandswert aufschlagen
    new_a = ((old_s * old_a) + (shares * price) + fees) / new_s if new_s > 0 else price
    pos["shares"]    = round(new_s, 6)
    pos["avg_price"] = round(new_a, 4)
    return pos

def apply_sell(pos: dict, shares: float) -> dict:
    """Verkauf: Anteile reduzieren, Ø-Kurs bleibt (average-cost Methode)."""
    old_s = safe_float(pos.get("shares")) or 0
    pos["shares"] = round(max(0.0, old_s - shares), 6)
    return pos

def add_portfolio_snapshot(port_data: dict) -> dict:
    """Speichert einen Zeitstempel-Snapshot des Gesamtwerts pro Portfolio."""
    today = datetime.now().strftime("%Y-%m-%d")
    for pname, pdata in port_data.items():
        positions  = pdata.get("positions", [])
        total_cv   = sum(calc_position_derived(p)["current_value"] for p in positions)
        total_inv  = sum(calc_position_derived(p)["invested"]       for p in positions)
        if total_cv == 0:
            continue
        pl_abs = total_cv - total_inv
        pl_pct = (pl_abs / total_inv * 100) if total_inv > 0 else 0
        snapshots = pdata.setdefault("snapshots", [])
        entry = {"date": today, "total_value": round(total_cv, 2),
                 "pl_abs": round(pl_abs, 2), "pl_pct": round(pl_pct, 2)}
        if snapshots and snapshots[-1].get("date") == today:
            snapshots[-1] = entry
        else:
            snapshots.append(entry)
        pdata["snapshots"] = snapshots[-365:]
    return port_data

def parse_tr_screenshot(image_bytes, api_key, model="gpt-4o-mini"):
    """Nutzt OpenAI Vision um TR-Portfolio-Screenshot zu lesen."""
    import base64
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    client = OpenAI(api_key=api_key)
    prompt = """Analysiere diesen Trade Republic Portfolio-Screenshot und extrahiere alle sichtbaren Positionen.
Gib NUR ein JSON-Objekt zurück, kein Markdown, keine Erklärungen:
{
  "total_value": 7788.81,
  "positions": [
    {"name": "Core MSCI World USD (Acc)", "current_value": 1360.90, "perf_since_buy_pct": 2.30, "shares": 8.5231},
    {"name": "Schneider Electric", "current_value": 576.16, "perf_since_buy_pct": 8.34, "shares": 3.0}
  ]
}
Hinweise:
- Komma ist deutsches Dezimaltrennzeichen → in Punkt umwandeln
- Verlust-% sind negativ (▼ = negativ)
- total_value nur wenn Gesamtdepotwert sichtbar, sonst null
- shares = Anzahl Anteile/Stück wenn sichtbar, sonst null
- Nur Positionen die klar lesbar sind"""
    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"}
        ]}]
    )
    raw = getattr(resp, "output_text", "").strip()
    # JSON aus Antwort extrahieren
    import re
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if not m: return None
    return json.loads(m.group())

def enrich_position(pos):
    """Berechnet invested, Anteile und Kaufkurs.
    Priorität: 1) Anteile aus TR-Screenshot, 2) Berechnung via Yahoo-Kurs."""
    cv  = pos.get("current_value")
    pct = pos.get("perf_since_buy_pct")
    if cv and pct is not None:
        pos["invested"] = round(cv / (1 + pct / 100), 2)

    ticker = pos.get("ticker")

    # Anteile aus TR-Screenshot direkt übernehmen wenn vorhanden
    screenshot_shares = safe_float(pos.get("shares"))

    if ticker:
        try:
            info = yf.Ticker(ticker).info or {}
            cp = safe_float(info.get("regularMarketPrice") or info.get("currentPrice"))
            pos["current_price"] = cp
            if cp and cv:
                # Screenshot-Anteile haben Priorität, sonst aus Kurs berechnen
                shares = screenshot_shares if screenshot_shares else round(cv / cp, 4)
                pos["shares"] = shares
                pos["avg_price"] = (round(pos["invested"] / shares, 2)
                                    if pos.get("invested") and shares else None)
        except Exception:
            # Kein Yahoo-Kurs — Screenshot-Anteile trotzdem nutzen
            if screenshot_shares and pos.get("invested"):
                pos["shares"]    = screenshot_shares
                pos["avg_price"] = round(pos["invested"] / screenshot_shares, 2)
    elif screenshot_shares and pos.get("invested"):
        # Kein Ticker, aber Anteile aus Screenshot → Kaufkurs berechnen
        pos["shares"]    = screenshot_shares
        pos["avg_price"] = round(pos["invested"] / screenshot_shares, 2)

    pos["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    return pos

def parse_tr_csv_for_costbasis(csv_bytes: bytes) -> dict:
    """
    Parst den TR Steuerübersicht-CSV-Export und berechnet investiertes Kapital pro ISIN.

    Logik:
    - BUY:  abs(Summe) wird aufaddiert  → gesamter Kaufbetrag inkl. Gebühren
    - SELL: Erlöse werden subtrahiert  → vereinfachte Netto-Methode
    - CORPORATE_ACTION / INTEREST werden ignoriert

    Returns: {isin: {"invested": float, "name": str, "has_sells": bool, "buy_count": int}}
    """
    import io

    text = None
    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin-1"]:
        try:
            text = csv_bytes.decode(enc)
            break
        except Exception:
            continue
    if not text:
        return {}

    first_line = text.split("\n")[0]
    sep = ";" if first_line.count(";") >= first_line.count(",") else ","

    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, on_bad_lines="skip")
    except Exception:
        return {}

    df.columns = [c.strip() for c in df.columns]

    def _parse_eur(s):
        if not s or str(s).strip() in ("", "-", "nan", "NaN"):
            return 0.0
        s = str(s).strip().replace("€", "").replace(" ", "").replace("\u202f", "")
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".")
        elif "," in s:
            s = s.replace(",", ".")
        try:
            return float(s)
        except Exception:
            return 0.0

    result = {}
    for _, row in df.iterrows():
        tx   = str(row.get("Transaktionen", "")).strip().upper()
        isin = str(row.get("ISIN", "")).strip()
        name = str(row.get("Name", "")).strip()
        if not isin or isin.upper() in ("NAN", "") or tx not in ("BUY", "SELL"):
            continue
        # Summe = realer Geldfluss inkl. Gebühren; Total ohne Gebühren
        raw = row.get("Summe") or row.get("Total") or "0"
        summe = _parse_eur(raw)

        if isin not in result:
            result[isin] = {"invested": 0.0, "name": name,
                             "has_sells": False, "buy_count": 0}
        if tx == "BUY":
            result[isin]["invested"]  += abs(summe)   # BUY-Summe ist negativ → abs
            result[isin]["buy_count"] += 1
        elif tx == "SELL":
            result[isin]["invested"]   = max(0.0, result[isin]["invested"] - abs(summe))
            result[isin]["has_sells"]  = True

    return {k: v for k, v in result.items() if v["invested"] > 0.1}


@st.cache_data(ttl=86400, show_spinner=False)
def lookup_ticker_from_isin(isin: str) -> str:
    """Sucht den Yahoo-Ticker für eine ISIN. Bevorzugt EUR-Börsen (Xetra > FRA > EU).
    Ergebnis wird 24h gecacht."""
    if not isin:
        return ""
    try:
        results = yf.Search(isin, max_results=10).quotes or []
        # Reihenfolge: EUR-Börsen zuerst, USD-Börsen als Fallback
        eur_exchanges  = ["XETR", "GER", "FRA", "STU", "TDG", "VIE",
                          "PAR", "AMS", "MIL", "BRU", "LIS", "OSL"]
        usd_exchanges  = ["NMS", "NYQ", "PCX", "NGM", "NCM", "ASE"]
        # Erst: EUR-Börse
        for r in results:
            sym = r.get("symbol", "")
            exc = (r.get("exchange", "") or "").upper()
            if sym and len(sym) <= 12 and exc in eur_exchanges:
                return sym
        # Dann: USD-Börse (als Notfall, Währungskorrektur übernimmt refresh)
        for r in results:
            sym = r.get("symbol", "")
            exc = (r.get("exchange", "") or "").upper()
            if sym and len(sym) <= 12 and exc in usd_exchanges:
                return sym
        # Letzter Fallback
        for r in results:
            sym = r.get("symbol", "")
            if sym and len(sym) <= 12 and "." not in sym[-3:]:
                return sym
    except Exception:
        pass
    return ""


@st.cache_data(ttl=3600, show_spinner=False)
def _get_fx_rate(from_ccy: str) -> float:
    """Holt EUR/XXX Wechselkurs von Yahoo. Gecacht für 1h."""
    if from_ccy == "EUR":
        return 1.0
    try:
        pair = f"{from_ccy}EUR=X"
        info = yf.Ticker(pair).info or {}
        rate = safe_float(info.get("regularMarketPrice") or info.get("currentPrice"))
        return rate if rate and rate > 0 else 1.0
    except Exception:
        return 1.0

def parse_vermoegensübersicht_pdf(pdf_bytes: bytes) -> list:
    """
    Parst die TR Vermögensübersicht PDF (offizielle Depotaufstellung).
    Extrahiert pro Position: isin, name, shares, current_price, current_value.

    Strategie: ISIN-basiertes Parsing.
    pypdf klebt Kurswert und Stückzahl der nächsten Position ohne Leerzeichen
    zusammen (z.B. "45,332,056256 Stk."). Deshalb wird die Stückzahl stabil
    aus Kurswert ÷ Kurs berechnet statt aus dem Rohtext geparst.
    """
    import io, re
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        full_text = " ".join(p.extract_text() or "" for p in reader.pages)
    except ImportError:
        return [{"_error": "pypdf nicht installiert — bitte: pip install pypdf"}]
    except Exception as e:
        return [{"_error": str(e)}]

    def _de_float(s):
        s = str(s).strip().replace("\u202f", "").replace("\xa0", "").replace(" ", "")
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".")
        elif "," in s:
            s = s.replace(",", ".")
        try:
            return float(s)
        except Exception:
            return None

    _NOISE_PAT = (
        r'Bearer Shares.*'
        r'|Inhaber-Aktien.*'
        r'|Registered Shares.*'
        r'|Aandelen op naam.*'
        r'|Actions Port\..*'
        r'|Azioni nom\..*'
        r'|Namn-Aktier.*'
        r'|Reg\.\s*Shares.*'
        r'|vink\.Namens-Aktien.*'
        r'|Namens-Aktien.*'
    )

    isin_matches = list(re.finditer(r'ISIN:\s*([A-Z]{2}[A-Z0-9]{10})', full_text))
    positions = []

    for idx, m_isin in enumerate(isin_matches):
        isin = m_isin.group(1)

        # ── Kurs + Datum + Kurswert (block_after der ISIN) ──────────────────
        end = isin_matches[idx + 1].start() if idx + 1 < len(isin_matches) else len(full_text)
        block_after = full_text[m_isin.end():end]

        # Format: {price,2dec} [whitespace] {TT.MM.JJJJ} [whitespace] {value,2dec}
        # pypdf-Versionen liefern teils Newlines, teils keine Trennzeichen
        m_pv = re.search(
            r'([\d\.]+,\d{2})\s*(\d{2}\.\d{2}\.\d{4})\s*([\d\.]+,\d{2})',
            block_after
        )
        if not m_pv:
            continue
        price       = _de_float(m_pv.group(1))
        total_value = _de_float(m_pv.group(3))

        # ── Stückzahl = Kurswert ÷ Kurs (robust, da Text uneindeutig) ───────
        if price and price > 0 and total_value:
            shares = round(total_value / price, 6)
        else:
            continue
        if shares <= 0 or shares > 100_000:
            continue

        # ── Name (block_before: von Ende letzter ISIN bis aktuelle ISIN) ────
        start = isin_matches[idx - 1].end() if idx > 0 else 0
        block_before = full_text[start:m_isin.start()]

        # Letztes "Stk." im Block → danach kommt der Name dieser Position
        m_stk_last = None
        for m in re.finditer(r'[\d\.]+,\d+\s*Stk\.', block_before):
            m_stk_last = m
        if m_stk_last:
            name_raw = block_before[m_stk_last.end():].strip()
        else:
            dates = list(re.finditer(r'\d{2}\.\d{2}\.\d{4}', block_before))
            name_raw = block_before[dates[-1].end():].strip() if dates else block_before.strip()

        name_clean = re.sub(_NOISE_PAT, '', name_raw, flags=re.IGNORECASE)
        name_clean = re.sub(r'Lagerland:.*', '', name_clean, flags=re.IGNORECASE)
        name_clean = re.sub(r'\s+', ' ', name_clean).strip(' ,')
        # Nachgestellten Nennwert-Rest entfernen (z.B. ",04")
        name_clean = re.sub(r',\d+$', '', name_clean).strip(' ,')

        positions.append({
            "isin":          isin,
            "name":          name_clean[:60],
            "shares":        shares,
            "current_price": price,
            "current_value": total_value,
        })

    return positions


# ══════════════════════════════════════════════════════════════════════════════
# Yahoo
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_history(ticker, period="9mo", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False, threads=False)
    except Exception: return pd.DataFrame()
    if df is None or df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_profile(ticker):
    out = {"ok": False, "profile": {"name":"","sector":"","industry":"","summary":"",
                                    "country":"","currency":"","exchange":""}, "errors":[]}
    if not ticker: out["errors"].append("Kein Ticker."); return out
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as e: out["errors"].append(str(e)); return out
    p = {"name": info.get("longName") or info.get("shortName") or ticker,
         "sector": info.get("sector") or "", "industry": info.get("industry") or "",
         "summary": info.get("longBusinessSummary") or "",
         "country": info.get("country") or "", "currency": info.get("currency") or "",
         "exchange": info.get("exchange") or ""}
    s = sum([bool(p["sector"]), bool(p["industry"]), len(p["summary"]) >= 80])
    out["ok"] = s >= 2; out["profile"] = p
    if not out["ok"] and not out["errors"]:
        out["errors"].append("Dünnes Profil → Story wird nicht bewertet.")
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def translate_headlines(titles_tuple, api_key, model="gpt-4.1-mini"):
    """Übersetzt englische Schlagzeilen in einem Batch-Call ins Deutsche."""
    if not api_key or not OPENAI_AVAILABLE: return list(titles_tuple)
    try:
        import re
        client = OpenAI(api_key=api_key)
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles_tuple))
        resp = client.responses.create(model=model, input=[
            {"role": "system", "content":
             "Übersetze diese Finanz-Schlagzeilen präzise ins Deutsche. "
             "Gib nur die nummerierten Übersetzungen zurück, eine pro Zeile, ohne Erklärungen."},
            {"role": "user", "content": numbered},
        ])
        lines = getattr(resp, "output_text", "").strip().split("\n")
        result = [re.sub(r"^\d+\.\s*", "", l.strip()) for l in lines if l.strip()]
        return result if len(result) == len(titles_tuple) else list(titles_tuple)
    except Exception:
        return list(titles_tuple)

def _clean_search_query(name: str) -> str:
    """Bereinigt Fondsnamen für Yahoo-Suche: Klammern, Währungs- und Fonds-Suffixe entfernen."""
    import re
    # Klammern inkl. Inhalt weg: "(Acc)", "(USD)", "(thes.)", ...
    name = re.sub(r'\s*\(.*?\)', '', name)
    # Bekannte Suffixe/Präfixe die Yahoo nicht kennt
    noise = r'\b(UCITS|ETF|USD|EUR|GBP|CHF|Acc|Dist|Thes|Aus|Inc|Cap|hedged|unhedged|Core)\b'
    name = re.sub(noise, '', name, flags=re.IGNORECASE)
    # Mehrfache Leerzeichen, Bindestriche am Rand
    name = re.sub(r'\s+', ' ', name).strip(' -·')
    return name

@st.cache_data(ttl=3600, show_spinner=False)
def search_yahoo_ticker(query):
    """Sucht Ticker auf Yahoo Finance — robuster mit mehreren Fallbacks."""
    import re

    def _parse_quotes(quotes):
        out = []
        for r in (quotes or []):
            sym = r.get("symbol","")
            if not sym: continue
            out.append({
                "symbol":   sym,
                "name":     r.get("shortname") or r.get("longname") or sym,
                "exchange": r.get("exchange",""),
                "type":     r.get("quoteType",""),
            })
        return out

    def _run(q):
        # Methode 1: yf.Search
        try:
            sr = yf.Search(q, max_results=8)
            quotes = getattr(sr, "quotes", None) or []
            if quotes:
                return _parse_quotes(quotes)
        except Exception:
            pass
        # Methode 2: Direkte Yahoo Finance API via requests
        try:
            import requests as _req
            url = (f"https://query1.finance.yahoo.com/v1/finance/search"
                   f"?q={_req.utils.quote(q)}&quotesCount=8&newsCount=0"
                   f"&enableFuzzyQuery=true&enableCb=false")
            headers = {"User-Agent": "Mozilla/5.0 (compatible)"}
            resp = _req.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                quotes = data.get("quotes", [])
                if quotes:
                    return _parse_quotes(quotes)
        except Exception:
            pass
        # Methode 3: Ticker-Direktlookup (z.B. "AAPL")
        try:
            q_upper = q.upper().strip()
            if re.match(r'^[A-Z0-9.^=-]{1,10}$', q_upper):
                info = yf.Ticker(q_upper).info or {}
                nm = info.get("longName") or info.get("shortName") or ""
                if nm:
                    return [{"symbol": q_upper, "name": nm,
                             "exchange": info.get("exchange",""),
                             "type": info.get("quoteType","")}]
        except Exception:
            pass
        return []

    results = _run(query)
    if not results:
        cleaned = _clean_search_query(query)
        if cleaned and cleaned.lower() != query.lower():
            results = _run(cleaned)
    # Duplikate raus (gleicher symbol)
    seen, deduped = set(), []
    for r in results:
        if r["symbol"] not in seen:
            seen.add(r["symbol"]); deduped.append(r)
    return deduped

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news(ticker, max_items=8, _refresh: int = 0):
    if not ticker: return []
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception: return []
    from datetime import timezone
    out = []
    for item in raw:
        content = item.get("content") or {}
        title   = content.get("title") or item.get("title") or ""
        pub     = (content.get("provider") or {}).get("displayName") or item.get("publisher") or ""
        ts      = content.get("pubDate") or ""
        url     = (content.get("canonicalUrl") or {}).get("url") or item.get("link") or ""
        if not title: continue
        # Timestamp als sortierbares Objekt behalten
        dt_obj  = None
        date_str = ""
        if ts:
            try:
                dt_obj   = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                date_str = dt_obj.strftime("%d.%m.%Y")
            except Exception:
                date_str = str(ts)[:10]
        elif item.get("providerPublishTime"):
            try:
                dt_obj   = datetime.fromtimestamp(item["providerPublishTime"], tz=timezone.utc)
                date_str = dt_obj.strftime("%d.%m.%Y")
            except Exception: pass
        out.append({"title": title, "publisher": pub, "date": date_str,
                    "url": url, "_dt": dt_obj})
    # Neueste zuerst — None ans Ende
    out.sort(key=lambda x: x["_dt"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return out[:max_items]

# Keyword-basiertes Sentiment (kein API-Key nötig)
_POS_WORDS = {"beat", "record", "strong", "growth", "profit", "upgrade", "buy",
              "bullish", "surge", "rally", "soars", "rises", "gains", "outperform",
              "raised", "positive", "dividend", "acquisition", "expands", "launches",
              "höchst", "rekord", "wächst", "gewinnt", "steigt", "stark", "erhöht"}
_NEG_WORDS = {"miss", "cut", "loss", "decline", "downgrade", "sell", "bearish",
              "plunges", "drops", "falls", "weak", "warning", "concern", "risk",
              "lawsuit", "investigation", "fraud", "debt", "layoffs", "recall",
              "verlust", "sinkt", "schwach", "warnung", "rückgang", "krise"}

def score_news_sentiment(titles: list) -> dict:
    """Einfaches Keyword-Sentiment ohne API. Gibt pos/neg/neu Counts + Label zurück."""
    pos = neg = 0
    for t in titles:
        low = t.lower()
        words = set(low.replace(",","").replace(".","").split())
        if words & _POS_WORDS: pos += 1
        elif words & _NEG_WORDS: neg += 1
    neu = len(titles) - pos - neg
    total = len(titles) or 1
    if pos / total >= 0.5:
        label, color = "Überwiegend positiv", "#00C864"
    elif neg / total >= 0.5:
        label, color = "Überwiegend negativ", "#FF4444"
    elif neg > pos:
        label, color = "Leicht negativ", "#FFA500"
    elif pos > neg:
        label, color = "Leicht positiv", "#26a69a"
    else:
        label, color = "Gemischt / neutral", "#9e9e9e"
    return {"label": label, "color": color, "pos": pos, "neg": neg, "neu": neu}

def gpt_news_summary(titles: list, api_key: str, model="gpt-4.1-mini") -> str:
    """1-2 Sätze Einordnung der Nachrichtenlage via GPT."""
    if not titles or not api_key: return ""
    try:
        client = OpenAI(api_key=api_key)
        joined = "\n".join(f"- {t}" for t in titles[:6])
        resp = client.responses.create(model=model, input=[{
            "role": "system",
            "content": ("Du bist ein prägnanter Finanzanalyst. Fasse die Nachrichtenlage "
                        "zur Aktie in 1-2 knappen Sätzen ein. Kein Bullshit, direkte Einschätzung. "
                        "Auf Deutsch.")
        }, {
            "role": "user",
            "content": f"Aktuelle Schlagzeilen:\n{joined}"
        }])
        return (resp.output_text or "").strip()
    except Exception:
        return ""

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_metrics(ticker):
    if not ticker: return {}
    try: info = yf.Ticker(ticker).info or {}
    except Exception: return {}
    mcap_r = safe_float(info.get("marketCap"))
    shares_r = safe_float(info.get("sharesOutstanding"))
    div_r = safe_float(info.get("dividendYield"))
    # Sanity-Check: yfinance liefert manchmal den Wert bereits als Prozentzahl (z.B. 0.91)
    # statt als Dezimal (0.0091). Wenn > 0.20 → schon in Prozent → nicht nochmal * 100
    if div_r and div_r > 0.20:
        div_yield_pct = round(div_r, 2)          # bereits Prozent (z.B. 0.91 → 0.91%)
    elif div_r:
        div_yield_pct = round(div_r * 100, 2)    # Dezimal → Prozent (z.B. 0.0091 → 0.91%)
    else:
        div_yield_pct = None
    return {
        "beta": safe_float(info.get("beta")),
        "pe": safe_float(info.get("trailingPE")),
        "peg": safe_float(info.get("pegRatio")),
        "ps": safe_float(info.get("priceToSalesTrailing12Months")),
        "pb": safe_float(info.get("priceToBook")),
        "div_yield": div_yield_pct,
        "mcap": round(mcap_r / 1e9, 2) if mcap_r else None,
        "shares": round(shares_r / 1e6, 2) if shares_r else None,
        "high52": safe_float(info.get("fiftyTwoWeekHigh")),
        "low52": safe_float(info.get("fiftyTwoWeekLow")),
        "currency": info.get("currency", ""),
    }

# ── Sektor-Benchmarks für relative Bewertung ─────────────────────────────────
SECTOR_BENCHMARKS: dict = {
    "Technology":             {"pe": 28, "pb": 8.0, "ps": 5.0},
    "Healthcare":             {"pe": 22, "pb": 4.0, "ps": 3.5},
    "Consumer Defensive":     {"pe": 19, "pb": 3.5, "ps": 1.5},
    "Consumer Cyclical":      {"pe": 21, "pb": 3.5, "ps": 1.2},
    "Industrials":            {"pe": 19, "pb": 3.0, "ps": 1.5},
    "Financial Services":     {"pe": 13, "pb": 1.4, "ps": 2.5},
    "Energy":                 {"pe": 11, "pb": 1.8, "ps": 0.9},
    "Utilities":              {"pe": 17, "pb": 1.7, "ps": 2.2},
    "Basic Materials":        {"pe": 14, "pb": 2.2, "ps": 1.2},
    "Real Estate":            {"pe": 28, "pb": 2.0, "ps": 5.5},
    "Communication Services": {"pe": 21, "pb": 3.8, "ps": 2.8},
}

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_short_desc(ticker: str) -> str:
    """Kurze Unternehmensbeschreibung (max ~120 Zeichen) — 24h gecacht."""
    if not ticker: return ""
    try:
        info = (yf.Ticker(ticker).info or {})
        summary = info.get("longBusinessSummary") or ""
        sector  = info.get("sector") or ""
        industry = info.get("industry") or ""
        if summary:
            # Ersten Satz nehmen, auf ~120 Zeichen kürzen
            first = summary.split(".")[0].strip()
            if len(first) > 115: first = first[:115] + "…"
            return first
        if sector and industry:
            return f"{sector} · {industry}"
        return sector or ""
    except Exception:
        return ""

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_extended_metrics(ticker: str) -> dict:
    """FCF, Verschuldung, Margen, Umsatzwachstum — erweiterte Fundamentaldaten."""
    if not ticker: return {}
    try:
        obj  = yf.Ticker(ticker)
        info = obj.info or {}

        mcap = safe_float(info.get("marketCap"))

        # ── FCF Yield ──────────────────────────────────────────────────────
        fcf = safe_float(info.get("freeCashflow"))
        fcf_yield = round(fcf / mcap * 100, 2) if fcf and mcap and mcap > 0 else None

        # ── Debt / EBITDA ─────────────────────────────────────────────────
        total_debt = safe_float(info.get("totalDebt"))
        ebitda     = safe_float(info.get("ebitda"))
        debt_ebitda = round(total_debt / ebitda, 2) if (
            total_debt is not None and ebitda and ebitda > 0) else None

        # ── Operating Cash Flow / Market Cap ──────────────────────────────
        op_cf = safe_float(info.get("operatingCashflow"))
        op_cf_ratio = round(op_cf / mcap * 100, 2) if op_cf and mcap and mcap > 0 else None

        # ── Margen ────────────────────────────────────────────────────────
        gross_margin = safe_float(info.get("grossMargins"))
        op_margin    = safe_float(info.get("operatingMargins"))
        net_margin   = safe_float(info.get("profitMargins"))
        roe          = safe_float(info.get("returnOnEquity"))

        # ── Wachstum aus info ─────────────────────────────────────────────
        rev_growth   = safe_float(info.get("revenueGrowth"))    # YoY decimal
        earn_growth  = safe_float(info.get("earningsGrowth"))   # YoY decimal

        # ── 3-Jahres Umsatz-CAGR aus Jahresabschlüssen ───────────────────
        rev_cagr_3y    = None
        earnings_years = None
        try:
            fin = None
            try: fin = obj.income_stmt       # newer yfinance
            except Exception: pass
            if fin is None or (hasattr(fin, "empty") and fin.empty):
                try: fin = obj.financials    # older yfinance
                except Exception: pass
            if fin is not None and hasattr(fin, "empty") and not fin.empty:
                # Umsatz-Zeile suchen
                _rev_keys = ["Total Revenue", "TotalRevenue", "Revenue"]
                rev_row = None
                for _k in _rev_keys:
                    if _k in fin.index:
                        rev_row = fin.loc[_k]; break
                if rev_row is not None:
                    revs = rev_row.dropna().sort_index(ascending=False).values
                    if len(revs) >= 3 and revs[-1] > 0:
                        n = len(revs) - 1
                        rev_cagr_3y = round(((revs[0] / revs[-1]) ** (1/n) - 1) * 100, 1)
                    # Konsistenz: wie viele Jahre Umsatzwachstum
                    if len(revs) >= 2:
                        growing = sum(1 for i in range(len(revs)-1) if revs[i] > revs[i+1])
                        earnings_years = growing  # 0-3 years growing
        except Exception:
            pass

        return {
            "fcf_yield":      fcf_yield,      # % of market cap
            "debt_ebitda":    debt_ebitda,     # x times
            "op_cf_ratio":    op_cf_ratio,     # % of market cap
            "gross_margin":   round(gross_margin * 100, 1) if gross_margin else None,
            "op_margin":      round(op_margin   * 100, 1) if op_margin   else None,
            "net_margin":     round(net_margin  * 100, 1) if net_margin  else None,
            "roe":            round(roe         * 100, 1) if roe          else None,
            "rev_growth_yoy": round(rev_growth  * 100, 1) if rev_growth  else None,
            "earn_growth_yoy":round(earn_growth * 100, 1) if earn_growth else None,
            "rev_cagr_3y":    rev_cagr_3y,     # % CAGR
            "earnings_years": earnings_years,  # int: years with revenue growth
        }
    except Exception:
        return {}

def watchlist_quick_check(ticker: str, mode: str) -> dict:
    """Schneller Fundamental-Score-Check für die Watchlist — kein Cache, immer frisch."""
    try:
        basic   = fetch_yahoo_metrics(ticker) or {}
        ext     = fetch_extended_metrics(ticker) or {}
        metrics = {**basic, **{k: v for k, v in ext.items() if v is not None}}
        if not metrics: return {"ok": False}

        score_fn = score_core_fundamentals if mode == "Core Asset" else score_hc_fundamentals
        fs, fr   = score_fn(metrics)

        # Relative Bewertung
        prof   = fetch_yahoo_profile(ticker)
        sector = (prof.get("profile") or {}).get("sector", "")
        rd, _  = score_relative_valuation(metrics, sector)
        if fs and rd: fs = clip_score(fs + rd)

        return {
            "ok":   True,
            "fund": fs,
            "ts":   datetime.now().strftime("%H:%M"),
            "top_reason": fr[0] if fr else "",
        }
    except Exception:
        return {"ok": False}

@st.cache_data(ttl=45, show_spinner=False)
def fetch_price_now(ticker: str) -> dict:
    """Aktueller Kurs via fast_info — ~45s Cache, nahezu live."""
    if not ticker: return {"ok": False}
    try:
        obj = yf.Ticker(ticker)

        # fast_info ist deutlich aktueller als .info["regularMarketPrice"]
        fi       = obj.fast_info
        price    = safe_float(getattr(fi, "last_price", None))
        prev     = safe_float(getattr(fi, "previous_close", None))
        currency = (getattr(fi, "currency", None) or "").upper()

        # Fallback auf .info wenn fast_info leer
        if not price:
            info  = obj.info or {}
            price = safe_float(info.get("regularMarketPrice") or info.get("currentPrice"))
            prev  = prev or safe_float(info.get("regularMarketPreviousClose"))
            currency = currency or (info.get("currency") or "").upper()

        exchange = (getattr(fi, "exchange", None) or "").upper()
        if not price: return {"ok": False}

        # EUR-Umrechnung via fast_info (ebenfalls aktueller)
        price_eur = price
        fx_note   = ""
        if currency and currency != "EUR":
            try:
                fx_fi = yf.Ticker(f"{currency}EUR=X").fast_info
                fx    = safe_float(getattr(fx_fi, "last_price", None))
                if not fx:
                    fx = safe_float(yf.Ticker(f"{currency}EUR=X").info.get("regularMarketPrice"))
                if fx and fx > 0:
                    price_eur = price * fx
                    fx_note   = f"aus {currency}"
            except Exception:
                fx_note = f"in {currency}"

        chg_pct = round((price - prev) / prev * 100, 2) if prev and prev > 0 else None
        chg_abs = round(price_eur - (prev * price_eur / price), 3) if prev and price else None
        return {
            "ok":        True,
            "price_eur": price_eur,
            "price_nat": price,
            "currency":  currency,
            "exchange":  exchange,
            "chg_abs":   chg_abs,
            "chg_pct":   chg_pct,
            "fx_note":   fx_note,
        }
    except Exception:
        return {"ok": False}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_sector_cached(ticker: str) -> str:
    """Sektor für einen Ticker — gecacht 1 Stunde."""
    if not ticker: return ""
    try:
        info = yf.Ticker(ticker).info or {}
        return info.get("sector") or ""
    except Exception:
        return ""

@st.cache_data(ttl=3600, show_spinner=False)
def _card_full_data(ticker: str, mode: str) -> dict:
    """Holt Metriken + Scores für eine Radar-Card — cached 1 Stunde."""
    try:
        obj  = yf.Ticker(ticker)
        info = obj.info or {}
        mcap_r = safe_float(info.get("marketCap"))
        div_r  = safe_float(info.get("dividendYield"))
        _div_pct = (round(div_r, 2) if div_r and div_r > 0.20
                    else round(div_r * 100, 2) if div_r else None)
        metrics = {
            "beta":          safe_float(info.get("beta")),
            "pe":            safe_float(info.get("trailingPE")),
            "peg":           safe_float(info.get("pegRatio")),
            "ps":            safe_float(info.get("priceToSalesTrailing12Months")),
            "pb":            safe_float(info.get("priceToBook")),
            "div_yield":     _div_pct,
            "mcap":          round(mcap_r / 1e9, 2) if mcap_r else None,
            "roe":           safe_float(info.get("returnOnEquity")),
            "roic":          safe_float(info.get("returnOnAssets")),
            "gross_margin":  safe_float(info.get("grossMargins")),
            "op_margin":     safe_float(info.get("operatingMargins")),
            "net_margin":    safe_float(info.get("netMargins")),
            "debt_eq":       safe_float(info.get("debtToEquity")),
            "rev_growth":    safe_float(info.get("revenueGrowth")),
            "earn_growth":   safe_float(info.get("earningsGrowth")),
            "current_ratio": safe_float(info.get("currentRatio")),
        }
        profile = {
            "name":     info.get("longName") or info.get("shortName") or ticker,
            "sector":   info.get("sector", ""),
            "industry": info.get("industry", ""),
            "summary":  info.get("longBusinessSummary", ""),
        }
        if mode == "Core Asset":
            fs, _ = score_core_fundamentals(metrics)
        else:
            fs, _ = score_hc_fundamentals(metrics)
        si = classify_business_profile(profile, metrics)
        ss = si.get("core_fit" if mode == "Core Asset" else "hc_fit") if si else None
        bm = si.get("business_model", "") if si else profile["sector"]
        # Kurze Firmenbeschreibung (erste 2 Sätze aus Yahoo)
        _raw_sum = profile.get("summary", "")
        _sentences = [s.strip() for s in _raw_sum.replace("  ", " ").split(". ")
                      if len(s.strip()) > 20]
        _short_desc = ". ".join(_sentences[:2]).strip()
        if _short_desc and not _short_desc.endswith("."):
            _short_desc += "."
        return {"fund": fs, "story": ss, "bm": bm, "desc": _short_desc, "ok": True}
    except Exception:
        return {"ok": False}

def _card_score_color(val, is_hc: bool = False):
    if val is None: return "rgba(180,180,200,0.35)"
    if val >= 7.5:  return "#10b981"
    if val >= 6.0:  return "#f59e0b"
    # HC stocks floor at amber — low fundamentals ≠ bad pick for growth mode
    if is_hc:       return "#a78bfa"   # violet: potential, not failure
    return "#ef4444"

def _card_bg(val, is_dark: bool = True, is_hc: bool = False):
    """Gradient-Farben für Card-Hintergrund basierend auf Score und Theme."""
    if is_dark:
        if val is None: return "#0d1117", "#161b26"
        if val >= 7.5:  return "#041510", "#07201a"
        if val >= 6.0:  return "#130e00", "#1e1600"
        if is_hc:       return "#0e0a1a", "#150f24"  # purple-tinted, not red
        return "#130404", "#1f0808"
    else:
        if val is None: return "#f7f8fc", "#edf0f7"
        if val >= 7.5:  return "#edfbf4", "#daf5e8"
        if val >= 6.0:  return "#fdfaf0", "#f8f2d8"
        if is_hc:       return "#f5f3ff", "#ede9fe"  # lavender tint, not red
        return "#fdf5f5", "#f8e8e8"

def render_radar_card(tk: str, name: str, why: str, data: dict, idx, mode: str,
                      is_dark: bool = True, show_cta: bool = True):
    """Rendert eine Premium Velox Radar-Card mit SVG Score-Ring."""
    fs    = data.get("fund")
    ss    = data.get("story")
    bm    = data.get("bm", "")
    desc  = data.get("desc", "")
    avail = [s for s in [fs, ss] if s is not None]
    total = round(sum(avail) / len(avail), 1) if avail else None

    _is_hc   = (mode != "Core Asset")   # HC & alle anderen = Potenzial-Modus
    sc       = _card_score_color(total, is_hc=_is_hc)
    bg1, bg2 = _card_bg(total, is_dark, is_hc=_is_hc)
    fs_c     = _card_score_color(fs, is_hc=_is_hc)
    ss_c     = _card_score_color(ss, is_hc=_is_hc)

    # ── Theme colours ────────────────────────────────────────────────────────
    if is_dark:
        t0         = "rgba(255,255,255,0.96)"
        t1         = "rgba(255,255,255,0.60)"
        t2         = "rgba(255,255,255,0.30)"
        sep        = "rgba(255,255,255,0.07)"
        trk        = "rgba(255,255,255,0.10)"
        shine      = "rgba(255,255,255,0.025)"
        card_extra = "inset 0 1px 0 rgba(255,255,255,0.06)"
    else:
        t0         = "rgba(8,8,12,0.92)"
        t1         = "rgba(20,20,30,0.58)"
        t2         = "rgba(20,20,30,0.38)"
        sep        = "rgba(0,0,0,0.08)"
        trk        = "rgba(0,0,0,0.11)"
        shine      = "rgba(255,255,255,0.55)"
        card_extra = "inset 0 1px 0 rgba(255,255,255,0.9)"

    # ── Score strings ────────────────────────────────────────────────────────
    total_str = f"{total:.1f}" if total is not None else "—"
    fs_str    = f"{fs:.1f}"    if fs    is not None else "—"
    ss_str    = f"{ss:.1f}"    if ss    is not None else "—"
    fs_pct    = int((fs    or 0) * 10)
    ss_pct    = int((ss    or 0) * 10)

    # ── SVG Ring ─────────────────────────────────────────────────────────────
    r     = 36
    cx    = cy = 44
    circ  = round(2 * 3.14159265 * r, 2)   # 226.19
    filled = round((total or 0) / 10 * circ, 2)
    gaplen = round(max(circ - filled, 0), 2)

    # ── Mode badge ───────────────────────────────────────────────────────────
    if mode == "Core Asset":
        mc = "#3b82f6"
        mb = "rgba(59,130,246,0.13)" if is_dark else "rgba(59,130,246,0.09)"
    else:
        mc = "#8b5cf6"
        mb = "rgba(139,92,246,0.13)" if is_dark else "rgba(139,92,246,0.09)"

    # ── Shadow / glow ────────────────────────────────────────────────────────
    if is_dark:
        if total and total >= 7.5:
            shadow = (f"0 0 40px {sc}44,0 0 18px {sc}22,"
                      f"0 8px 32px rgba(0,0,0,0.65),{card_extra}")
        elif total and total >= 6.0:
            shadow = (f"0 0 20px {sc}28,0 6px 24px rgba(0,0,0,0.58),{card_extra}")
        else:
            shadow = f"0 6px 24px rgba(0,0,0,0.52),{card_extra}"
    else:
        shadow = (f"0 2px 18px rgba(0,0,0,0.08),0 0 0 1px {sep},{card_extra}")

    bm_short   = (bm[:30]   + "…") if len(bm)   > 30  else bm
    why_short  = (why[:135] + "…") if len(why)  > 135 else why
    name_short = (name[:26] + "…") if len(name) > 26  else name

    card_html = (
        # ── Outer card ───────────────────────────────────────────────────────
        '<div style="'
        f'background:linear-gradient(148deg,{bg1} 0%,{bg2} 55%,{bg1}f5 100%);'
        'border-radius:18px;'
        f'border-top:3px solid {sc};'
        f'border-left:1px solid {sc}1a;'
        f'border-right:1px solid {sep};'
        f'border-bottom:1px solid {sep};'
        f'box-shadow:{shadow};'
        'padding:1.15rem 1.1rem 1.05rem 1.1rem;'
        'position:relative;overflow:hidden;'
        'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;">'

        # Radial shine highlight
        f'<div style="position:absolute;top:-55%;left:-15%;width:130%;height:160%;'
        f'background:radial-gradient(ellipse at 28% 18%,{shine} 0%,transparent 62%);'
        'pointer-events:none;"></div>'

        # ── Top row: ring  +  info ────────────────────────────────────────────
        '<div style="display:flex;align-items:center;gap:0.85rem;">'

        # SVG ring
        '<div style="flex-shrink:0;position:relative;width:88px;height:88px;">'
        f'<svg width="88" height="88" viewBox="0 0 88 88" '
        'style="transform:rotate(-90deg);display:block;">'
        # Track
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" '
        f'stroke="{trk}" stroke-width="5.5"/>'
        # Progress arc
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" '
        f'stroke="{sc}" stroke-width="5.5" stroke-linecap="round" '
        f'stroke-dasharray="{filled} {gaplen}"/>'
        '</svg>'
        # Score label centred over ring
        f'<div style="position:absolute;inset:0;display:flex;'
        'flex-direction:column;align-items:center;justify-content:center;">'
        f'<div style="font-size:1.7rem;font-weight:800;color:{sc};'
        'line-height:1;letter-spacing:-0.03em;'
        f'text-shadow:{"0 0 12px " + sc + "55" if is_dark else "none"};">'
        f'{total_str}</div>'
        f'<div style="font-size:0.4rem;letter-spacing:0.22em;color:{t2};'
        'text-transform:uppercase;margin-top:3px;">VELOX</div>'
        '</div></div>'  # end ring

        # Info column
        f'<div style="flex:1;min-width:0;">'
        # Mode badge pill
        f'<div style="display:inline-flex;align-items:center;gap:4px;'
        f'font-size:0.5rem;letter-spacing:0.16em;text-transform:uppercase;'
        f'color:{mc};background:{mb};border:1px solid {mc}35;'
        'border-radius:30px;padding:3px 9px;margin-bottom:0.38rem;">'
        f'<span style="width:5px;height:5px;border-radius:50%;'
        f'background:{mc};display:inline-block;flex-shrink:0;"></span>'
        f'{mode}</div>'
        # Ticker
        f'<div style="font-size:1.22rem;font-weight:800;color:{t0};'
        'letter-spacing:0.05em;line-height:1;">'
        f'{tk}</div>'
        # Name
        f'<div style="font-size:0.72rem;color:{t1};margin-top:0.18rem;'
        'line-height:1.35;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
        f'{name_short}</div>'
        # BM tag
        f'<div style="font-size:0.56rem;color:{t2};margin-top:0.32rem;'
        'text-transform:uppercase;letter-spacing:0.12em;'
        'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
        f'{bm_short}</div>'
        '</div>'  # end info

        '</div>'  # end top row

        # ── Firmenbeschreibung (wenn verfügbar) ───────────────────────────────
        + (f'<div style="font-size:0.72rem;color:{t1};line-height:1.55;'
           f'margin-top:0.6rem;padding-top:0.5rem;border-top:1px solid {sep};">'
           f'{desc[:220] + "…" if len(desc) > 220 else desc}'
           f'</div>'
           if desc else '')
        +

        # ── Sub-scores ────────────────────────────────────────────────────────
        f'<div style="margin-top:0.8rem;padding-top:0.7rem;'
        f'border-top:1px solid {sep};">'

        # Fundament
        '<div style="margin-bottom:0.55rem;">'
        '<div style="display:flex;justify-content:space-between;'
        'align-items:baseline;margin-bottom:0.28rem;">'
        f'<span style="font-size:0.58rem;letter-spacing:0.13em;'
        f'text-transform:uppercase;color:{t2};">Fundament</span>'
        f'<span style="font-size:0.88rem;font-weight:700;color:{fs_c};">'
        f'{fs_str}</span></div>'
        f'<div style="height:3px;background:{trk};border-radius:2px;overflow:hidden;">'
        f'<div style="width:{fs_pct}%;height:100%;border-radius:2px;'
        f'background:linear-gradient(90deg,{fs_c}88,{fs_c});"></div>'
        '</div></div>'

        # Story-Fit
        '<div>'
        '<div style="display:flex;justify-content:space-between;'
        'align-items:baseline;margin-bottom:0.28rem;">'
        f'<span style="font-size:0.58rem;letter-spacing:0.13em;'
        f'text-transform:uppercase;color:{t2};">Story-Fit</span>'
        f'<span style="font-size:0.88rem;font-weight:700;color:{ss_c};">'
        f'{ss_str}</span></div>'
        f'<div style="height:3px;background:{trk};border-radius:2px;overflow:hidden;">'
        f'<div style="width:{ss_pct}%;height:100%;border-radius:2px;'
        f'background:linear-gradient(90deg,{ss_c}88,{ss_c});"></div>'
        '</div></div>'

        '</div>'  # end sub-scores

        # ── Why text ──────────────────────────────────────────────────────────
        f'<div style="margin-top:0.75rem;padding-top:0.65rem;'
        f'border-top:1px solid {sep};'
        f'font-size:0.7rem;color:{t1};line-height:1.65;">'
        f'<span style="color:{sc};opacity:0.7;margin-right:4px;'
        'font-size:0.8rem;">▸</span>'
        f'{why_short}</div>'

        '</div>'  # end card
    )
    st.markdown(card_html, unsafe_allow_html=True)

    if not show_cta:
        return  # Caller rendert eigene Buttons

    # CTA button — Ticker + Name ins Eingabefeld, Tab wechseln, fertig
    if st.button(f"▶  Vollanalyse  ·  {tk}", key=f"rc_{tk}_{idx}",
                 use_container_width=True,
                 help=f"{name} analysieren"):
        # Metrics vorladen (gecacht → Felder sofort ausgefüllt wenn User klickt)
        with st.spinner(f"Lade {tk}…"):
            _pm = fetch_yahoo_metrics(tk)
            _px = fetch_extended_metrics(tk)
        if _pm:
            st.session_state["ace_yf_metrics"]  = _pm
            st.session_state["ace_ext_metrics"] = _px or {}
            st.session_state["ace_yf_ticker"]   = tk
        # Ticker + Name setzen, Widget-Keys zurücksetzen
        st.session_state["ace_selected_ticker"]     = tk
        st.session_state["ace_search_q"]            = name
        st.session_state["ace_selected_name"]       = name
        st.session_state["_auto_switch_to_analyse"] = True
        for _k in ("ace_search_results","story_info","fund_score",
                   "timing_score","story_score","chart_df","ace_long_fazit",
                   "red_flags","entry_triggers","risk_hints",
                   "ace_direct_ticker","ace_search_input",
                   "auto_run_fund"):   # kein Auto-Run — User entscheidet
            st.session_state.pop(_k, None)
        # A8: show_radar NICHT zurücksetzen — Radar bleibt für weiteres Stöbern offen
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# Technicals
# ══════════════════════════════════════════════════════════════════════════════
def compute_rsi(series, window=14):
    """Wilder's RSI — EWM mit alpha=1/window (Standard-Methode, glatter als SMA-RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def analyze_trend_structure(df):
    reasons = []; score_delta = 0.0
    if df is None or df.empty or len(df) < 60: return score_delta, reasons
    last = df.iloc[-1]; last5 = df.tail(5); last20 = df.tail(20)
    lc = safe_float(last.get("Close")); ma20 = safe_float(last.get("MA20"))
    ma50 = safe_float(last.get("MA50")); rsi = safe_float(last.get("RSI14"))
    ll10p = safe_float(df["Low"].tail(20).head(10).min())
    ll10l = safe_float(df["Low"].tail(10).min())
    if (ll10p and ll10l and ll10l >= ll10p and lc and ma20 and ma50
            and lc > ma20 and ma20 >= ma50):
        score_delta += 0.8; reasons.append("Trendstruktur: höheres Tief + MA20 über MA50 → gesunder Pullback.")
    elif lc and ma20 and ma50 and lc < ma20 and ma20 < ma50:
        score_delta -= 0.8; reasons.append("Trendstruktur: unter MA20 + MA20 unter MA50 → angeschlagen.")
    if rsi:
        if 45 <= rsi <= 62: score_delta += 0.5; reasons.append(f"RSI: {rsi:.1f} → abgekühlt, brauchbarer Bereich.")
        elif rsi >= 72: score_delta -= 0.5; reasons.append(f"RSI: {rsi:.1f} → heißgelaufen.")
        elif rsi <= 38: reasons.append(f"RSI: {rsi:.1f} → schwach, beobachtbar.")
    if len(last5) >= 2:
        prev = last5.iloc[-2]; curr = last5.iloc[-1]
        pc = safe_float(prev.get("Close")); cc = safe_float(curr.get("Close"))
        co = safe_float(curr.get("Open")); cl = safe_float(curr.get("Low")); ch = safe_float(curr.get("High"))
        if all(v is not None for v in [pc, cc, co, cl, ch]):
            body = abs(cc - co); rng = max(ch - cl, 1e-9); lwick = min(cc, co) - cl
            if cc > pc and body/rng <= 0.55 and lwick/rng >= 0.25:
                score_delta += 0.4; reasons.append("Kerze: grüne Bestätigung mit unterem Docht.")
    hh = safe_float(last20["High"].head(19).max()) if len(last20) >= 20 else None
    if hh and lc and lc > hh: score_delta += 0.5; reasons.append("Ausbruch über lokales 20-Tage-Hoch.")
    return score_delta, reasons

def chart_check_shortterm(ticker, period="2y"):
    df = fetch_price_history(ticker, period=period, interval="1d")
    if df is None or df.empty: return None
    needed = {"Open","High","Low","Close","Volume"}
    if not needed.issubset(set(df.columns)): return None
    df = df.dropna(subset=["Open","High","Low","Close"]).copy()
    if df.empty: return None
    close = df["Close"].copy(); high = df["High"].copy(); low = df["Low"].copy()
    df["ret"] = close.pct_change()
    df["MA20"] = close.rolling(20).mean(); df["MA50"] = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean(); df["RSI14"] = compute_rsi(close, 14)
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()
    # MACD (12/26/9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    # Volumen-Durchschnitt
    if "Volume" in df.columns:
        df["Vol_MA20"] = df["Volume"].rolling(20).mean()
    latest = df.iloc[-1]
    lc = safe_float(latest.get("Close")); ma20 = safe_float(latest.get("MA20"))
    ma50 = safe_float(latest.get("MA50")); ma200 = safe_float(latest.get("MA200"))
    atr = safe_float(latest.get("ATR14"))
    bg = []
    if ma200:
        if ma50 and ma50 > ma200: bg.append("Langfristig: MA50 > MA200 (Rückenwind).")
        elif ma50 and ma50 < ma200: bg.append("Langfristig: MA50 < MA200 (Gegenwind).")
    else: bg.append("MA200 nicht verfügbar (zu wenig Historie).")
    score = 5.0; reasons = []
    last10 = df.tail(10); prev20 = df.tail(30).head(20)
    last15 = df.tail(15); last12 = df.tail(12)
    vol10 = float(last10["ret"].std()) if len(last10) >= 8 else np.nan
    vol20 = float(prev20["ret"].std()) if len(prev20) >= 15 else np.nan
    if np.isfinite(vol10) and np.isfinite(vol20) and vol20 > 0:
        ratio = vol10 / vol20
        if ratio <= 0.85: score += 0.9; reasons.append(f"Unruhe nimmt ab ({ratio:.2f}) → Timing sauberer.")
        elif ratio >= 1.15:
            om20 = lc and ma20 and lc > ma20; om50 = lc and ma50 and lc > ma50
            if om20 and om50: score -= 0.2; reasons.append(f"Unruhe hoch ({ratio:.2f}), MAs halten → Momentum.")
            else: score -= 0.9; reasons.append(f"Unruhe hoch ({ratio:.2f}) + MAs nicht gehalten → riskant.")
        else: reasons.append(f"Unruhe ähnlich ({ratio:.2f}) → neutral.")
    dn = int((last15["ret"] < 0).sum()); up = int((last15["ret"] > 0).sum())
    avg_dn = float(last15.loc[last15["ret"] < 0,"ret"].mean()) if dn > 0 else 0.0
    avg_up = float(last15.loc[last15["ret"] > 0,"ret"].mean()) if up > 0 else 0.0
    if dn >= 10 and avg_dn <= -0.012: score -= 1.0; reasons.append(f"Verkaufsdruck hoch ({dn}/15) → eher warten.")
    elif dn <= 6 and avg_dn > -0.012: score += 0.8; reasons.append(f"Verkaufsdruck lässt nach ({dn}/15) → Timing besser.")
    else: reasons.append(f"Verkaufsdruck: rot {dn}/15 | Ø rot {percent(avg_dn)} | grün {percent(avg_up)}.")
    def dist(a, b): return None if (not a or not b or b == 0) else (a/b) - 1.0
    d20 = dist(lc, ma20); d50 = dist(lc, ma50)
    if d20 is not None:
        if abs(d20) <= 0.02: score += 0.7; reasons.append(f"Nahe MA20 ({percent(d20)}) → Entry-Fenster.")
        elif d20 >= 0.06: score -= 0.7; reasons.append(f"Weit über MA20 ({percent(d20)}) → hinterherlaufen.")
        elif d20 <= -0.04: score -= 0.6; reasons.append(f"Unter MA20 ({percent(d20)}) → Momentum schwächer.")
    if d50 is not None:
        if 0 <= d50 <= 0.05: score += 0.3
        elif d50 < -0.05: score -= 0.4
    if len(last12) >= 10 and lc:
        r = (last12["Close"].max() / last12["Close"].min()) - 1.0
        if atr and atr > 0 and lc > 0:
            atr_rel = atr / lc
            if r <= max(0.06, 3.0 * atr_rel): score += 0.6; reasons.append(f"Konsolidierung eng ({percent(r)}) → sammelt sich.")
            else: reasons.append(f"Konsolidierung eher breit ({percent(r)}).")
        else:
            if r <= 0.06: score += 0.6; reasons.append(f"Konsolidierung eng ({percent(r)}) → sammelt sich.")
            else: reasons.append(f"Konsolidierung eher breit ({percent(r)}).")
    if "Volume" in last15.columns and last15["Volume"].sum() > 0:
        vd = float(last15.loc[last15["ret"] < 0,"Volume"].mean()) if dn > 0 else np.nan
        vu = float(last15.loc[last15["ret"] > 0,"Volume"].mean()) if up > 0 else np.nan
        if np.isfinite(vd) and np.isfinite(vu) and vu > 0:
            vr = vd / vu
            if vr >= 1.25 and dn >= up: score -= 0.6; reasons.append(f"Volumen: Abverkauf dominiert ({vr:.2f}) → Distribution.")
            elif vr <= 0.90 and up >= dn: score += 0.4; reasons.append(f"Volumen: Aufwärts-Tage tragen ({vr:.2f}) → Aufbau.")
            else: reasons.append(f"Volumen: down/up ~ {vr:.2f} (neutral).")
        # Aktuelles Volumen vs. 20-Tage-Schnitt
        vol_ma20_v = safe_float(latest.get("Vol_MA20"))
        curr_vol = safe_float(latest.get("Volume"))
        if vol_ma20_v and curr_vol and vol_ma20_v > 0:
            vol_r = curr_vol / vol_ma20_v
            if vol_r >= 1.5 and lc and ma20 and lc > ma20:
                score += 0.5; reasons.append(f"Volumen-Bestätigung: aktuell {vol_r:.1f}x Ø20 bei steigendem Kurs → starkes Signal.")
            elif vol_r >= 1.5 and lc and ma20 and lc < ma20:
                score -= 0.4; reasons.append(f"Volumen-Druck: aktuell {vol_r:.1f}x Ø20 bei fallendem Kurs → erhöhter Abgabedruck.")
            elif vol_r <= 0.5:
                reasons.append(f"Aktuelles Volumen sehr schwach ({vol_r:.1f}x Ø20) — Bewegung wenig bestätigt.")
    # MACD-Scoring
    if len(df) >= 2:
        macd_c = safe_float(df.iloc[-1].get("MACD")); sig_c = safe_float(df.iloc[-1].get("MACD_Signal"))
        macd_p = safe_float(df.iloc[-2].get("MACD")); sig_p = safe_float(df.iloc[-2].get("MACD_Signal"))
        if all(v is not None for v in [macd_c, sig_c, macd_p, sig_p]):
            crossed_up = macd_p <= sig_p and macd_c > sig_c
            crossed_dn = macd_p >= sig_p and macd_c < sig_c
            if crossed_up:
                score += 0.7; reasons.append(f"MACD: Bullische Kreuzung (Linie über Signal) — frisches Aufwärts-Momentum.")
            elif crossed_dn:
                score -= 0.7; reasons.append(f"MACD: Bärische Kreuzung (Linie unter Signal) — Momentum dreht negativ.")
            elif macd_c > sig_c:
                score += 0.3; reasons.append(f"MACD: Bullisch ({macd_c:.3f} > {sig_c:.3f}) — Momentum weiterhin positiv.")
            else:
                score -= 0.3; reasons.append(f"MACD: Bärisch ({macd_c:.3f} < {sig_c:.3f}) — Momentum unter Druck.")
    td, tr2 = analyze_trend_structure(df); score += td; reasons.extend(tr2)
    return df, clip_score(score), reasons, bg

# ══════════════════════════════════════════════════════════════════════════════
# Fundamentals
# ══════════════════════════════════════════════════════════════════════════════
def score_core_fundamentals(metrics):
    reasons = []; score = 5.5; avail = 0
    beta = metrics.get("beta"); pe = metrics.get("pe"); peg = metrics.get("peg")
    ps = metrics.get("ps"); div = metrics.get("div_yield"); pb = metrics.get("pb")
    if beta is not None:
        avail += 1
        if beta <= 0.60: score += 1.2; reasons.append(f"Stabilität: Beta {beta:.2f} → sehr ruhig, ideal für Core.")
        elif beta <= 0.80: score += 0.8; reasons.append(f"Stabilität: Beta {beta:.2f} → gut kontrollierbar.")
        elif beta <= 1.00: score += 0.4; reasons.append(f"Stabilität: Beta {beta:.2f} → marktähnlich, solide.")
        elif beta <= 1.20: score -= 0.3; reasons.append(f"Stabilität: Beta {beta:.2f} → leicht über Markt.")
        elif beta <= 1.40: score -= 0.7; reasons.append(f"Stabilität: Beta {beta:.2f} → für Core zu unruhig.")
        else: score -= 1.2; reasons.append(f"Stabilität: Beta {beta:.2f} → klar zu volatil für Core.")
    else: reasons.append("Stabilität: Beta nicht verfügbar.")
    if pe is not None:
        avail += 1
        if pe < 10: score -= 0.2; reasons.append(f"Bewertung: KGV {pe:.1f} → günstig, Vorsicht Value-Trap.")
        elif pe <= 18: score += 0.7; reasons.append(f"Bewertung: KGV {pe:.1f} → attraktiv für Qualität.")
        elif pe <= 30: score += 0.4; reasons.append(f"Bewertung: KGV {pe:.1f} → vernünftiges Multiple.")
        elif pe <= 40: score += 0.1; reasons.append(f"Bewertung: KGV {pe:.1f} → teuer, für starke Qualität OK.")
        elif pe <= 55: score -= 0.4; reasons.append(f"Bewertung: KGV {pe:.1f} → klar ambitioniert.")
        else: score -= 0.8; reasons.append(f"Bewertung: KGV {pe:.1f} → sehr teuer.")
    else: reasons.append("Bewertung: KGV nicht verfügbar.")
    if peg is not None:
        avail += 1
        if peg < 1.0: score += 0.8; reasons.append(f"PEG {peg:.2f} → sehr attraktiv.")
        elif peg <= 2.0: score += 0.5; reasons.append(f"PEG {peg:.2f} → gut vertretbar.")
        elif peg <= 3.0: score += 0.2; reasons.append(f"PEG {peg:.2f} → akzeptabel.")
        elif peg <= 4.5: score -= 0.3; reasons.append(f"PEG {peg:.2f} → gestreckt.")
        elif peg <= 6.0: score -= 0.6; reasons.append(f"PEG {peg:.2f} → teuer relativ zum Wachstum.")
        else: score -= 0.9; reasons.append(f"PEG {peg:.2f} → deutlich zu teuer.")
    else: reasons.append("PEG nicht verfügbar.")
    if ps is not None:
        avail += 1
        if ps <= 3: score += 0.5; reasons.append(f"KUV {ps:.1f} → sehr günstig.")
        elif ps <= 6: score += 0.3; reasons.append(f"KUV {ps:.1f} → angenehm.")
        elif ps <= 10: reasons.append(f"KUV {ps:.1f} → für Qualität vertretbar.")
        elif ps <= 15: score -= 0.4; reasons.append(f"KUV {ps:.1f} → hoch, braucht Marge.")
        else: score -= 0.7; reasons.append(f"KUV {ps:.1f} → sehr hoch.")
    if pb is not None:
        avail += 1
        if pb <= 4: score += 0.2; reasons.append(f"KBV {pb:.1f} → moderat.")
        elif pb <= 8: reasons.append(f"KBV {pb:.1f} → im Rahmen.")
        elif pb <= 12: score -= 0.2; reasons.append(f"KBV {pb:.1f} → hoch bewertet.")
        else: score -= 0.4; reasons.append(f"KBV {pb:.1f} → sehr hoch.")
    if div is not None:
        avail += 1
        if 1.5 <= div <= 4.0: score += 0.3; reasons.append(f"Dividende {div:.2f}% → Stabilitätsbonus.")
        elif div > 6: score -= 0.1; reasons.append(f"Dividende {div:.2f}% → sehr hoch.")
        elif div < 0.5: reasons.append(f"Dividende {div:.2f}% → keine Yield-Story.")
        else: reasons.append(f"Dividende {div:.2f}% → moderat.")
    # ── FCF Yield ──────────────────────────────────────────────────────────────
    fcf_yield = metrics.get("fcf_yield")
    if fcf_yield is not None:
        avail += 1
        if fcf_yield >= 6:   score += 1.0; reasons.append(f"FCF-Rendite {fcf_yield:.1f}% → sehr starke Cashgenerierung.")
        elif fcf_yield >= 3: score += 0.6; reasons.append(f"FCF-Rendite {fcf_yield:.1f}% → solide Cashgenerierung.")
        elif fcf_yield >= 1: score += 0.2; reasons.append(f"FCF-Rendite {fcf_yield:.1f}% → moderate Cashgenerierung.")
        elif fcf_yield < 0:  score -= 0.8; reasons.append(f"FCF-Rendite {fcf_yield:.1f}% → negativer Free Cashflow, Vorsicht.")
        else:                              reasons.append(f"FCF-Rendite {fcf_yield:.1f}% → gering.")

    # ── Verschuldung ────────────────────────────────────────────────────────────
    debt_ebitda = metrics.get("debt_ebitda")
    if debt_ebitda is not None:
        avail += 1
        if debt_ebitda < 0:     score += 0.3; reasons.append("Verschuldung: Nettocash-Position → sehr solide Bilanz.")
        elif debt_ebitda < 1.0: score += 0.5; reasons.append(f"Verschuldung: {debt_ebitda:.1f}x EBITDA → sehr konservativ.")
        elif debt_ebitda < 2.5: score += 0.3; reasons.append(f"Verschuldung: {debt_ebitda:.1f}x EBITDA → gesunde Bilanz.")
        elif debt_ebitda < 4.0: score -= 0.2; reasons.append(f"Verschuldung: {debt_ebitda:.1f}x EBITDA → erhöht, noch tragbar.")
        elif debt_ebitda < 6.0: score -= 0.5; reasons.append(f"Verschuldung: {debt_ebitda:.1f}x EBITDA → hoch für Core.")
        else:                   score -= 0.9; reasons.append(f"Verschuldung: {debt_ebitda:.1f}x EBITDA → kritisch hoch.")

    # ── Operative Marge ─────────────────────────────────────────────────────────
    op_margin = metrics.get("op_margin")
    if op_margin is not None:
        avail += 1
        if op_margin >= 25:   score += 0.6; reasons.append(f"Op. Marge {op_margin:.1f}% → außergewöhnlich profitabel.")
        elif op_margin >= 15: score += 0.4; reasons.append(f"Op. Marge {op_margin:.1f}% → starke Profitabilität.")
        elif op_margin >= 8:  score += 0.1; reasons.append(f"Op. Marge {op_margin:.1f}% → solide.")
        elif op_margin >= 0:               reasons.append(f"Op. Marge {op_margin:.1f}% → gering.")
        else:                 score -= 0.6; reasons.append(f"Op. Marge {op_margin:.1f}% → operativer Verlust.")

    # ── Umsatzwachstum & Konsistenz ─────────────────────────────────────────────
    rev_growth  = metrics.get("rev_growth_yoy")
    rev_cagr    = metrics.get("rev_cagr_3y")
    earn_years  = metrics.get("earnings_years")
    if rev_growth is not None:
        if rev_growth >= 15:   score += 0.4; reasons.append(f"Umsatzwachstum {rev_growth:.1f}% YoY → dynamisches Wachstum.")
        elif rev_growth >= 5:  score += 0.2; reasons.append(f"Umsatzwachstum {rev_growth:.1f}% YoY → solides Wachstum.")
        elif rev_growth >= 0:               reasons.append(f"Umsatzwachstum {rev_growth:.1f}% YoY → stagnierend.")
        else:                 score -= 0.4; reasons.append(f"Umsatzwachstum {rev_growth:.1f}% YoY → schrumpfender Umsatz.")
    if rev_cagr is not None:
        if rev_cagr >= 10:   score += 0.3; reasons.append(f"3J-Wachstum: {rev_cagr:.1f}% p.a. → starke Wachstumsbahn.")
        elif rev_cagr >= 3:  score += 0.1; reasons.append(f"3J-Wachstum: {rev_cagr:.1f}% p.a. → moderates Wachstum.")
        elif rev_cagr < 0:   score -= 0.3; reasons.append(f"3J-Wachstum: {rev_cagr:.1f}% p.a. → Schrumpfung über 3 Jahre.")
    if earn_years is not None:
        if earn_years >= 3:  score += 0.3; reasons.append(f"Gewinnkonsistenz: {earn_years}/3 Jahre Umsatzwachstum → sehr konsistent.")
        elif earn_years == 2: score += 0.1; reasons.append(f"Gewinnkonsistenz: {earn_years}/3 Jahre Umsatzwachstum → größtenteils stabil.")
        elif earn_years <= 1: score -= 0.2; reasons.append(f"Gewinnkonsistenz: {earn_years}/3 Jahre Umsatzwachstum → unregelmäßig.")

    if avail == 0: return None, ["Keine Fundamentaldaten → Score nicht berechenbar."]
    _base_avail = sum(1 for k in ["beta","pe","peg","ps","pb","div_yield"] if metrics.get(k) is not None)
    if _base_avail <= 2: score -= 0.4; reasons.append(f"Datenlage: nur {_base_avail}/6 Basiskennzahlen.")
    elif _base_avail >= 5: score += 0.1; reasons.append(f"Datenlage: {_base_avail}/6 Kennzahlen → gute Basis.")
    return clip_score(score), reasons

def score_hc_fundamentals(metrics):
    """Hidden Champion Scoring — Potenzial vor Stabilität.
    HCs dürfen höher bewertet sein wenn Wachstum + Margen + Nischenstärke stimmen.
    Volatilität (Beta) wird NICHT bestraft — ist HC-typisch."""
    reasons = []; score = 5.0; avail = 0
    beta = metrics.get("beta"); pe = metrics.get("pe"); peg = metrics.get("peg")
    ps = metrics.get("ps"); mcap = metrics.get("mcap"); pb = metrics.get("pb"); div = metrics.get("div_yield")

    # ── Größe: HC lebt in Small/Mid-Cap ───────────────────────────────────────
    if mcap is not None:
        avail += 1
        if 0.1 <= mcap <= 5:   score += 1.2; reasons.append(f"Größe: {mcap:.1f} Mrd → klassischer HC-Sweet-Spot.")
        elif 5 < mcap <= 20:   score += 0.9; reasons.append(f"Größe: {mcap:.1f} Mrd → typisch HC.")
        elif 20 < mcap <= 50:  score += 0.4; reasons.append(f"Größe: {mcap:.1f} Mrd → noch HC-kompatibel.")
        elif 50 < mcap <= 100: score -= 0.2; reasons.append(f"Größe: {mcap:.1f} Mrd → groß, aber möglich.")
        elif mcap > 100: score -= 0.8; reasons.append(f"Größe: {mcap:.1f} Mrd → zu groß für klassischen HC.")

    # ── Beta: HC darf schwanken — kein Abzug für Volatilität ─────────────────
    if beta is not None:
        avail += 1
        if beta <= 0.8:  score += 0.3; reasons.append(f"Beta {beta:.2f} → ungewöhnlich ruhig für HC.")
        elif beta <= 1.5: reasons.append(f"Beta {beta:.2f} → typische HC-Schwankungsbreite.")
        elif beta <= 2.5: score += 0.2; reasons.append(f"Beta {beta:.2f} → volatil — für HC mit hohem Potenzial OK.")
        else: score -= 0.4; reasons.append(f"Beta {beta:.2f} → sehr hoch, Risiko beachten.")

    # ── KGV: HCs wachsen — höhere Multiples sind OK wenn Wachstum stimmt ──────
    if pe is not None:
        avail += 1
        if pe < 8:    score -= 0.4; reasons.append(f"KGV {pe:.1f} → zu günstig, prüfe ob These gebrochen.")
        elif pe <= 25: score += 0.6; reasons.append(f"KGV {pe:.1f} → attraktiv für HC mit Wachstum.")
        elif pe <= 45: score += 0.4; reasons.append(f"KGV {pe:.1f} → Wachstumsprämie, akzeptabel.")
        elif pe <= 70: score += 0.1; reasons.append(f"KGV {pe:.1f} → hoch — nur ok bei starkem Wachstum.")
        else: score -= 0.5; reasons.append(f"KGV {pe:.1f} → sehr ambitioniert, Wachstum muss liefern.")
    if peg is not None:
        avail += 1
        if peg < 1.5: score += 0.7; reasons.append(f"PEG {peg:.2f} → sehr gut für HC.")
        elif peg <= 3.0: score += 0.3; reasons.append(f"PEG {peg:.2f} → brauchbar.")
        elif peg <= 5.0: score -= 0.2; reasons.append(f"PEG {peg:.2f} → gestreckt.")
        else: score -= 0.5; reasons.append(f"PEG {peg:.2f} → teuer.")
    if ps is not None:
        avail += 1
        if ps <= 5: score += 0.4; reasons.append(f"KUV {ps:.1f} → gut für HC.")
        elif ps <= 10: score += 0.2; reasons.append(f"KUV {ps:.1f} → im Rahmen.")
        elif ps <= 18: score -= 0.3; reasons.append(f"KUV {ps:.1f} → hoch.")
        else: score -= 0.6; reasons.append(f"KUV {ps:.1f} → sehr hoch.")
    if div is not None:
        avail += 1
        if 0.5 <= div <= 3.0: score += 0.2; reasons.append(f"Dividende {div:.2f}% → leichter Bonus.")
        elif div > 5: reasons.append(f"Dividende {div:.2f}% → hoch, nicht typisch für HC.")
    # ── FCF + Debt für HC ───────────────────────────────────────────────────────
    fcf_yield = metrics.get("fcf_yield")
    if fcf_yield is not None:
        avail += 1
        if fcf_yield >= 4:   score += 0.8; reasons.append(f"FCF-Rendite {fcf_yield:.1f}% → starke Innenfinanzierung für HC.")
        elif fcf_yield >= 2: score += 0.4; reasons.append(f"FCF-Rendite {fcf_yield:.1f}% → positive Cashgenerierung.")
        elif fcf_yield < 0:  score -= 0.6; reasons.append(f"FCF-Rendite {fcf_yield:.1f}% → negativer Free Cashflow.")
    debt_ebitda = metrics.get("debt_ebitda")
    if debt_ebitda is not None:
        avail += 1
        if debt_ebitda < 1.5: score += 0.4; reasons.append(f"Verschuldung {debt_ebitda:.1f}x EBITDA → sehr konservativ für HC.")
        elif debt_ebitda < 3: score += 0.2; reasons.append(f"Verschuldung {debt_ebitda:.1f}x EBITDA → tragbar.")
        elif debt_ebitda > 5: score -= 0.5; reasons.append(f"Verschuldung {debt_ebitda:.1f}x EBITDA → für HC zu hoch.")

    # ── Margen und Wachstum für HC ──────────────────────────────────────────────
    op_margin  = metrics.get("op_margin")
    rev_cagr   = metrics.get("rev_cagr_3y")
    earn_years = metrics.get("earnings_years")
    if op_margin is not None:
        avail += 1
        if op_margin >= 20:   score += 0.7; reasons.append(f"Op. Marge {op_margin:.1f}% → HC-typische Premiummargen.")
        elif op_margin >= 12: score += 0.4; reasons.append(f"Op. Marge {op_margin:.1f}% → solide für HC.")
        elif op_margin >= 5:  score += 0.1; reasons.append(f"Op. Marge {op_margin:.1f}% → noch akzeptabel.")
        else:                 score -= 0.5; reasons.append(f"Op. Marge {op_margin:.1f}% → zu dünn für HC-Profil.")
    if rev_cagr is not None:
        if rev_cagr >= 12:   score += 0.5; reasons.append(f"3J-Wachstum {rev_cagr:.1f}% p.a. → starkes organisches Wachstum.")
        elif rev_cagr >= 5:  score += 0.2; reasons.append(f"3J-Wachstum {rev_cagr:.1f}% p.a. → moderates Wachstum.")
        elif rev_cagr < 0:   score -= 0.4; reasons.append(f"3J-Wachstum {rev_cagr:.1f}% p.a. → rückläufiger Umsatz.")
    if earn_years is not None:
        if earn_years >= 3:   score += 0.3; reasons.append(f"Konsistenz: {earn_years}/3 Jahre Umsatzwachstum → sehr verlässlich.")
        elif earn_years <= 1: score -= 0.2; reasons.append(f"Konsistenz: {earn_years}/3 Jahre Umsatzwachstum → unregelmäßig.")

    if avail == 0: return None, ["Keine Fundamentaldaten → Score nicht berechenbar."]
    _base_avail = sum(1 for k in ["mcap","beta","pe","peg","ps","div_yield"] if metrics.get(k) is not None)
    if _base_avail <= 2: score -= 0.4; reasons.append(f"Datenlage: nur {_base_avail}/6 Kennzahlen.")
    elif _base_avail >= 5: score += 0.1; reasons.append(f"Datenlage: {_base_avail}/6 → gute Basis.")
    return clip_score(score), reasons

# ══════════════════════════════════════════════════════════════════════════════
# Relative Bewertung
# ══════════════════════════════════════════════════════════════════════════════
def score_relative_valuation(metrics: dict, sector: str) -> tuple:
    """Vergleicht PE/PB/PS mit Sektor-Benchmark. Gibt (delta_score, reasons) zurück."""
    bm = SECTOR_BENCHMARKS.get(sector)
    if not bm or not sector:
        return 0.0, []
    delta = 0.0; reasons = []
    pe = metrics.get("pe"); bm_pe = bm.get("pe")
    if pe and bm_pe:
        ratio = pe / bm_pe
        if ratio < 0.75:   delta += 0.6; reasons.append(f"Rel. KGV: {pe:.1f} vs. Sektor {bm_pe} → 25%+ günstiger als Peer-Gruppe.")
        elif ratio < 0.90: delta += 0.3; reasons.append(f"Rel. KGV: {pe:.1f} vs. Sektor {bm_pe} → leicht günstiger als Peer-Gruppe.")
        elif ratio < 1.15:              reasons.append(f"Rel. KGV: {pe:.1f} vs. Sektor {bm_pe} → im Rahmen der Peer-Gruppe.")
        elif ratio < 1.40: delta -= 0.3; reasons.append(f"Rel. KGV: {pe:.1f} vs. Sektor {bm_pe} → Aufschlag zur Peer-Gruppe.")
        else:              delta -= 0.6; reasons.append(f"Rel. KGV: {pe:.1f} vs. Sektor {bm_pe} → deutlich teurer als Peer-Gruppe.")
    pb = metrics.get("pb"); bm_pb = bm.get("pb")
    if pb and bm_pb:
        ratio = pb / bm_pb
        if ratio < 0.80:   delta += 0.3; reasons.append(f"Rel. KBV: {pb:.1f} vs. Sektor {bm_pb} → günstig zum Sektor.")
        elif ratio > 1.50: delta -= 0.3; reasons.append(f"Rel. KBV: {pb:.1f} vs. Sektor {bm_pb} → Bewertungsaufschlag zum Sektor.")
    ps = metrics.get("ps"); bm_ps = bm.get("ps")
    if ps and bm_ps:
        ratio = ps / bm_ps
        if ratio < 0.70:   delta += 0.3; reasons.append(f"Rel. KUV: {ps:.1f} vs. Sektor {bm_ps} → günstig zum Sektor-Schnitt.")
        elif ratio > 1.60: delta -= 0.3; reasons.append(f"Rel. KUV: {ps:.1f} vs. Sektor {bm_ps} → Aufschlag zum Sektor-Schnitt.")
    return round(delta, 2), reasons

# ══════════════════════════════════════════════════════════════════════════════
# Business / Story
# ══════════════════════════════════════════════════════════════════════════════
def has_story_basis(profile):
    if not profile: return False
    return sum([bool((profile.get("sector") or "").strip()),
                bool((profile.get("industry") or "").strip()),
                len((profile.get("summary") or "").strip()) >= 80]) >= 2

# ══════════════════════════════════════════════════════════════════════════════
# Vergleichbare Aktien — kuratierte Map nach Geschäftsmodell
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# Kuratierter Aktien-Katalog
# Format: (Ticker, Name, Warum-Satz)
# ══════════════════════════════════════════════════════════════════════════════
SIMILAR_STOCKS: dict = {
    "Software / SaaS": {
        "Core Asset": [
            ("MSFT",  "Microsoft",    "Plattform-Monopol mit Cloud-Wachstum — das stabilste Fundament im Tech."),
            ("NOW",   "ServiceNow",   "Enterprise-Workflow-Plattform mit 97%+ Renewal Rate und starker Pricing Power."),
            ("ADBE",  "Adobe",        "Kreativ-Software-Monopol — Wechselkosten so hoch wie kaum ein anderes Produkt."),
        ],
        "Hidden Champion": [
            ("CSWI",  "CSW Industrials",  "Vertical-Market-Software mit serieller Akquisitionsstrategie wie Constellation."),
            ("TYL",   "Tyler Technologies","Marktführer für Behörden-Software — kaum Konkurrenz, langfristige Verträge."),
            ("VITB.ST","Vitec Software",   "Schwedischer Serial Acquirer von Nischen-Software — Constellation-Modell in Klein."),
        ],
    },
    "Halbleiter / Chip-Zulieferer": {
        "Core Asset": [
            ("ASML.AS","ASML",          "Globales Monopol auf EUV-Lithographie — kein Chip ohne ASML-Maschinen."),
            ("TSM",   "TSMC",           "Weltweit führende Chip-Fabrik — 90%+ Marktanteil bei 5nm-Prozessen."),
            ("AMAT",  "Applied Materials","Größter Ausrüster für Chip-Herstellung — profitiert von jedem Kapex-Zyklus."),
        ],
        "Hidden Champion": [
            ("BESI.AS","BE Semiconductor","80%+ Bruttomargen, Weltmarktführer für Die-Attach-Equipment — kaum bekannt."),
            ("ENTG",  "Entegris",        "Spezialchemikalien für Chips — ohne Entegris stoppt jede Fab-Linie."),
            ("FORM",  "FormFactor",       "Marktführer für Wafer-Probe-Karten — jeder neue Chip muss durch FormFactor."),
        ],
    },
    "MedTech / Diagnostics": {
        "Core Asset": [
            ("SYK",   "Stryker",          "Robotik-Chirurgie und Implantate — hohe Wechselkosten nach Installation."),
            ("EW",    "Edwards Lifesciences","Herzklappen-Spezialist mit einzigartiger TAVI-Technologie, keine echte Alternative."),
            ("ISRG",  "Intuitive Surgical","Roboter-Chirurgie-Monopol — da Vault schon installiert ist, kommt keiner rein."),
        ],
        "Hidden Champion": [
            ("INSP",  "Inspire Medical",  "Einzige FDA-zugelassene Neurostimulation gegen Schlafapnoe — Nische mit 10x-Potenzial."),
            ("MMSI",  "Merit Medical",    "Outsourcing-Partner für große MedTech-Konzerne — profitiert von deren Wachstum."),
            ("ITGR",  "Integer Holdings", "Unsichtbarer Zulieferer hinter Herzschrittmachern und Kathetern großer Marken."),
        ],
    },
    "Industrie / Zulieferer": {
        "Core Asset": [
            ("ITW",   "Illinois Tool Works","80/20-Prinzip gelebt — fokussierte Nischen mit überragenden Margen seit Jahrzehnten."),
            ("ROK",   "Rockwell Automation","Industrieautomation trifft Digitalisierung — kaum austauschbar nach Integration."),
            ("ROP",   "Roper Technologies", "Serial Acquirer von Software für Industrie — Cashflow-Maschine, kaum Capex."),
        ],
        "Hidden Champion": [
            ("RAA.DE","Rational AG",       "Weltmarktführer für Kombidämpfer — 60% Bruttomargen, kein Wettbewerber in Sicht."),
            ("DPLM.L","Diploma PLC",       "UK-Vertriebsnische für Technikkomponenten — serieller Acquirer mit 20J. Kursanstieg."),
            ("LAGR.ST","Lagercrantz Group","Schwedischer Serial Acquirer in der B2B-Nische — kaum bekannt, starke Zahlen."),
        ],
    },
    "Royalty / Streaming": {
        "Core Asset": [
            ("WPM",   "Wheaton Precious",  "Reinster Royalty-Player — keine Mine, kaum Kosten, maximale Hebelwirkung auf Gold."),
            ("FNV",   "Franco-Nevada",     "Breiter diversifiziertes Royalty-Portfolio — das stabilste Modell im Sektor."),
            ("RGLD",  "Royal Gold",        "Fokussiert auf große Minen-Royalties — hoher Free Cashflow, konservatives Management."),
        ],
        "Hidden Champion": [
            ("OR.TO", "Osisko Gold Royalties","Kanadischer Royalty-Spezialist mit starker Canadian Malartic-Basis-Royalty."),
            ("SAND",  "Sandstorm Gold",    "Günstigste Bewertung im Royalty-Sektor — wächst schneller als große Peers."),
            ("TFPM.TO","Triple Flag Metals","Diversifiziertes Junior-Royalty mit Silber/Zink-Gewichtung — unterbewertet vs. Peers."),
        ],
    },
    "Rohstoff / Mining": {
        "Core Asset": [
            ("NEM",   "Newmont",           "Weltweit größter Goldproduzent — liquide, diversifiziert, langfristige Reserven."),
            ("BHP",   "BHP Group",         "Breite Rohstoff-Diversifikation — Kupfer, Eisenerz, Kohle in einer Aktie."),
            ("FCX",   "Freeport-McMoRan",  "Weltgrößter Kupferproduzent — direktester Play auf die Elektrifizierungswende."),
        ],
        "Hidden Champion": [
            ("MAG",   "MAG Silver",        "Hochgradige Silber-Mine in Mexiko — eines der weltweit besten Silber-Deposits."),
            ("WPM",   "Wheaton Precious",  "Royalty statt Mining — dasselbe Upside, ein Bruchteil des Risikos."),
            ("SFR.AX","Sandfire Resources","Australischer Kupfer-Spezialist — strukturelles Wachstum durch neue Mine in Marokko."),
        ],
    },
    "Plattform / Payments": {
        "Core Asset": [
            ("V",     "Visa",              "Duopol-Infrastruktur — jede Kartenzahlung weltweit trägt Visa-Margen."),
            ("MA",    "Mastercard",        "Dasselbe Duopol wie Visa — vielleicht noch stärker im internationalen B2B-Bereich."),
            ("COIN",  "Coinbase",          "Quasi-Infrastruktur für Krypto — profitiert unabhängig davon welche Coin gewinnt."),
        ],
        "Hidden Champion": [
            ("FOUR",  "Shift4 Payments",   "Marktführer für Zahlungen im Hotel/Restaurant-Sektor — komplexe Nische, kaum Konkurrenz."),
            ("FLYW",  "Flywire",           "Spezialist für grenzüberschreitende Zahlungen in Bildung und Gesundheit."),
            ("WEX",   "WEX Inc.",          "Fleet-Card-Monopol für LKW-Fahrer — hohe Wechselkosten, wiederkehrender Umsatz."),
        ],
    },
    "Infrastruktur / Konzessionen": {
        "Core Asset": [
            ("AMT",   "American Tower",    "Globales Mobilfunkmast-Netz — jedes Smartphone zahlt indirekt Miete an AMT."),
            ("BIP",   "Brookfield Infrastr.","Diversifizierter Infrastruktur-Gigant — Häfen, Pipelines, Datenzentren."),
            ("CPRT",  "Copart",            "Toll-Booth für Schrottfahrzeuge — Netz-Effekt macht jeden Wettbewerber unmöglich."),
        ],
        "Hidden Champion": [
            ("POOL",  "Pool Corporation",  "Monopolartiger Großhändler für Schwimmbad-Equipment — jede Marge ist Infra-Qualität."),
            ("SDIP.ST","Sdiptech",         "Schwedischer Infrastruktur-Nischenanbieter — Wasser, Energie, Abfall."),
            ("HLMA.L","Halma PLC",         "UK-Konglomerat aus 45 Nischen-Sicherheitstechnik-Firmen — 20J. Dividenden-Wachstum."),
        ],
    },
    "Versicherung": {
        "Core Asset": [
            ("CB",    "Chubb",             "Premium-Versicherer mit diszipliniertem Underwriting — Warren Buffetts Liebling."),
            ("MKL",   "Markel",            "'Mini-Berkshire' mit Versicherungs-Float als Investmentkapital."),
            ("BRK-B", "Berkshire Hathaway","Insurance-Float als Kapitalmaschine — das Original-Modell hinter allem anderen."),
        ],
        "Hidden Champion": [
            ("ERIE",  "Erie Indemnity",    "Regional-Versicherer mit 100J. Dividendenhistorie — profitabelster Nischen-Versicherer USA."),
            ("KNSL",  "Kingsway Financial","Turnaround-Story im Specialty-Insurance — Potenzial wenn Underwriting diszipliniert bleibt."),
            ("RYAN",  "Ryan Specialty",    "Insurance-Broker-Nische für schwer versicherbare Risiken — strukturelles Wachstum."),
        ],
    },
    "Konsum / Marke": {
        "Core Asset": [
            ("NESN.SW","Nestlé",           "Breitestes Konsumgüter-Portfolio — 2000+ Marken in jedem Land der Erde."),
            ("PG",    "Procter & Gamble",  "Konsumgüter-Monopol in Hygiene — Pricing Power und Markenbindung seit Jahrzehnten."),
            ("LVMH.PA","LVMH",            "Luxus-Konglomerat — Preismacht ohne Grenzen, in Asien strukturell im Aufwind."),
        ],
        "Hidden Champion": [
            ("MNST",  "Monster Beverage",  "Energy-Drink Nische mit Coca-Cola als Vertriebspartner — unterschätztes Schutzwall."),
            ("CELH",  "Celsius Holdings",  "Gesundheitsorientierter Energy-Drink — schnellster Wachstumstrend im Segment."),
            ("POOL",  "Pool Corporation",  "Unsichtbarer Monopolist im US-Pool-Markt — diskret, profitabel, wächst stetig."),
        ],
    },
    "Versorger / Netz": {
        "Core Asset": [
            ("NEE",   "NextEra Energy",    "Größter Erneuerbaren-Versorger der Welt — regulierter Rückenwind für Jahrzehnte."),
            ("SO",    "Southern Company",  "Stabile Dividende, reguliertes Monopolgebiet — klassisches defensives Core-Asset."),
            ("AEP",   "American Electric", "Höchste Übertragungs-Netz-Investitionen in USA — Infrastruktur-Investmentstory."),
        ],
        "Hidden Champion": [
            ("AMPS",  "Altus Power",       "Gewerbliche Solar-Versorger-Nische — direkter Mietvertrag ohne Netz-Abhängigkeit."),
            ("CWEN",  "Clearway Energy",   "Erneuerbare-Energie-YieldCo mit langfristigen Power Purchase Agreements."),
            ("BEPC",  "Brookfield Renewable","Weltweites Erneuerbare-Portfolio — Hydro, Wind, Solar über alle Märkte."),
        ],
    },
    "Energie / Services": {
        "Core Asset": [
            ("SLB",   "Schlumberger",      "Globale Ölfeldservice-Infrastruktur — ohne SLB läuft keine große Exploration."),
            ("CVX",   "Chevron",           "Integrierter Major mit starker Bilanz und LNG-Wachstum in Asien."),
            ("XOM",   "ExxonMobil",        "Größte Free-Cashflow-Maschine im Energiesektor — diszipliniertes Kapital-Rückgabe."),
        ],
        "Hidden Champion": [
            ("XPRO",  "Expro Group",       "Spezialist für Bohrlochtests — Nische mit hohen technischen Eintrittsbarrieren."),
            ("NGL",   "NGL Energy Partners","Midstream-Nische für Wasser-Management bei Ölförderung — unterschätzt."),
            ("PTEN",  "Patterson-UTI",     "Führender Bohrtechnik-Spezialist — profitiert direkt von US-Rig-Count-Anstieg."),
        ],
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# Velox Radar — nicht-offensichtliche Ideen basierend auf Profil-Tags
# Format: tag → [(ticker, name, warum-satz), ...]
# ══════════════════════════════════════════════════════════════════════════════
VELOX_RADAR: dict = {
    "niche_b2b": [
        ("CSU.TO",  "Constellation Software", "Der unbekannteste Serienakquisiteur der Welt — kauft Nischen-Software und lässt sie nie wieder los. 30%+ CAGR über 20 Jahre."),
        ("DPLM.L",  "Diploma PLC",            "Britischer Vertrieb von Nischen-Technikkomponenten — still, profitabel, seit 20 Jahren Kursanstieg ohne Schlagzeilen."),
        ("SDIP.ST", "Sdiptech",               "Schwedischer Infrastruktur-Nischenanbieter — Wasser, Abfall, Energie. Außerhalb Skandinaviens kaum auf dem Radar."),
    ],
    "asset_light": [
        ("CPRT",    "Copart",                 "Toll-Booth-Modell für Totalschaden-Fahrzeuge — Netzwerk-Effekt verhindert jeden Wettbewerber, fast null Capex."),
        ("ROP",     "Roper Technologies",     "Industrial Software Serial Acquirer — 50%+ Operating Margins, wächst durch Käufe ohne je Capex zu brauchen."),
        ("POOL",    "Pool Corporation",       "Monopolistischer Pool-Großhändler — klingt langweilig, ist es auch, macht aber seit 20 Jahren 20%+ p.a."),
    ],
    "recurring": [
        ("TYL",     "Tyler Technologies",     "Regierungs-Software mit 10-Jahres-Verträgen — Kündigung ist quasi unmöglich, Renewals über 95%."),
        ("FICO",    "Fair Isaac (FICO)",       "Credit-Scoring-Monopol — jede Bank, jeder Kredit, jede Hypothek zahlt FICO. Pricing Power ohne Grenzen."),
        ("VRSK",    "Verisk Analytics",       "Daten-Monopol für Versicherungsbranche — einmal drin, nie mehr raus. Recurring Revenues über 80%."),
    ],
    "quality_bias": [
        ("HEIA.AS", "Heineken",               "Premium-Bier-Marktführer Europa — unterschätzte Pricing Power, starke Emerging-Markets-Expansion."),
        ("IDEXY",   "IMCD Group",             "Chemikalien-Distributor mit Asset-light-Modell — hohe ROIC, skalierbar, Nische die kaum jemand kennt."),
        ("HLMA.L",  "Halma PLC",              "UK-Konglomerat aus 45 Sicherheitstechnik-Nischen — 20 Jahre kontinuierliches Dividendenwachstum."),
    ],
    "commodity_sensitive": [
        ("WPM",     "Wheaton Precious Metals","Royalty statt Mining — Gold-Upside ohne Mine, ohne Streik, ohne Capex-Risiko."),
        ("OR.TO",   "Osisko Gold Royalties",  "Günstigster Royalty-Player mit Wachstum — kauft Royalties bevor andere sie bemerken."),
        ("NOVN.SW", "Novartis",               "Pharma als Rohstoff-Hedge — defensiv, hohe Dividende, Pipeline unterschätzt."),
    ],
    "cyclical": [
        ("BESI.AS", "BE Semiconductor",       "Halb-Zykliker mit Moat — kommt jeder Halbleiter-Boom, verdoppelt BESI. 80%+ Bruttomarge."),
        ("ADDTECH.ST","Addtech",              "Schwedischer Technik-Distributor — profitiert von Industrie-Aufschwung ohne Zyklus-Risiko einer Fabrik."),
        ("ALFA.ST",  "Alfa Laval",            "Wärmetauscher-Nische — klingt unspektakulär, ist aber in jedem Schiff, jeder Fabrik, jeder Brauerei."),
    ],
    "defensive": [
        ("ERIE",    "Erie Indemnity",         "100-jährige Dividendenhistorie, stabiler als Staatsanleihen — der langweiligste Compounder der USA."),
        ("OTIS",    "Otis Worldwide",         "Aufzug-Monopol — eingebaut, 20 Jahre Wartungsvertrag, nie ausgetauscht. 100% Recurring."),
        ("MSA",     "MSA Safety",             "Arbeitsschutz-Ausrüstung — Regulierung erzwingt Kauf, Marktführer in der Nische seit 100 Jahren."),
    ],
    "regulated": [
        ("AWK",     "American Water Works",   "Wasserversorger-Monopol — reguliert, monopolistisch, strukturell wachsend durch Infrastruktur-Bedarf."),
        ("WMS",     "Advanced Drainage Sys.", "Drainage-Infrastruktur-Monopol — Klimawandel treibt Nachfrage, regulierte Nische."),
        ("GLBL",    "Clearfield Inc.",        "Glasfaser-Infrastruktur-Nische — von Regulierung gestützt, profitiert von Broadband-Ausbau."),
    ],
    "capital_intensive": [
        ("STLD",    "Steel Dynamics",         "Effizientester Stahlproduzent USA — Kapitalintensität durch Technologie zum Wettbewerbsvorteil gemacht."),
        ("MLM",     "Martin Marietta",        "Aggregat-Monopol — Steinbrüche sind nicht replizierbar, reguliert, mit Preismacht."),
        ("VMC",     "Vulcan Materials",       "Dieselbe Steinbruch-Story wie Martin Marietta — Oligopol, quasi-reguliert, Infrastruktur-Wachstum."),
    ],
}

def get_similar_stocks(business_model: str, prefer_mode: str, limit: int = 3) -> list:
    """Gibt kuratierte Vergleichswerte zurück. limit > 3 kombiniert beide Modi."""
    bucket = SIMILAR_STOCKS.get(business_model, {})
    other  = "Hidden Champion" if prefer_mode == "Core Asset" else "Core Asset"
    primary   = bucket.get(prefer_mode, [])
    secondary = bucket.get(other, [])
    if not primary:
        primary, secondary = secondary, primary
    # Kombiniere: zuerst primary, dann secondary ohne Duplikate
    seen   = set()
    result = []
    for tk, name, why in (primary + secondary):
        if tk not in seen:
            seen.add(tk)
            result.append((tk, name, why))
        if len(result) >= limit:
            break
    return result

# ── Velox Radar Themes ────────────────────────────────────────────────────────
VELOX_RADAR_THEMES: dict = {
    "Erneuerbare Energien": {
        "color": "#10b981", "bg": "rgba(16,185,129,0.08)",
        "desc": "Die Energiewende als Investment — Sonne, Wind und smarte Netze.",
        "mode": "Core Asset",
        "stocks": [
            ("ENPH",    "Enphase Energy",    "Marktführer bei Mikro-Wechselrichtern — jede neue Solaranlage braucht sie. Starkes Recurring-Revenue durch Monitoring-Software."),
            ("NEE",     "NextEra Energy",    "Größter Windkraft-Betreiber der USA mit regulierten Einnahmen. Kombiniert Utility-Stabilität mit Wachstum aus erneuerbaren Energien."),
            ("FSLR",    "First Solar",       "Einziger US-Solarmodul-Hersteller mit eigener Cadmium-Tellurid-Technologie — profitiert massiv vom US Inflation Reduction Act."),
            ("HASI",    "Hannon Armstrong",  "Spezialist für die Finanzierung nachhaltiger Infrastruktur. Regelmäßige Dividenden aus langfristigen Projektverträgen."),
            ("ORSTED.CO","Ørsted",           "Dänischer Pionier der Offshore-Windenergie — hat sich komplett von fossilen Brennstoffen transformiert."),
            ("SEDG",    "SolarEdge",         "Israelischer Marktführer bei Solar-Wechselrichtern und Smart-Energy-Management — globale Präsenz in über 130 Ländern."),
            ("BEP",     "Brookfield Renewable","Einer der größten Betreiber erneuerbarer Energien weltweit — Hydro, Wind, Solar mit stabilen Cashflows."),
            ("CEG",     "Constellation Energy","Größter US-Atomkraft-Betreiber — profitiert vom wachsenden Strombedarf durch KI-Rechenzentren."),
        ],
    },
    "KI & Cloud": {
        "color": "#3b82f6", "bg": "rgba(59,130,246,0.08)",
        "desc": "Die Infrastruktur der nächsten Ära — Chips, Software und Plattformen.",
        "mode": "Core Asset",
        "stocks": [
            ("NVDA",    "NVIDIA",            "De-facto-Monopolist für KI-Trainingschips. Jedes große KI-Modell der Welt läuft auf NVIDIA-Hardware — auch ChatGPT."),
            ("MSFT",    "Microsoft",         "Azure Cloud + OpenAI-Partnership machen Microsoft zur stärksten KI-Plattform für Unternehmen. Office-Integration schafft Lock-in."),
            ("GOOGL",   "Alphabet",          "Google DeepMind + TPU-Chips + Gemini-Modelle. Kombiniert beste KI-Forschung mit dem weltgrößten Werbenetzwerk als Cashgenerator."),
            ("PLTR",    "Palantir",          "KI-Plattform für Datenanalyse in Regierungen und Konzernen. AIP-Produkt wächst stark — profitiert von US-Verteidigungsausgaben."),
            ("AMD",     "AMD",               "NVIDIA-Alternative mit MI300X-KI-Chips. Gewinnt Marktanteile im Rechenzentrum — deutlich günstiger bewertet als NVIDIA."),
            ("CRM",     "Salesforce",        "Weltweit führendes CRM-System — KI-Integration via Einstein macht es zur Plattform für Vertriebs-Automatisierung."),
            ("SNOW",    "Snowflake",         "Cloud-Datenbankplattform — Unternehmen zahlen pro Datenmenge. KI-Workloads werden hier verarbeitet, Wachstumspotenzial enorm."),
            ("CFLT",    "Confluent",         "Real-Time-Datenstreaming — die Infrastruktur hinter KI-Entscheidungen in Echtzeit. Wird von Netflix, Uber und Co. genutzt."),
        ],
    },
    "Healthcare & Medtech": {
        "color": "#8b5cf6", "bg": "rgba(139,92,246,0.08)",
        "desc": "Gesundheit als Megatrend — Medikamente, Roboter und Diagnostik.",
        "mode": "Core Asset",
        "stocks": [
            ("NVO",     "Novo Nordisk",      "GLP-1-Medikamente (Ozempic, Wegovy) revolutionieren Diabetes und Adipositas-Behandlung. Weltmarktführer mit riesigem Pipeline."),
            ("ISRG",    "Intuitive Surgical","Quasi-Monopolist bei robotergestützter Chirurgie. Da-Vinci-Systeme stehen in tausenden OPs — Verbrauchsmaterial schafft Recurring Revenue."),
            ("DXCM",    "DexCom",            "Weltmarktführer bei kontinuierlicher Glukosemessung. Wearable-Sensor ersetzt tägliches Stechen für Diabetiker — riesiger Adressable Market."),
            ("VEEV",    "Veeva Systems",     "Cloud-Software exklusiv für Life Sciences. CRM, klinische Trials, Regulierung — Pharmaunternehmen kommen kaum raus aus dem Ökosystem."),
            ("EVO",     "Evotec",            "Hamburger Wirkstoffforschungs-Plattform für Big Pharma. Niedrig bewertet bei hohem Potenzial — echter Hidden Champion im Biotech."),
            ("LLY",     "Eli Lilly",         "Hersteller von Mounjaro (Tirzepatid) — neben Ozempic das wichtigste GLP-1-Medikament. Einer der stärksten Pharma-Wachstumswerte weltweit."),
            ("SHL.DE",  "Siemens Healthineers","Weltmarktführer in der Medizintechnik — CT, MRT, Labor. Stabiles B2B-Geschäft mit Krankenhäusern als Ankerkunden."),
            ("GEHC",    "GE HealthCare",     "Abspaltung von GE — starke Position in Bildgebung und Patientenmonitoring. Günstig bewertet bei solidem Wachstum."),
        ],
    },
    "Dividenden-Champions": {
        "color": "#f59e0b", "bg": "rgba(245,158,11,0.08)",
        "desc": "Verlässliche Ausschüttungen — Qualitätsunternehmen die dich bezahlen.",
        "mode": "Core Asset",
        "stocks": [
            ("O",       "Realty Income",     "REIT mit monatlicher Dividende — über 600 Monate in Folge. Supermärkte, Apotheken und Convenience-Stores als stabile Mieter."),
            ("JNJ",     "Johnson & Johnson", "Dividendenaristokrat seit über 60 Jahren. Pharma + Medtech-Kombination mit AAA-Kreditrating — das Stabilitäts-Fundament schlechthin."),
            ("TXN",     "Texas Instruments", "Chip-Dividendenkönig mit freiem Cashflow-Maschinen-Geschäft. Analoge Halbleiter für Industrie und Automotive — kaum disruptierbar."),
            ("MAIN",    "Main Street Capital","Business Development Company mit monatlicher Dividende ca. 6%. Investiert in US-Mittelstand — außergewöhnliche Dividendenhistorie."),
            ("BLK",     "BlackRock",         "Weltgrößter Asset Manager mit über 10 Billionen USD AUM. Wächst mit jedem Börsenanstieg automatisch — struktureller Profiteur."),
            ("PG",      "Procter & Gamble",  "Dividendenaristokrat seit 67+ Jahren — Pampers, Gillette, Ariel. Defensive Qualität die in jeder Marktphase funktioniert."),
            ("KO",      "Coca-Cola",         "Warren Buffetts Lieblingsaktie: 60+ Jahre Dividendenwachstum. Globale Marke mit Pricing Power und stabilen Free Cashflows."),
            ("ENB",     "Enbridge",          "Kanadische Pipeline-Infrastruktur — transportiert 30% des nordamerikanischen Öls. Stabile, regulierte Einnahmen, hohe Dividende."),
        ],
    },
    "Hidden Champions": {
        "color": "#ec4899", "bg": "rgba(236,72,153,0.08)",
        "desc": "Unbekannte Marktführer in Nischen — die meisten haben noch nie von ihnen gehört.",
        "mode": "Hidden Champion",
        "stocks": [
            ("CSU.TO",  "Constellation Software", "Unbekanntester Serienakquisiteur der Welt. Kauft Nischen-Software-Firmen und lässt sie nie wieder los. 30%+ CAGR über 20 Jahre."),
            ("DPLM.L",  "Diploma PLC",       "Britischer Vertrieb von Nischen-Technikkomponenten. Still, profitabel und seit 20 Jahren im Aufwärtstrend — ohne Schlagzeilen."),
            ("BESI.AS", "BE Semiconductor",  "Holländer mit Quasi-Monopol bei Halbleiter-Packaging-Equipment. Profitiert überproportional vom Chipboom — kaum jemand kennt sie."),
            ("RAA.DE",  "Rational AG",       "Deutsche Perfektion: Kombidämpfer für Gastronomieprofis. 50%+ Marktanteil, 30%+ Margen, kein Debt — der Rolls-Royce der Küche."),
            ("DB1.DE",  "Deutsche Börse",    "Infrastruktur des deutschen Finanzmarkts — Clearstream, Eurex. Profitiert von Volatilität und wächst durch Regulierung."),
            ("EXPD",    "Expeditors Intl",  "US-Logistikkonzern für internationale Fracht — asset-light, margenstarkes Modell, kaum bekannt aber systemrelevant."),
            ("RYAN",    "Ryan Specialty",    "Spezialversicherungs-Nische — bringt Risiken zusammen die kein normaler Versicherer nimmt. Starkes Wachstum."),
            ("WCH.DE",  "Wacker Chemie",     "Spezialist für Polysilizium (Solarwafer) und Silicone — Hidden Champion mit starkem Marktanteil und günstiger Bewertung."),
        ],
    },
    "Fintech & Payments": {
        "color": "#06b6d4", "bg": "rgba(6,182,212,0.08)",
        "desc": "Das globale Geldsystem wird digital — wer profitiert davon?",
        "mode": "Core Asset",
        "stocks": [
            ("V",       "Visa",              "Das größte Zahlungsnetzwerk der Welt — 4+ Milliarden Karten, 80+ Millionen Händler. Kein Kreditrisiko, reines Fee-Geschäft."),
            ("MA",      "Mastercard",        "Duopol-Partner mit Visa im globalen Payments. Wächst mit jeder Digitalisierungswelle und profitiert von Schwellenländern."),
            ("ADYEN.AS","Adyen",             "Europas führende Payment-Tech-Plattform. Ein System für Online, Mobile und POS — ohne Legacy-Systeme. Liebling der Tech-Konzerne."),
            ("NU",      "Nu Holdings",       "Die disruptivste Neobank Lateinamerikas. 90+ Millionen Kunden in Brasilien, Mexiko, Kolumbien — noch früh im Wachstumszyklus."),
            ("SOFI",    "SoFi Technologies", "US-Neobank mit Vollbank-Lizenz. Student Loans, Hypotheken, ETFs — One-Stop-Shop für Millennials. Profitiert von Zinsumfeld."),
            ("GPN",     "Global Payments",   "Zahlungsabwicklung für Händler weltweit — SaaS-Modell mit hohen Switching Costs. Günstig bewertet bei starkem Free Cashflow."),
            ("WEX",     "WEX Inc.",          "Nischenplattform für Flotten- und Healthcare-Zahlungen — wiederkehrende Einnahmen mit starkem Lock-in."),
            ("FOUR",    "Shift4 Payments",   "Schnell wachsende Payment-Plattform für Hotels, Restaurants und Stadien. Gewinnt Marktanteile von Legacy-Anbietern."),
        ],
    },
}

# ── Freitext → Tags Mapping für Radar-Suche ──────────────────────────────────
THEME_KEYWORD_MAP: dict = {
    # Technologie
    "software":       ["asset_light","recurring","quality_bias","niche_b2b"],
    "saas":           ["asset_light","recurring","quality_bias"],
    "cloud":          ["asset_light","recurring","quality_bias"],
    "internet":       ["asset_light","recurring"],
    "ki":             ["quality_bias","niche_b2b"],
    "künstliche intelligenz": ["quality_bias","niche_b2b"],
    "ai":             ["quality_bias","niche_b2b"],
    "halbleiter":     ["quality_bias","niche_b2b","specialist"],
    "chip":           ["quality_bias","niche_b2b","specialist"],
    "semiconductor":  ["quality_bias","niche_b2b","specialist"],
    "gaming":         ["asset_light","brand_moat"],
    "cybersecurity":  ["asset_light","recurring","niche_b2b"],
    # Gesundheit
    "biotech":        ["specialist","quality_bias"],
    "biotechnologie": ["specialist","quality_bias"],
    "pharma":         ["specialist","quality_bias"],
    "medtech":        ["specialist","quality_bias"],
    "medizin":        ["specialist"],
    "healthcare":     ["specialist","quality_bias"],
    "gesundheit":     ["specialist","quality_bias"],
    # Energie & Rohstoffe
    "erneuerbar":     ["commodity_sensitive","clean_energy"],
    "solar":          ["commodity_sensitive","clean_energy"],
    "wind":           ["commodity_sensitive","clean_energy"],
    "energie":        ["commodity_sensitive"],
    "öl":             ["commodity_sensitive"],
    "gas":            ["commodity_sensitive"],
    "rohstoff":       ["commodity_sensitive"],
    "gold":           ["commodity_sensitive","specialist"],
    "mining":         ["commodity_sensitive"],
    "chemie":         ["commodity_sensitive","industrial_niche"],
    # Industrie
    "industrie":      ["industrial_niche","b2b_infrastructure"],
    "automatisierung":["industrial_niche","niche_b2b"],
    "robotik":        ["industrial_niche","niche_b2b","specialist"],
    "infrastruktur":  ["b2b_infrastructure","asset_heavy"],
    "logistik":       ["industrial_niche","b2b_infrastructure"],
    "rüstung":        ["industrial_niche","b2b_infrastructure"],
    "defense":        ["industrial_niche","b2b_infrastructure"],
    # Finanzen
    "fintech":        ["asset_light","recurring"],
    "payments":       ["asset_light","recurring"],
    "bank":           ["capital_allocator"],
    "versicherung":   ["capital_allocator","recurring"],
    "insurance":      ["capital_allocator","recurring"],
    "asset management":["capital_allocator"],
    # Konsum & Lifestyle
    "luxus":          ["brand_moat","quality_bias"],
    "luxury":         ["brand_moat","quality_bias"],
    "sport":          ["brand_moat"],
    "fashion":        ["brand_moat"],
    "lebensmittel":   ["defensive","recurring"],
    "food":           ["defensive","recurring"],
    "immobilien":     ["asset_heavy","recurring"],
    "real estate":    ["asset_heavy","recurring"],
    "reit":           ["asset_heavy","recurring"],
    "e-commerce":     ["asset_light","recurring"],
    "retail":         ["brand_moat"],
    # Nischen
    "nische":         ["niche_b2b","specialist"],
    "hidden champion":["niche_b2b","specialist"],
    "dividende":      ["defensive","recurring","capital_allocator"],
    "dividend":       ["defensive","recurring"],
    "royalty":        ["asset_light","commodity_sensitive","specialist"],
}

# ── Keyword → direktes Theme-Mapping (kuratierte Stocks, kein Tag-Chaos) ────
KEYWORD_TO_THEME: dict = {
    # Healthcare
    "biotech":        "Healthcare & Medtech",
    "biotechnologie": "Healthcare & Medtech",
    "pharma":         "Healthcare & Medtech",
    "healthcare":     "Healthcare & Medtech",
    "gesundheit":     "Healthcare & Medtech",
    "medtech":        "Healthcare & Medtech",
    "medizin":        "Healthcare & Medtech",
    "arzt":           "Healthcare & Medtech",
    "krankenhaus":    "Healthcare & Medtech",
    # KI & Tech
    "ki":             "KI & Cloud",
    "ai":             "KI & Cloud",
    "künstliche intelligenz": "KI & Cloud",
    "cloud":          "KI & Cloud",
    "software":       "KI & Cloud",
    "saas":           "KI & Cloud",
    "halbleiter":     "KI & Cloud",
    "chip":           "KI & Cloud",
    "semiconductor":  "KI & Cloud",
    "nvidia":         "KI & Cloud",
    "microsoft":      "KI & Cloud",
    # Erneuerbare
    "solar":          "Erneuerbare Energien",
    "erneuerbar":     "Erneuerbare Energien",
    "wind":           "Erneuerbare Energien",
    "grün":           "Erneuerbare Energien",
    "green":          "Erneuerbare Energien",
    "energie":        "Erneuerbare Energien",
    "klimaschutz":    "Erneuerbare Energien",
    # Dividenden
    "dividende":      "Dividenden-Champions",
    "dividend":       "Dividenden-Champions",
    "ausschüttung":   "Dividenden-Champions",
    "reit":           "Dividenden-Champions",
    "income":         "Dividenden-Champions",
    # Hidden Champions
    "nische":         "Hidden Champions",
    "hidden champion":"Hidden Champions",
    "mittelstand":    "Hidden Champions",
    "versteckt":      "Hidden Champions",
    "unbekannt":      "Hidden Champions",
    # Fintech
    "fintech":        "Fintech & Payments",
    "payments":       "Fintech & Payments",
    "zahlung":        "Fintech & Payments",
    "bank":           "Fintech & Payments",
    "neobank":        "Fintech & Payments",
    "kreditkarte":    "Fintech & Payments",
    "visa":           "Fintech & Payments",
    "mastercard":     "Fintech & Payments",
}

# ── Freitext → direkte Aktien (ohne Theme, für Spezialbereiche) ───────────────
KEYWORD_DIRECT_STOCKS: dict = {
    "rüstung":    [("LMT","Lockheed Martin","Größter US-Rüstungskonzern — F-35, Patriot, Hyperschallraketen. Enormer Auftragsbestand durch NATO-Aufrüstung."),
                   ("RTX","RTX Corp","Patriot-Raketensystem, Triebwerke — Kernprofiteur der globalen Aufrüstung."),
                   ("RHEINMETALL.DE","Rheinmetall","Deutschlands führender Rüstungskonzern — Munition, Panzer, Fahrzeuge. Massiv von der Zeitenwende profitierend."),
                   ("NOC","Northrop Grumman","B-21-Stealth-Bomber, Weltraumrüstung — langfristige Staatsverträge sichern Umsatz für Dekaden."),
                   ("HO.PA","Thales","Französischer Rüstungs- und Technologiekonzern — Rüstungselektronik, Cybersecurity, Avionik.")],
    "defense":    [("LMT","Lockheed Martin","Größter US-Rüstungskonzern — F-35, Patriot, Hyperschallraketen."),
                   ("NOC","Northrop Grumman","B-21-Stealth-Bomber, Weltraumrüstung — langfristige Staatsverträge."),
                   ("RTX","RTX Corp","Patriot-Raketensystem, Pratt & Whitney Triebwerke."),
                   ("GD","General Dynamics","Stryker-Fahrzeuge, U-Boote, Gulfstream-Jets — diversifizierter Rüstungskonzern."),
                   ("RHEINMETALL.DE","Rheinmetall","Deutschlands führender Rüstungskonzern.")],
    "luxus":      [("MC.PA","LVMH","Weltgrößter Luxuskonzern — Louis Vuitton, Moët, Dior. Einzigartige Pricing Power."),
                   ("CFR.SW","Richemont","Cartier, IWC, Van Cleef & Arpels — Luxusuhren mit unerreichbaren Marktpositionen."),
                   ("RMS.PA","Hermès","Birkin Bag — absichtlich knappes Angebot schafft unbegrenzte Preismacht."),
                   ("MONCLER.MI","Moncler","Aufsteiger im Luxussegment — Daunenjacken als Statussymbol mit starkem Markenwachstum."),
                   ("EL","Estée Lauder","Prestige-Beauty — MAC, La Mer, Clinique. Turnaround-Kandidat nach Asien-Schwäche.")],
    "rohstoffe":  [("FCX","Freeport-McMoRan","Weltgrößter börsennot. Kupferproduzent — Kupfer ist das 'Öl der Energiewende'."),
                   ("RIO","Rio Tinto","Weltmarktführer bei Eisenerz und Kupfer — Infrastruktur für die Energiewende."),
                   ("BHP","BHP Group","Australischer Bergbauriese — Kupfer, Eisenerz, Nickel mit starker Dividende."),
                   ("VALE","Vale","Brasilianischer Eisenerz- und Nickelkonzern — Nickel ist kritisch für E-Auto-Batterien."),
                   ("RGLD","Royal Gold","Royalty-Modell auf Gold — kassiert Einnahmen ohne Betriebsrisiko.")],
    "gold":       [("RGLD","Royal Gold","Royalty-Modell auf Gold — kein Mining-Risiko, nur Einnahmen."),
                   ("WPM","Wheaton Precious Metals","Streaming-Modell für Silber und Gold."),
                   ("GOLD","Barrick Gold","Weltgrößter Goldproduzent — direkter Hebel auf Goldpreis."),
                   ("AEM","Agnico Eagle Mines","Canadischer Goldproduzent mit niedrigen Förderkosten."),
                   ("NEM","Newmont","Größter Goldproduzent der Welt.")],
    "immobilien": [("O","Realty Income","Monatliche Dividende — Supermärkte, Apotheken als Ankermieter."),
                   ("PLD","Prologis","Weltgrößter Logistik-REIT — Amazon und Co. mieten hier ihre Lager."),
                   ("DLR","Digital Realty","Rechenzentrum-REIT — KI-Boom treibt Nachfrage massiv."),
                   ("WELL","Welltower","Seniorenheime und Gesundheitsimmobilien — demografischer Megatrend."),
                   ("EQR","Equity Residential","US-Wohnimmobilien in Top-Städten — stabiler Cashflow.")],
    "gaming":     [("TTWO","Take-Two Interactive","GTA VI, NBA 2K — blockbuster Pipeline für nächste Jahre."),
                   ("EA","Electronic Arts","FIFA/EA Sports, Apex Legends — starkes Mobile- und Live-Service-Portfolio."),
                   ("RBLX","Roblox","Metaverse-Plattform für Gen Z — 80M+ tägl. Nutzer, starkes Creator-Ökosystem."),
                   ("NTES","NetEase","Zweitgrößter chinesischer Gaming-Konzern — International expandierend."),
                   ("NTDOY","Nintendo","Mario, Zelda, Pokemon — zeitlose IPs mit Hardware-Software-Ökosystem.")],
    # Infrastruktur
    "infrastruktur": [("ETN","Eaton","Energiemanagement-Infrastruktur — Stromverteilung für Rechenzentren und Industrie. Profiteur der Elektrifizierung."),
                      ("PWR","Quanta Services","Baut und wartet Strom- und Glasfasernetze — jede Energiewende braucht sein Netzwerk."),
                      ("SU.PA","Schneider Electric","Weltmarktführer für Energiemanagement und Automatisierung — von Smart Buildings bis Rechenzentrum."),
                      ("FERG","Ferguson","Führender Verteiler für Sanitär- und HVAC-Infrastruktur — kritische Versorgungskette."),
                      ("ACM","AECOM","Weltgrößtes Ingenieur- und Infrastrukturunternehmen — plant Brücken, Tunnels, Wasserwerke."),
                      ("J","Jacobs Solutions","Infrastruktur, Umwelt, Verteidigung — staatliche Großaufträge weltweit.")],
    "infrastructure": [("ETN","Eaton","Energiemanagement-Infrastruktur — Profiteur der Elektrifizierung."),
                       ("PWR","Quanta Services","Baut Strom- und Glasfasernetze — jede Energiewende braucht sein Netzwerk."),
                       ("SU.PA","Schneider Electric","Weltmarktführer für Energiemanagement."),
                       ("BIP","Brookfield Infrastructure","Größter globaler Infrastruktur-Investor — Häfen, Pipelines, Glasfaser, Versorgungsnetze."),
                       ("AWK","American Water Works","Größter US-Wasserversorger — regulierte Einnahmen, defensives Wachstum.")],
    # Cybersecurity
    "cybersecurity":  [("CRWD","CrowdStrike","Marktführer bei KI-gestützter Endpoint-Security — Cloud-nativ, starkes ARR-Wachstum."),
                       ("PANW","Palo Alto Networks","Umfassendste Security-Plattform — konsolidiert viele Tools in einem. Starker Free Cashflow."),
                       ("ZS","Zscaler","Zero-Trust-Netzwerksicherheit in der Cloud — Unternehmen können nicht ohne."),
                       ("FTNT","Fortinet","Firewall-Spezialist mit günstigem Hardware+Software-Modell — starke SMB-Präsenz."),
                       ("S","SentinelOne","KI-Konkurrent zu CrowdStrike — wächst schnell, noch unprofitabel aber mit hohem Potenzial.")],
    "security":       [("CRWD","CrowdStrike","KI-gestützte Endpoint-Security — Cloud-nativ."),
                       ("PANW","Palo Alto Networks","Umfassendste Security-Plattform."),
                       ("OKTA","Okta","Identity-Management — jeder Login geht durch Okta. SaaS-Modell mit starkem Lock-in."),
                       ("ZS","Zscaler","Zero-Trust-Netzwerksicherheit — unverzichtbar für Remote-Arbeit."),
                       ("CIBR","ETF","First Trust Cybersecurity ETF — diversifizierter Zugang zum gesamten Sektor.")],
    # Robotik & Automatisierung
    "robotik":        [("ISRG","Intuitive Surgical","Da Vinci-Roboterchirurgie — Quasi-Monopol mit hohem Recurring Revenue."),
                       ("ABB","ABB Ltd","Schweizer Industrierobotik- und Automatisierungskonzern — breite Kundenbasis."),
                       ("ROK","Rockwell Automation","Industrieautomatisierung made in USA — Profiteur von Nearshoring."),
                       ("FANUC","Fanuc Corp","Japanischer Weltmarktführer bei Industrierobotern — hohe Margen, netto schuldenfrei."),
                       ("BRKS","Brooks Automation","Robotik für Halbleiterfertigung — jeder Chip läuft durch Brooks-Systeme.")],
    "automatisierung":[("ROK","Rockwell Automation","Industrieautomatisierung — Profiteur von Nearshoring und Reshoring."),
                       ("HON","Honeywell","Diversifizierter Automatisierungskonzern — Gebäudetechnik bis Aerospace."),
                       ("ABB","ABB Ltd","Schweizer Industrierobotik- und Automatisierungskonzern."),
                       ("ITRI","Itron","Smart Grid und Smart Metering — Infrastruktur für intelligente Energienetze."),
                       ("AVAV","AeroVironment","Drohnen für Militär und Energie-Inspektion — wächst mit Verteidigungsbudgets.")],
    # E-Commerce & Plattformen
    "e-commerce":     [("SHOP","Shopify","Die Plattform hinter Millionen Online-Shops — konkurriert mit Amazon durch Händler-Empowerment."),
                       ("MELI","MercadoLibre","Das Amazon Lateinamerikas — dominiert E-Commerce, Payments und Logistik in 18 Ländern."),
                       ("SE","Sea Limited","Südostasiens E-Commerce- und Gaming-Riese — Shopee, Garena, SeaMoney in einer Aktie."),
                       ("ETSY","Etsy","Nischen-Marktplatz für Handgemachtes — höhere Margen als Amazon durch Differenzierung."),
                       ("PDD","PDD Holdings","Temu-Mutter — disruptives Preismodell erobert westliche Märkte rasant.")],
    # Nachhaltigkeit / ESG
    "nachhaltigkeit": [("NEE","NextEra Energy","Größter Windkraft-Betreiber der USA — Renewable Leader."),
                       ("ORSTED.CO","Ørsted","Offshore-Wind-Pionier aus Dänemark."),
                       ("HASI","Hannon Armstrong","Finanzierung nachhaltiger Infrastruktur — reine ESG-Investmentstory."),
                       ("CSCO","Cisco","Netzwerk-Infrastruktur mit starkem ESG-Rating und Dividende."),
                       ("VWS.CO","Vestas Wind","Weltmarktführer bei Windturbinen — direkte Wette auf Energiewende.")],
    "esg":            [("NEE","NextEra Energy","Renewable-Leader mit regulierten Einnahmen."),
                       ("HASI","Hannon Armstrong","Spezialist für ESG-Infrastrukturfinanzierung."),
                       ("ORSTED.CO","Ørsted","Offshore-Wind-Pionier."),
                       ("DSM-FIRMENICH.AS","DSM-Firmenich","Nutrition, Health und Biosciences — starkes ESG-Profil in der Lebensmittelindustrie."),
                       ("WM","Waste Management","Marktführer bei Recycling und Abfallentsorgung — Profiteur der Kreislaufwirtschaft.")],
    # Emerging Markets
    "emerging markets":[("MELI","MercadoLibre","Das Amazon Lateinamerikas."),
                        ("NU","Nu Holdings","Lateinamerikas schnellst wachsende Neobank — 90M+ Kunden."),
                        ("INFY","Infosys","Indischer IT-Riese — profitiert von globalem Outsourcing-Trend."),
                        ("SE","Sea Limited","Südostasiens Digital-Champion."),
                        ("BABA","Alibaba","Chinas E-Commerce-Riese — stark günstig bewertet, aber regulatorisches Risiko.")],
    "indien":          [("INFY","Infosys","Indischer IT-Dienstleistungsriese — globale Digitalisierung treibt Nachfrage."),
                        ("WIT","Wipro","IT-Services aus Indien — günstig bewertet mit starkem Wachstum."),
                        ("HDB","HDFC Bank","Beste private Bank Indiens — starkes Kreditwachstum in aufstrebender Mittelklasse."),
                        ("INDY","iShares India ETF","Diversifizierter ETF-Zugang zum indischen Markt."),
                        ("IBN","ICICI Bank","Zweitgrößte private Bank Indiens — profitiert vom Banken-Boom.")],
    # Consumer & Marken
    "sport":           [("NKE","Nike","Weltmarktführer bei Sportbekleidung — Direct-to-Consumer Transformation läuft."),
                        ("ADDYY","Adidas","Europas Sportmarke Nr. 1 — Comeback nach Yeezy-Krise, günstig bewertet."),
                        ("LULU","Lululemon","Premium-Athleisure — hohe Margen, loyale Community, starkes International-Wachstum."),
                        ("NVO","Novo Nordisk","Ozempic ist auch ein Lifestyle-Produkt — Überschneidung Sport/Health."),
                        ("ONON","On Holding","Schweizer Sneaker-Brand — rasantes Wachstum durch premium Positionierung.")],
    "lebensmittel":    [("NESN.SW","Nestlé","Weltgrößter Lebensmittelkonzern — Nescafé, Maggi, KitKat. Klassischer Defensivwert."),
                        ("PEP","PepsiCo","Getränke und Snacks (Lay's, Doritos) — diversifizierter als Coca-Cola."),
                        ("KO","Coca-Cola","Buffetts Liebling — 60+ Jahre Dividendenwachstum, globale Marke."),
                        ("MDLZ","Mondelez","Oreo, Milka, Toblerone — globaler Snack-Riese mit starker Pricing Power."),
                        ("SYY","Sysco","Weltgrößter Food-Service-Verteiler — Restaurants kommen nicht ohne Sysco aus.")],
}

def radar_search_by_keyword(query: str, current_ticker: str = "", limit: int = 6) -> list:
    """Sucht Radar-Stocks — Stufe 1: direkte Stocks, 2: Theme, 3: Tags, 4: Discovery-Mix."""
    q = query.lower().strip()

    # Stufe 0: Direkter Keyword-Match auf spezifische Aktien-Listen
    for kw, stocks in KEYWORD_DIRECT_STOCKS.items():
        if kw in q or q in kw or (len(q) >= 4 and q[:4] in kw) or (len(kw) >= 4 and kw[:4] in q):
            return stocks[:limit]

    # Stufe 1: direktes Theme-Mapping → kuratierte, korrekte Stocks
    matched_theme = None
    for kw, theme in KEYWORD_TO_THEME.items():
        if kw in q or q in kw:
            matched_theme = theme
            break

    if matched_theme and matched_theme in VELOX_RADAR_THEMES:
        th = VELOX_RADAR_THEMES[matched_theme]
        return [(tk, name, why) for tk, name, why in th["stocks"]][:limit]

    # Stufe 2: Tag-basierter Fallback (für unbekannte Begriffe)
    tags = []
    for kw, kw_tags in THEME_KEYWORD_MAP.items():
        if kw in q or q in kw:
            tags.extend(kw_tags)
    tags = list(dict.fromkeys(tags))
    if tags:
        results = get_radar_stocks(tags, current_ticker=current_ticker, limit=limit)
        if results:
            return results

    # Stufe 4: Discovery-Fallback — interessante Stocks aus allen Themes
    import random
    _all_discovery = []
    for _th in VELOX_RADAR_THEMES.values():
        _all_discovery.extend(_th.get("stocks", []))
    random.shuffle(_all_discovery)
    return _all_discovery[:limit]

def get_radar_stocks(tags: list, current_ticker: str = "", limit: int = 6,
                     sector: str = "", bm: str = "") -> list:
    """Velox Radar: mehrstufiges Matching — Tags → Sektor → Business Model.
    Gibt (ticker, name, warum) zurück — schließt current_ticker aus."""
    seen = set(); results = []
    ck = (current_ticker or "").upper().split(".")[0]

    def _add(tk, name, why):
        tk_base = tk.upper().split(".")[0]
        if tk_base != ck and tk_base not in seen:
            seen.add(tk_base)
            results.append((tk, name, why))

    # Stufe 1: direkte Tag-Übereinstimmung (höchste Qualität)
    for tag in tags:
        for tk, name, why in VELOX_RADAR.get(tag, []):
            _add(tk, name, why)
            if len(results) >= limit: return results

    # Stufe 2: Sektor-basiertes Fallback
    SECTOR_TAG_MAP = {
        "Technology": ["quality_bias","asset_light","niche_b2b","recurring"],
        "Healthcare":  ["specialist","quality_bias"],
        "Financial Services": ["capital_allocator","recurring"],
        "Consumer Defensive": ["defensive","recurring"],
        "Consumer Cyclical":  ["brand_moat","quality_bias"],
        "Industrials": ["industrial_niche","niche_b2b","b2b_infrastructure"],
        "Basic Materials": ["commodity_sensitive","specialist"],
        "Energy": ["commodity_sensitive"],
        "Utilities": ["defensive","regulated"],
        "Real Estate": ["recurring","asset_heavy"],
        "Communication Services": ["asset_light","recurring"],
    }
    fallback_tags = SECTOR_TAG_MAP.get(sector, [])
    for tag in fallback_tags:
        if len(results) >= limit: break
        for tk, name, why in VELOX_RADAR.get(tag, []):
            _add(tk, name, why)
            if len(results) >= limit: break

    # Stufe 3: Business-Model-Fallback (breitestes Netz)
    if len(results) < 2 and bm:
        from ace_stock_check import SIMILAR_STOCKS
        bm_key = next((k for k in SIMILAR_STOCKS if k.lower() in bm.lower()), None)
        if bm_key:
            for tk, name, why in (SIMILAR_STOCKS.get(bm_key, {}).get("Core Asset", [])
                                  + SIMILAR_STOCKS.get(bm_key, {}).get("Hidden Champion", [])):
                _add(tk, name, why)
                if len(results) >= limit: break

    return results[:limit]

@st.cache_data(ttl=3600, show_spinner=False)
def get_theme_stocks_with_scores(theme_name: str) -> list:
    """Lädt Scores für alle Aktien eines Themes — gecacht 1 Stunde."""
    theme = VELOX_RADAR_THEMES.get(theme_name, {})
    mode  = theme.get("mode", "Core Asset")
    result = []
    for tk, name, why in theme.get("stocks", []):
        data = _card_full_data(tk, mode)
        result.append((tk, name, why, data))
    return result

def classify_business_profile(profile, metrics):
    if not has_story_basis(profile): return None
    sector = (profile.get("sector") or "").strip()
    industry = (profile.get("industry") or "").strip()
    summary = (profile.get("summary") or "").strip()
    name = (profile.get("name") or "").strip()
    text = lower_text(sector, industry, summary, name)
    bm = "Diversifiziert"; chars = []; strengths = []; risks = []; tags = []
    if any(k in text for k in ["royalty","streaming","stream interests","production-based interests"]):
        bm="Royalty / Streaming"; tags+=["asset_light","commodity_sensitive","specialist"]
        strengths+=["Weniger operative Asset-Risiken als klassische Produzenten."]
        risks+=["Abhängigkeit von Rohstoffpreisen bleibt."]
    elif any(k in text for k in ["software","saas","cloud","subscription","subscriptions"]):
        bm="Software / SaaS"; tags+=["asset_light","recurring","quality_bias"]
        strengths+=["Skalierbares, wiederkehrendes Geschäftsmodell."]
        risks+=["Bewertung kann schnell ambitioniert werden."]
    elif any(k in text for k in ["semiconductor","chip","wafer","lithography"]):
        bm="Halbleiter / Chip-Zulieferer"; tags+=["niche_b2b","cyclical","quality_bias"]
        strengths+=["Technologische Eintrittsbarrieren können hoch sein."]
        risks+=["Zyklisch und capex-getrieben."]
    elif any(k in text for k in ["medical device","diagnostic","diagnostics","medtech","life sciences"]):
        bm="MedTech / Diagnostics"; tags+=["niche_b2b","defensive","quality_bias"]
        strengths+=["Spezialisierte, schwer austauschbare Produkte."]
        risks+=["Regulatorik und Produktzyklen können bremsen."]
    elif any(k in text for k in ["insurance","insurer","reinsurance"]):
        bm="Versicherung"; tags+=["defensive","cashflow","regulated"]
        strengths+=["Planbares Core-Geschäft."]; risks+=["Kapitalmarktzyklen bleiben relevant."]
    elif any(k in text for k in ["utility","utilities","grid","transmission","distribution network"]):
        bm="Versorger / Netz"; tags+=["defensive","regulated","capital_intensive"]
        strengths+=["Reguliertes, planbares Geschäft."]; risks+=["Kapitalintensität begrenzt Flexibilität."]
    elif any(k in text for k in ["platform","marketplace","payments","payment network"]):
        bm="Plattform / Payments"; tags+=["asset_light","quality_bias","recurring"]
        strengths+=["Starke Skaleneffekte."]; risks+=["Oft hoch bewertet."]
    elif any(k in text for k in ["industrial","automation","controls","electrical","motion","components","equipment"]):
        bm="Industrie / Zulieferer"; tags+=["niche_b2b","cyclical"]
        strengths+=["B2B-Nische oder Systemrelevanz."]; risks+=["Konjunkturabhängig."]
    elif any(k in text for k in ["gold","silver","copper","uranium","mining","minerals","metals"]):
        bm="Rohstoff / Mining"; tags+=["commodity_sensitive","cyclical","capital_intensive"]
        strengths+=["Profitiert stark im Rohstoffzyklus."]; risks+=["Zyklisch, preisgetrieben, kapitalintensiv."]
    elif any(k in text for k in ["consumer","brand","apparel","beverage","food","retail"]):
        bm="Konsum / Marke"; tags+=["consumer"]
        strengths+=["Markenstärke gibt Pricing Power."]; risks+=["Oft zu bekannt für HC."]
    elif any(k in text for k in ["energy services","oilfield","oil","gas","drilling"]):
        bm="Energie / Services"; tags+=["cyclical","commodity_sensitive","capital_intensive"]
        strengths+=["Profitiert im Energieaufschwung."]; risks+=["Hohe Zyklik."]
    elif any(k in text for k in ["rail","airport","toll road","concession","infrastructure"]):
        bm="Infrastruktur / Konzessionen"; tags+=["cashflow","quality_bias","capital_intensive"]
        strengths+=["Robuste Langfrist-Assets."]; risks+=["Kapitalintensität + Zinsumfeld."]
    char_map = {"asset_light":"asset-light","recurring":"wiederkehrende Umsätze",
                "quality_bias":"Qualitäts-Tendenz","niche_b2b":"B2B-/Nischencharakter",
                "defensive":"eher defensiv","regulated":"reguliert",
                "commodity_sensitive":"rohstoffsensitiv","cyclical":"zyklisch",
                "capital_intensive":"kapitalintensiv"}
    chars = [label for t, label in char_map.items() if t in tags]
    cs = 5.8; cr = []
    if "defensive" in tags: cs += 0.9; cr.append("Defensiver Charakter stützt Core-Fit.")
    if "recurring" in tags: cs += 0.8; cr.append("Wiederkehrende Umsätze helfen der Ruhe.")
    if "asset_light" in tags: cs += 0.5; cr.append("Asset-light → robuste Kapitalrenditen.")
    if "quality_bias" in tags: cs += 0.4; cr.append("Qualitäts-/Systemcharakter stützt Core-Fit.")
    if "regulated" in tags: cs += 0.3; cr.append("Regulierung macht Erträge planbarer.")
    if "commodity_sensitive" in tags: cs -= 0.6; cr.append("Rohstoffabhängigkeit macht Core-Fit unruhiger.")
    if "cyclical" in tags: cs -= 0.5; cr.append("Zyklik drückt auf Compounder-Charakter.")
    if "capital_intensive" in tags: cs -= 0.3; cr.append("Kapitalintensität erhöht Hebel.")
    if bm == "Royalty / Streaming": cs += 0.4; cr.append("Defensiver als klassische Produzenten.")
    hs = 5.8; hr = []; mcap = metrics.get("mcap")
    if mcap:
        if 1 <= mcap <= 25: hs += 0.8; hr.append("Small/Mid-Cap passt zum HC-Profil.")
        elif mcap > 100: hs -= 0.7; hr.append("Zu groß für klassischen HC.")
    if "niche_b2b" in tags: hs += 0.9; hr.append("B2B-Nischencharakter stützt HC-Fit.")
    if "quality_bias" in tags: hs += 0.3; hr.append("Qualitative Sonderrolle unterstützt HC.")
    if "asset_light" in tags: hs += 0.2; hr.append("Asset-light begünstigt Re-Rating.")
    if "cyclical" in tags: hs += 0.2; hr.append("Zyklik macht HC spannender (kontrolliert).")
    if bm in ["Konsum / Marke","Versicherung","Versorger / Netz"]: hs -= 0.6; hr.append("Zu breit / bekannt für klassischen HC.")
    if bm == "Royalty / Streaming": hs -= 0.2; hr.append("Eher Spezialwert als klassischer HC.")
    return {"business_model": bm, "characteristics": list(dict.fromkeys(chars)),
            "strengths": list(dict.fromkeys(strengths)), "risks": list(dict.fromkeys(risks)),
            "core_fit": clip_score(cs), "core_reasons": list(dict.fromkeys(cr)),
            "hc_fit": clip_score(hs), "hc_reasons": list(dict.fromkeys(hr)),
            "sector": sector, "industry": industry, "summary": summary, "tags": tags}

# ══════════════════════════════════════════════════════════════════════════════
# Entry Triggers  ← NEU
# ══════════════════════════════════════════════════════════════════════════════
def build_entry_triggers(mode, metrics, timing_score, chart_df, has_position, buy_price_val, last_price):
    """Konkrete, zeitpunktgebundene Bedingungen — WANN und WIE handeln."""
    triggers = []
    lc = ma20 = ma50 = ma200 = rsi = macd_c = sig_c = macd_p = sig_p = None
    if chart_df is not None and not chart_df.empty:
        lat = chart_df.iloc[-1]
        lc = safe_float(lat.get("Close")); ma20 = safe_float(lat.get("MA20"))
        ma50 = safe_float(lat.get("MA50")); ma200 = safe_float(lat.get("MA200"))
        rsi = safe_float(lat.get("RSI14"))
        macd_c = safe_float(lat.get("MACD")); sig_c = safe_float(lat.get("MACD_Signal"))
    if chart_df is not None and len(chart_df) >= 2:
        prev = chart_df.iloc[-2]
        macd_p = safe_float(prev.get("MACD")); sig_p = safe_float(prev.get("MACD_Signal"))

    # Übergeordneter Trend
    if ma50 and ma200:
        if ma50 < ma200:
            triggers.append(f"Trendkontext: MA50 ({ma50:.2f}) < MA200 ({ma200:.2f}) — übergeordnet negativ. "
                            "Größeren Einsatz erst nach Umkehr dieser Konstellation.")
        else:
            triggers.append(f"Trendkontext: MA50 ({ma50:.2f}) > MA200 ({ma200:.2f}) — struktureller Rückenwind vorhanden.")

    # MA20-Lage → Entry-Zone
    if lc and ma20:
        d20 = (lc / ma20) - 1.0
        if abs(d20) <= 0.02:
            triggers.append(f"Entry-Zone aktiv: Kurs ({lc:.2f}) nahe MA20 ({ma20:.2f}, {d20*100:+.1f}%) — "
                            "bevorzugtes Einstiegsfenster ist jetzt offen.")
        elif d20 > 0.07:
            triggers.append(f"Abwarten: Kurs {d20*100:.1f}% über MA20 ({ma20:.2f}) — "
                            "warte auf Rücksetzer zur MA20 als günstigeres Entry.")
        elif 0.02 < d20 <= 0.07:
            triggers.append(f"Leicht über MA20 ({d20*100:.1f}%): kein ideales Entry, "
                            "aber kleine erste Tranche bei starkem Fundament vertretbar.")
        elif d20 < -0.04:
            triggers.append(f"Unter MA20 ({d20*100:.1f}%): warte auf Tagesschluss über MA20 ({ma20:.2f}) "
                            "als Einstiegssignal.")

    # RSI-Filter (Wilder's)
    if rsi is not None:
        if rsi > 70:
            triggers.append(f"RSI-Filter: {rsi:.1f} → überkauft. Kein neuer Entry; erst nach Abkühlung auf ~50–55.")
        elif 45 <= rsi <= 58:
            triggers.append(f"RSI-Timing: {rsi:.1f} → gut abgekühlt — günstiger Bereich für Einstieg oder Nachkauf.")
        elif rsi < 38:
            triggers.append(f"RSI {rsi:.1f} → schwach. Erst bei Stabilisierung + 2 aufeinanderfolgenden grünen "
                            "Tagen Entry erwägen.")

    # MACD-Signal
    if all(v is not None for v in [macd_c, sig_c, macd_p, sig_p]):
        crossed_up = macd_p <= sig_p and macd_c > sig_c
        crossed_dn = macd_p >= sig_p and macd_c < sig_c
        if crossed_up:
            triggers.append(f"MACD-Kreuzung bullisch: Linie kreuzte Signal nach oben — frisches Kaufsignal, zeitnah handeln.")
        elif crossed_dn:
            triggers.append(f"MACD-Kreuzung bärisch: Linie kreuzte Signal nach unten — Momentum dreht, Entry verschieben.")
        elif macd_c > sig_c:
            triggers.append(f"MACD bullisch ({macd_c:.3f} > {sig_c:.3f}) — Momentum stützt Entry.")
        else:
            triggers.append(f"MACD bärisch ({macd_c:.3f} < {sig_c:.3f}) — Gegenwind; auf MA20 + RSI warten bevor Entry.")

    # 52-Wochen-Position
    h52 = metrics.get("high52"); l52 = metrics.get("low52")
    lc_52 = lc or last_price
    if h52 and l52 and lc_52 and h52 > l52:
        pos52 = (lc_52 - l52) / (h52 - l52) * 100
        if not (0 <= pos52 <= 100):
            triggers.append("52W-Daten nicht aktuell — bitte 'Auto von Yahoo' klicken für korrekte Range-Einordnung.")
        else:
            if pos52 <= 25:
                triggers.append(f"52W-Position: {pos52:.0f}% der Jahresrange (Tief {l52:.2f} / Hoch {h52:.2f}) — "
                                "nahe Jahrestief, mögliches Schnäppchen. Bodenbildung (grüne Tage + Volumen) erst bestätigen.")
            elif pos52 >= 85:
                triggers.append(f"52W-Position: {pos52:.0f}% — nahe Jahreshoch ({h52:.2f}), wenig Luft aus Sicht der Range. "
                                "Nur bei klarem Momentum-Setup einsteigen.")
            elif 35 <= pos52 <= 65:
                triggers.append(f"52W-Position: {pos52:.0f}% — solide Mitte der Jahresrange, ausgewogenes Chance/Risiko.")
            else:
                triggers.append(f"52W-Position: {pos52:.0f}% (Tief: {l52:.2f} / Hoch: {h52:.2f}).")

    # Portfolio-spezifische Trigger
    if has_position and buy_price_val and last_price:
        perf = (last_price / buy_price_val) - 1.0
        if perf > 0.15:
            triggers.append(f"Nachkauf-Disziplin: Position bereits +{perf*100:.1f}% im Plus. "
                            "Nachkauf nur bei klarem Pullback zur MA20, nicht der Stärke hinterherlaufen.")
        elif perf < -0.12:
            triggers.append(f"Verlust-Kontrolle: Position {perf*100:.1f}%. Kein emotionaler Nachkauf. "
                            "Erst wenn Timing-Score > 6 und Kurs über MA20.")
        elif -0.05 <= perf <= 0.05:
            triggers.append(f"Break-even-Zone ({perf*100:+.1f}%): Geduld. "
                            "Nachkauf erst wenn RSI 45–58 UND Kurs nahe/über MA20.")

    # Modus-Hinweis
    if not has_position:
        if mode == "Core Asset":
            triggers.append("Core-Einstieg: Staffelung empfohlen — erste Tranche bei Bestätigung, "
                           "zweite nach weiterem Rücksetzer oder positiver Quartalszahl.")
        else:
            triggers.append("HC-Einstieg: Klein starten (max. 5–7% Zielgewicht), "
                           "Story und Chart müssen sich gegenseitig bestätigen.")

    if not triggers:
        triggers.append("Kein klares Entry-Signal erkennbar. Watchlist und beobachten.")
    return triggers

# ══════════════════════════════════════════════════════════════════════════════
# Risk Hints  ← NEU
# ══════════════════════════════════════════════════════════════════════════════
def build_risk_hints(mode, metrics, story_info, fund_score, timing_score, chart_df):
    """Was könnte die These zerreißen? Was muss genau beobachtet werden?"""
    risks = []
    beta = metrics.get("beta"); peg = metrics.get("peg")
    ps = metrics.get("ps"); pe = metrics.get("pe"); mcap = metrics.get("mcap")

    # Bewertungsrisiken
    if ps is not None and ps > 12:
        risks.append(f"Multiple-Kompression: KUV {ps:.1f} lässt wenig Fehlertoleranz. "
                    "Verpasste Umsatzziele treffen den Kurs überproportional.")
    elif ps is not None and ps > 7:
        risks.append(f"Bewertungspuffer: KUV {ps:.1f} erhöht — Revenue-Wachstum im Auge behalten.")
    if pe is not None and pe > 45:
        risks.append(f"KGV-Risiko: {pe:.1f} lässt kaum Spielraum für Enttäuschungen. "
                    "Zinsanstieg oder Verlangsamung können scharf treffen.")
    if peg is not None and peg > 5:
        risks.append(f"Wachstumsprämie: PEG {peg:.2f} — Markt preist viel Wachstum ein. "
                    "Wenn Lieferung ausbleibt, droht Neubewertung nach unten.")

    # Volatilitäts-/Beta-Risiko
    if beta is not None and beta > 1.3:
        risks.append(f"Marktrisiko: Beta {beta:.2f} → in breiter Korrektur überproportionaler Rückgang. "
                    "Position-Sizing entsprechend klein halten.")

    # Business-/Sektor-Risiken
    if story_info:
        tags = story_info.get("tags", []); bm = story_info.get("business_model", "")
        if "commodity_sensitive" in tags:
            risks.append(f"Rohstoffrisiko ({bm}): Kurs hängt stark vom Commodity-Preis ab — "
                        "externer Faktor außer deiner Kontrolle.")
        if "cyclical" in tags:
            risks.append(f"Zyklusrisiko ({bm}): Nachfrage kann in Abschwung schnell einbrechen. "
                        "ISM-Index und Auftragseingänge als Frühindikator.")
        if "capital_intensive" in tags:
            risks.append("Kapitalrisiko: Hoher Capex-Bedarf — steigende Zinsen oder schwache Cash Flows "
                        "können Investitionen und Dividende unter Druck setzen.")
        if mode == "Hidden Champion" and mcap and mcap < 5:
            risks.append(f"Liquiditätsrisiko: Marktkapitalisierung {mcap:.1f} Mrd → "
                        "niedrige Markttiefe, Kurs kann bei Nachrichten stark ausschlagen.")
        if mode == "Hidden Champion" and "niche_b2b" in tags:
            risks.append("Konzentrationsrisiko: B2B-Nische kann hohe Kundenkonzentration bedeuten — "
                        "Verlust eines Großkunden trifft überproportional.")

    # Charttechnische Risiken
    if chart_df is not None and not chart_df.empty:
        lat = chart_df.iloc[-1]
        ma50 = safe_float(lat.get("MA50")); ma200 = safe_float(lat.get("MA200"))
        rsi_v = safe_float(lat.get("RSI14")); lc = safe_float(lat.get("Close"))
        if ma50 and ma200 and ma50 < ma200:
            risks.append(f"Trendrisiko: MA50 ({ma50:.2f}) unter MA200 ({ma200:.2f}) → "
                        "übergeordnete Struktur negativ, erster Rücksetzer kann tiefer gehen.")
        if rsi_v and rsi_v > 72:
            risks.append(f"Momentum-Risiko: RSI {rsi_v:.1f} → überkauft, kurzfristige Korrektur wahrscheinlicher.")
        if lc and ma50 and lc < ma50 * 0.95:
            risks.append(f"Unterstützungsrisiko: Kurs ({lc:.2f}) klar unter MA50 ({ma50:.2f}) — "
                        "nächste Haltelinie beobachten.")

    # Fundamentalwarnung
    if fund_score is not None and fund_score < 5.5:
        risks.append(f"Fundamentalrisiko: Score {fund_score:.1f} — Basis nicht stark genug "
                    "für aggressiven Einstieg.")

    if not risks:
        risks.append("Keine spezifischen Hochrisiko-Signale identifiziert. "
                    "Basisrisiken (Markt, Sektor, Einzelwert) immer im Blick.")
    return risks

# ══════════════════════════════════════════════════════════════════════════════
# Total Score
# ══════════════════════════════════════════════════════════════════════════════
def overall_score(mode, fund_score, timing_score, story_score):
    if all(v is None for v in [fund_score, timing_score, story_score]): return None
    if story_score is None:
        w = {"fund":0.60,"timing":0.40} if mode == "Core Asset" else {"fund":0.50,"timing":0.50}
    else:
        w = ({"fund":0.45,"timing":0.30,"story":0.25} if mode == "Core Asset"
             else {"fund":0.30,"timing":0.30,"story":0.40})
    vals = {"fund":fund_score,"timing":timing_score,"story":story_score}
    active = {k:v for k,v in vals.items() if v is not None and k in w}
    if not active: return None
    ws = sum(w[k] for k in active)
    return clip_score(sum(vals[k]*w[k] for k in active) / ws)

# ══════════════════════════════════════════════════════════════════════════════
# Red Flags
# ══════════════════════════════════════════════════════════════════════════════
def build_red_flags(mode, metrics, timing_score, story_score, has_position, portfolio_total, position_value):
    flags = []
    beta=metrics.get("beta"); peg=metrics.get("peg"); ps=metrics.get("ps"); pe=metrics.get("pe")
    if mode == "Core Asset":
        if beta and beta > 1.35: flags.append(f"Core-Flag: Beta {beta:.2f} für Compounder zu hoch.")
        if peg and peg > 5: flags.append(f"Core-Flag: PEG {peg:.2f} sehr ambitioniert.")
        if ps and ps > 12: flags.append(f"Core-Flag: KUV {ps:.1f} braucht operative Qualität.")
        if pe and pe > 45: flags.append(f"Core-Flag: KGV {pe:.1f} klar ambitioniert.")
    if timing_score and timing_score < 5: flags.append("Timing-Flag: Chartbild aktuell unruhig.")
    if story_score:
        if mode == "Core Asset" and story_score < 5.3: flags.append("Business-Flag: Modell nicht ganz sauber für Core.")
        if mode == "Hidden Champion" and story_score < 5.3: flags.append("Business-Flag: Profil nicht überzeugend HC-typisch.")
    if has_position:
        w = compute_position_weight(portfolio_total, position_value)
        if w and w >= 0.10: flags.append(f"Portfolio-Flag: Position mit {w*100:.1f}% bereits groß.")
    return flags

# ══════════════════════════════════════════════════════════════════════════════
# Fazit
# ══════════════════════════════════════════════════════════════════════════════
def build_fazit(mode, fund_score, timing_score, story_score, story_info, total,
                has_position, buy_price, portfolio_total, position_value, last_price):
    if total is None: return "Bitte erst Fundament oder Chart starten.", "", []
    label = "Core Asset" if mode == "Core Asset" else "Hidden Champion"
    gates = []
    if fund_score:
        if mode == "Core Asset" and fund_score < 5.6: gates.append("Fundament < 5,6 → kein sauberes Core-Neukauf-Setup.")
        if mode == "Hidden Champion" and fund_score < 5.2: gates.append("Fundament < 5,2 → HC nur tragbar wenn Timing/Story klar helfen.")
    if timing_score:
        if timing_score < 5: gates.append("Timing < 5 → unruhig (gestaffelt/geduldig).")
        elif timing_score >= 7: gates.append("Timing ≥ 7 → Entry-Fenster offen (ohne Kaufdruck).")
    if story_score:
        if mode == "Core Asset" and story_score < 5.5: gates.append("Story-Fit < 5,5 → Modell passt nur bedingt in Core.")
        elif mode == "Hidden Champion" and story_score >= 7: gates.append("Story-Fit ≥ 7 → HC-typisch genug für nähere Prüfung.")
    perf = status = None
    if has_position and buy_price and last_price and buy_price > 0 and last_price > 0:
        perf = (last_price / buy_price) - 1.0
        if perf > 0.03: status = "läuft (klar im Plus)"
        elif perf >= -0.03: status = "neutral (um Break-even)"
        else: status = "unter Wasser (im Minus)"
    action = ""; why = []
    if has_position and perf is not None:
        if perf > 0.03:
            action = "Halten (Position läuft)."; why = [f"Seit Kauf {percent(perf)} → gutes Signal."]
            if (fund_score and fund_score >= 6.2) and (not timing_score or timing_score >= 6):
                action = "Halten + Nachkauf möglich (gestaffelt)."; why.append("Fundament und Timing nicht im Weg.")
            elif timing_score and timing_score < 5: why.append("Timing unruhig → gestaffelt, kein Druck.")
        elif -0.03 <= perf <= 0.03:
            action = "Halten / Beobachten."; why = ["Um Break-even ist Geduld oft beste Entscheidung."]
        else:
            action = "Beobachten (kein Nachkauf ins fallende Messer)."; why = [f"Seit Kauf {percent(perf)} im Minus."]
    if not action:
        if mode == "Core Asset":
            if fund_score and fund_score < 5.6: action="Beobachten / kein Einstieg."; why=["Basis fehlt für Core-Entry."]
            elif total >= 7.5 and (not timing_score or timing_score >= 6): action="Einstieg/Nachkauf möglich (gestaffelt)."; why=["Fundament und Timing tragfähig."]
            elif total >= 6.5: action="Beobachten oder klein starten."; why=["Interessant, aber noch nicht klar."]
            elif total >= 5.7: action="Beobachten."; why=["Substanz da, mehr Bestätigung nötig."]
            else: action="Kein Einstieg."; why=["Setup nicht sauber genug."]
        else:
            if fund_score and fund_score < 5.2 and (not story_score or story_score < 6.8):
                action="Eher streichen / Watchlist."; why=["HC ohne Basis kippt früh."]
            elif total >= 7.5 and (not timing_score or timing_score >= 6): action="Einstieg möglich (klein, Plan)."; why=["HC-typisch, Timing passt."]
            elif total >= 6.5: action="Watchlist – Entry bei Bestätigung."; why=["Spannend, aber noch nicht reif."]
            else: action="Beobachten oder streichen."; why=["Asymmetrie überzeugt nicht genug."]
    notes = []
    if has_position:
        if perf is not None and status: notes.append(f"Status: {status}.")
        w = compute_position_weight(portfolio_total, position_value)
        if w:
            notes.append(f"Positionsgewicht: ca. {w*100:.1f}%.")
            if w >= 0.10: notes.append("≥10% ist groß – Nachkauf nur bei sehr guten Gründen.")
            elif w >= 0.06: notes.append("6–10% ist mittel – Nachkauf selektiv/gestaffelt.")
    expl = []
    if fund_score: expl.append(f"Fundament {fund_score:.1f}/10 → {bucket(fund_score)}.")
    if timing_score: expl.append(f"Timing {timing_score:.1f}/10 → {bucket(timing_score)}.")
    if story_score: expl.append(f"Business / Story {story_score:.1f}/10 → {bucket(story_score)}.")
    expl.append(f"Gesamt {total:.1f}/10 → {bucket(total)}.")
    out = [f"**Einordnung:** {label}"]
    if story_info:
        bm = story_info.get("business_model"); cs = story_info.get("characteristics",[])
        out.append(""); 
        if bm: out.append(f"**Geschäftsprofil:** {bm}")
        if cs: out.append(f"**Charakter:** {', '.join(cs)}")
    out += ["","**Scores:**"] + [f"- {l}" for l in expl]
    out += ["","**Handlung:**", f"- **{action}**"] + [f"- {w}" for w in why]
    if gates: out += ["","**Schutzregeln:**"] + [f"- {g}" for g in gates]
    if notes: out += ["","**Portfolio-Kontext:**"] + [f"- {n}" for n in notes]
    return "\n".join(out), action, why

# ══════════════════════════════════════════════════════════════════════════════
# OpenAI / Ace
# ══════════════════════════════════════════════════════════════════════════════
ACE_PROMPT = """Du bist Ace, ein ruhiger, erfahrener Investment-Sparringspartner.
Schreibe ein ausführliches Fazit wie für einen guten Freund.
Du bekommst: Asset, Einordnung, Fundament-/Chart-/Business-Daten, Empfehlung, Trigger, Risiken.
Aufgabe: Kontext (was ist das?), Gesamtbild interpretieren, klare Empfehlung im Einklang mit der Handlung.
Stil: 1 Text 120–220 Wörter, ruhig, klar, keine Bulletpoints, keine Emojis, keine Kursziele."""

ACE_PROMPT_BEGINNER = """Du bist Ace, ein geduldiger, freundlicher Investment-Erklärer.
Dein Gesprächspartner ist ein Einsteiger — erkläre alles in einfacher, klarer Sprache ohne Fachjargon.
Vermeide Begriffe wie MA20, RSI, MACD, KGV, Beta, Volatilität. Nutze stattdessen:
"Durchschnittspreis", "Kaufinteresse", "Kurstrend", "Bewertung", "Schwankungen".
Du bekommst: Asset, Einordnung, Fundament-/Chart-/Business-Daten, Empfehlung.
Aufgabe: Erkläre in 3 kurzen Absätzen: (1) Was macht das Unternehmen? (2) Wie sieht die Aktie gerade aus — gut oder schlecht? (3) Was würde Ace einem Freund empfehlen?
Stil: 100–160 Wörter, freundlich, direkt wie eine Sprachnachricht, keine Bulletpoints, keine Emojis."""

@st.cache_data(ttl=86400, show_spinner=False)
def get_company_brief_de(ticker: str, name: str, sector: str,
                         industry: str, api_key: str = "") -> str:
    """Einen deutschen Satz was das Unternehmen macht — KI-generiert, 24h gecacht."""
    if not name or name == ticker:
        return ""
    if api_key and OPENAI_AVAILABLE:
        try:
            client = OpenAI(api_key=api_key)
            resp = client.responses.create(
                model="gpt-4.1-mini",
                input=[{"role": "user", "content":
                    f"Beschreibe in EINEM deutschen Satz (max. 130 Zeichen) präzise was "
                    f"{name} ({ticker}) macht. Sektor: {sector or '?'}, Branche: {industry or '?'}. "
                    f"Keine Marketing-Floskeln, nur Fakten. Kein Punkt am Ende nötig."}]
            )
            result = (getattr(resp, "output_text", "") or "").strip().strip('"').strip("'")
            return result[:150] if result else ""
        except Exception:
            pass
    # Fallback ohne API
    if sector and industry:
        return f"{name} ist im Bereich {sector} ({industry}) tätig."
    elif sector:
        return f"{name} ist im Sektor {sector} tätig."
    return ""


def ace_fazit(snapshot, model="gpt-4.1-mini", user_level="pro"):
    if not OPENAI_AVAILABLE: raise RuntimeError("openai SDK nicht installiert.")
    if "OPENAI_API_KEY" not in st.secrets: raise RuntimeError("OPENAI_API_KEY fehlt in Secrets.")
    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    _sys_prompt = ACE_PROMPT_BEGINNER if user_level == "beginner" else ACE_PROMPT
    resp = client.responses.create(model=model, input=[
        {"role": "system", "content": _sys_prompt},
        {"role": "user", "content": f"Schreibe das Fazit:\n{snapshot}"},
    ])
    return getattr(resp, "output_text", "").strip()


# ══════════════════════════════════════════════════════════════════════════════
# Depot-Fit  — passt diese Aktie ins bestehende Portfolio?
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# Velox KI-Radar — OpenAI-powered Stock Discovery mit Qualitätsfilter
# ══════════════════════════════════════════════════════════════════════════════
ACE_DEPOT_PROMPT = """You are "Ace", a strategic investment analyst.

IMPORTANT:
- Write the entire analysis in German.
- Do NOT use emojis.
- Do NOT use decorative symbols.
- Use a clear, professional tone.

You receive a full investment portfolio with positions, values, and performance data.

Your task is to analyze the portfolio as a whole, identify structural patterns, and provide clear, actionable insights. Do not assume any predefined strategy. Instead, infer the structure yourself.

---

ANALYSIS GOAL

Evaluate whether the portfolio:
- follows a clear strategy
- is structurally sound
- or appears inconsistent and driven by isolated decisions

---

THINKING RULES

- Think in patterns, not individual stocks
- Interpret data — do NOT just describe it
- Prioritize the 1–2 most important issues
- Avoid generic statements unless clearly justified
- Avoid hedging language like "might", "could", "possibly"
- Take a clear stance

---

ANALYSIS LOGIC

1. Identify structure
→ What roles exist in the portfolio? (e.g. stable, growth, speculative)

2. Evaluate balance
→ Relationship between stability, growth, risk

3. Detect concentration
→ Sector exposure, dependency on few winners, hidden overlap

4. Assess strategy
→ Does the portfolio feel intentional or random?

---

OUTPUT FORMAT (STRICT)

### Gesamtbild
- 2–3 Sätze
- 1 klarer Satz: „Das ist ein … Depot"

---

### Struktur und Rollen
- Welche Arten von Investments dominieren?
- Was ist unterrepräsentiert?

---

### Zentrale Erkenntnisse
(max. 4 Stichpunkte)
- Nur echte Insights, keine offensichtlichen Aussagen

---

### Kritische Einordnung
- Größtes strukturelles Risiko
- Strategische Inkonsistenz
- Schwächen in der Allokation

Sei direkt.

---

### Handlungsoptionen
Gib klare Einordnungen zur Überlegung:
- Halten
- Nachkaufen prüfen
- Reduzieren prüfen
- Umschichten prüfen

Keine Anlageberatung — nur strukturelle Überlegungen zur eigenen Entscheidungsfindung.

---

### Reflexionsfrage
Formuliere genau EINE starke Frage, die die Strategie hinterfragt

---

TONE

- Klar, ruhig, direkt
- Keine Emojis
- Keine Floskeln
- Keine motivierenden oder werblichen Formulierungen

---

IMPORTANT

- Do not repeat raw input data
- Do not explain basic concepts
- Focus on interpretation and decision-making

---

PORTFOLIO DATA:
{{portfolio_json}}"""


ACE_CORE_PROMPT = """You are "Ace", a strategic investment analyst focused on long-term wealth building ("Quiet Compounder").

IMPORTANT:
- Write the entire analysis in German.
- Do NOT use emojis.
- Do NOT use decorative symbols.
- Use a clear, professional tone.

You receive a portfolio intended as the user's core investments (ETFs and long-term quality stocks).
Evaluate whether this portfolio is structurally sound for long-term, stable wealth accumulation.

ANALYSIS GOAL: Evaluate whether the portfolio has a strong foundation, is built for consistency, or contains weaknesses undermining long-term compounding.

THINKING RULES: Think in structures not stocks. Focus on stability. Take a clear stance. No hedging language.

ANALYSIS LOGIC:
1. Foundation: stable base (ETFs) or dependent on individual stocks?
2. Balance: broad market vs individual company risk
3. Concentration: sector/regional clustering
4. Role clarity: clear purpose or unnecessarily complex?
5. Strategy consistency: matches long-term compounder approach?

OUTPUT FORMAT (STRICT):

### Gesamtbild
- 2-3 Saetze
- 1 klarer Satz: Das ist ein ... Core-Depot

### Struktur-Bewertung
- Qualitaet des Fundaments
- Rolle der Einzelaktien
- Balance zwischen Stabilitaet und Konzentration

### Zentrale Erkenntnisse
(max. 4 Stichpunkte - nur strukturelle Muster)

### Kritische Einordnung
- Groesstes strukturelles Risiko
- Strategische Inkonsistenz
- Unnoetige Komplexitaet oder Schwaechen

### Handlungsoptionen
Zur eigenen Entscheidungsfindung (keine Anlageberatung):
- Halten
- Nachkaufen pruefen (nur wenn strukturell sinnvoll)
- Reduzieren pruefen
- Vereinfachen pruefen

### Reflexionsfrage
Genau EINE starke Frage zur langfristigen Strategie

TONE: Ruhig, rational, strukturiert. Kein Hype. Fokus auf Vermoegensaufbau.

IMPORTANT: Do not repeat raw data. Focus on structure and long-term viability.

PORTFOLIO DATA:
{{portfolio_json}}"""


ACE_NEXT_STEPS_PROMPT = """Based on the following portfolio analysis, generate exactly 3-5 concrete next steps.

Return ONLY a valid JSON array. No other text. No markdown. No explanation.

Each item must have:
- "aktion": one of: "Halten", "Reduzieren pruefen", "Streichen pruefen", "Aufstocken pruefen", "Umschichten pruefen", "Radar oeffnen", "Nichts tun"
- "ticker": specific ticker symbol if applicable, else null
- "titel": short title max 50 chars (German)
- "grund": brief reason max 90 chars (German, direct)
- "radar_query": search term for investment radar if applicable (German, 2-4 words), else null

Rules:
- At least one positive step ("Halten" or "Nichts tun" if portfolio is strong)
- Be specific, not generic
- If thesis is unclear for a position: "Streichen pruefen"
- If sector is missing: "Radar oeffnen" with relevant radar_query

ANALYSIS:
{analysis_text}"""


def generate_next_steps(analysis_text: str, api_key: str) -> list:
    """Generiert 3-5 strukturierte Next Steps aus einem Analyse-Text via GPT."""
    if not analysis_text or not api_key or not OPENAI_AVAILABLE:
        return []
    try:
        import json as _js
        client = OpenAI(api_key=api_key)
        prompt = ACE_NEXT_STEPS_PROMPT.replace("{analysis_text}", analysis_text[:3000])
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[{"role": "user", "content": prompt}]
        )
        raw = (getattr(resp, "output_text", "") or "").strip()
        # JSON extrahieren
        import re as _re4
        m = _re4.search(r'\[.*\]', raw, _re4.DOTALL)
        if not m:
            return []
        return _js.loads(m.group(0))
    except Exception:
        return []


def render_next_steps(steps: list, is_hc: bool = False,
                      api_key: str = "", pname: str = ""):
    """Rendert Next Steps Cards mit optionalem Radar-Button."""
    if not steps:
        return
    _aktion_color = {
        "Halten":             "#10b981",
        "Nichts tun":         "#10b981",
        "Aufstocken pruefen": "#3b82f6",
        "Reduzieren pruefen": "#f59e0b",
        "Umschichten pruefen":"#f59e0b",
        "Streichen pruefen":  "#ef4444",
        "Radar oeffnen":      "#8b5cf6",
    }
    st.markdown(
        '<div style="font-size:0.62rem;font-weight:700;letter-spacing:0.16em;'
        'text-transform:uppercase;color:var(--text-color);opacity:0.4;'
        'margin:1rem 0 0.5rem 0;">Konkrete Next Steps</div>',
        unsafe_allow_html=True)

    for _i, step in enumerate(steps):
        _ak    = step.get("aktion", "")
        _tk    = step.get("ticker")
        _ti    = step.get("titel", "")
        _gr    = step.get("grund", "")
        _rq    = step.get("radar_query")
        _col   = _aktion_color.get(_ak, "#888")
        _bg    = f"rgba({','.join(str(int(int(_col[i:i+2],16))) for i in (1,3,5) if _col.startswith('#'))},0.07)" \
                 if _col.startswith('#') else "rgba(128,128,128,0.05)"

        st.markdown(
            f'<div style="background:var(--secondary-background-color);'
            f'border:1px solid rgba(128,128,128,0.12);border-left:3px solid {_col};'
            f'border-radius:10px;padding:0.65rem 0.9rem;margin-bottom:0.4rem;">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
            f'<span style="font-size:0.62rem;font-weight:700;letter-spacing:0.1em;'
            f'text-transform:uppercase;color:{_col};">{_ak}</span>'
            + (f'<span style="font-size:0.72rem;font-weight:700;'
               f'color:var(--text-color);opacity:0.6;">{_tk}</span>' if _tk else '')
            + f'</div>'
            f'<div style="font-size:0.82rem;font-weight:600;color:var(--text-color);'
            f'margin:0.15rem 0 0.1rem 0;">{_ti}</div>'
            f'<div style="font-size:0.72rem;color:var(--text-color);opacity:0.5;">{_gr}</div>'
            f'</div>',
            unsafe_allow_html=True)

        if _rq:
            if st.button(f"◎ Im Radar suchen: {_rq}", key=f"ns_radar_{pname}_{_i}",
                         use_container_width=False):
                st.session_state["radar_mode_sel"]        = "KI"
                st.session_state["ki_radar_query"]        = _rq
                st.session_state["ki_radar_prefill"]      = _rq   # Textfeld vorausfüllen
                st.session_state["ki_radar_result"]       = None
                st.session_state.pop("ki_radar_input", None)       # Widget zurücksetzen
                st.session_state["_auto_switch_to_radar"] = True
                st.rerun()

    # HC: Radar-Teaser
    if is_hc:
        st.markdown(
            '<div style="background:linear-gradient(135deg,'
            'rgba(245,158,11,0.08),rgba(245,158,11,0.03));'
            'border:1px solid rgba(245,158,11,0.22);border-radius:12px;'
            'padding:0.75rem 1rem;margin-top:0.6rem;">'
            '<div style="font-size:0.6rem;font-weight:800;letter-spacing:0.18em;'
            'text-transform:uppercase;color:#f59e0b;margin-bottom:0.25rem;">'
            '◎ Velox Radar</div>'
            '<div style="font-size:0.78rem;color:var(--text-color);opacity:0.6;">'
            'Neue Hidden Champions entdecken — KI-Radar findet Ideen passend '
            'zu deiner Strategie.</div>'
            '</div>',
            unsafe_allow_html=True)
        if st.button("◎ Hidden Champions im Radar entdecken",
                     key=f"hc_radar_hint_{pname}", use_container_width=True):
            _hc_q = "Hidden Champions Nischenwerte"
            st.session_state["radar_mode_sel"]        = "KI"
            st.session_state["ki_radar_query"]        = _hc_q
            st.session_state["ki_radar_prefill"]      = _hc_q
            st.session_state["ki_radar_result"]       = None
            st.session_state.pop("ki_radar_input", None)
            st.session_state["_auto_switch_to_radar"] = True
            st.rerun()


ACE_HC_PROMPT = """You are "Ace", a strategic investment analyst specialized in high-risk, high-upside portfolios ("Hidden Champions").

IMPORTANT:
- Write the entire analysis in German.
- Do NOT use emojis.
- Do NOT use decorative symbols.
- Use a clear, direct, and critical tone.

You receive a portfolio of speculative individual stocks aiming for asymmetric opportunities (5x-10x potential).
Evaluate the quality of these ideas, not just their performance.

ANALYSIS GOAL: Evaluate whether the portfolio contains real high-upside opportunities with clear theses, or is a collection of weak, unstructured bets.

THINKING RULES:
- Focus on idea quality, not short-term performance
- Losses are acceptable only with a clear thesis
- Each position should have: a clear story, a plausible trigger (3-9 months), re-rating potential
- Think critically: Is this a hidden champion? Or just a fallen or popular stock?
- Prioritize the 1-2 most important problems
- Take a clear stance. No hedging.

ANALYSIS LOGIC:
1. Evaluate idea quality: Do positions represent strong, differentiated ideas?
2. Detect weak positions: Which investments lack a clear thesis?
3. Assess concentration: Capital focused on strong ideas or spread too thin?
4. Identify inefficiencies: Where is capital likely wasted?
5. Evaluate portfolio discipline: Intentional or random?

OUTPUT FORMAT (STRICT):

### Gesamtbild
- 2-3 Saetze
- 1 klarer Satz: Das ist ein ... High-Risk-Depot

### Portfolio-Qualitaet
- Anteil ueberzeugender Ideen vs. schwacher Wetten
- Einschaetzung der Gesamtqualitaet

### Zentrale Erkenntnisse
(max. 4 Stichpunkte - Fokus auf Qualitaet der Ideen, nicht Performance)

### Kritische Einordnung
- Welche Positionen wirken wie Fehler?
- Wo fehlt eine klare Story?
- Wo ist Kapital ineffizient gebunden?
Sei direkt und kompromisslos.

### Handlungsoptionen
Zur eigenen Entscheidungsfindung (keine Anlageberatung):
- Halten (nur bei klarer These)
- Aufstocken pruefen (bei hoher Ueberzeugung)
- Streichen pruefen (wenn keine klare Story)
- Beobachten
Optional: Kapital auf wenige starke Ideen buendeln

### Reflexionsfrage
Genau EINE harte Frage zur Qualitaet der eigenen Entscheidungen

TONE: Direkt, kritisch, analytisch. Kein Beschoenigen. Fokus auf Qualitaet der Entscheidungen.

IMPORTANT: Do not repeat raw data. Do not justify weak positions. Focus on filtering and prioritization.

PORTFOLIO DATA:
{{portfolio_json}}"""


AI_RADAR_SYSTEM_PROMPT = """Du bist der Velox KI-Investment-Analyst — ein erfahrener Portfoliomanager mit dem Blick eines Research-Analysten.

VELOX INVESTMENT-PHILOSOPHIE:
Du denkst in zwei Kategorien: "Core Assets" (Qualitäts-Compounder die über Jahrzehnte wachsen) und "Hidden Champions" (Nischenmarktführer die kaum jemand kennt aber die ihre Märkte dominieren). Beide Typen haben: klaren Wettbewerbsvorteil, solide Bilanz, und eine verständliche These.

QUALITÄTSFILTER (flexibel, nicht dogmatisch):
- Klarer Wettbewerbsvorteil: Moat, Nische, Pricing Power, Switching Costs, Netzwerkeffekte
- Solide Bilanz: Debt/EBITDA < 5x bevorzugt (Ausnahmen: Versorger, REITs, Wachstum mit klarem Pfad)
- Profitabel oder klarer Weg zur Profitabilität (kein Kapitalvernichter)
- Kein Penny Stock, keine Shell Companies, keine reinen Meme-Wetten

WARUM JETZT? — Das ist entscheidend:
Für jede Empfehlung überlege: Was macht sie AKTUELL interessant? Gibt es einen Katalysator, eine Neubewertung, einen Trend, einen Rücksetzer als Einstiegschance?

KREATIVITÄT — Pflicht:
- Mindestens 1-2 Picks die überraschen — Hidden Champions, vergessene Qualitätswerte, thematische ETFs die niemand kennt
- Nicht immer die offensichtlichen Namen — wenn jemand "KI" sucht, nicht nur NVIDIA vorschlagen
- Denke auch an Profiteure zweiter Ordnung: Wer liefert die Schaufeln wenn andere Gold suchen?

Investment-Typen:
- "Core Asset": Stabile Qualitäts-Compounder, berechenbar, langfristig
- "Hidden Champion": Nischenmarktführer, oft B2B, kaum bekannt aber dominant
- "ETF": Themen-/Sektor-ETFs mit >300 Mio USD AUM, klar positioniert
- "Bond ETF": Anleihen-ETFs für Stabilität/Income

Für jeden Vorschlag gibst du zurück:
- ticker: Yahoo Finance Ticker (EXAKT — z.B. MSFT, ASML.AS, DB1.DE, SXR8.DE)
- name: Offizieller Name
- type: "Core Asset" | "Hidden Champion" | "ETF" | "Bond ETF"
- thesis: Investment-These auf Deutsch, 2-3 präzise Sätze — WAS macht sie interessant, WARUM jetzt, WAS könnte schiefgehen
- confidence: "hoch" (überzeugende Qualität) | "mittel" (gut mit Vorbehalt) | "explorativ" (asymmetrisches Chancen/Risiko)

KRITISCH:
- Ticker müssen auf Yahoo Finance verfügbar sein — bei Unsicherheit weglassen
- Mische Bekanntes mit Überraschendem
- Sei ehrlich: lieber "explorativ" als inflationäres "hoch"
- Die Thesis muss analytisch sein, nicht beschreibend: nicht "XY macht Software" sondern "XY hat 90% Marktanteil in X und wird von Y-Trend getrieben"

Antworte NUR mit einem validen JSON Array. Keine Erklärungen außerhalb des JSON."""

@st.cache_data(ttl=86400, show_spinner=False)  # 24h Cache; bust=N erzwingt neuen Call
def ai_radar_discovery(query: str, n: int = 6, model: str = "gpt-4.1-mini",
                       bust: int = 0) -> list:
    """OpenAI-powered Stock Discovery mit Velox-Qualitätsfilter.
    Gibt validierte (ticker, name, type, thesis, confidence) Tuples zurück."""
    if not OPENAI_AVAILABLE:
        return []
    api_key = ""
    try:
        import streamlit as _st
        api_key = (_st.secrets.get("OPENAI_API_KEY") or
                   __import__("os").environ.get("OPENAI_API_KEY") or "")
    except Exception:
        pass
    if not api_key:
        return []
    try:
        client = OpenAI(api_key=api_key)
        _req_n   = min(n + 4, 16)  # mehr anfragen als nötig (Validierung filtert einige)
        user_msg = (f"Schlage {_req_n} qualitativ hochwertige Investments vor zum Thema: "
                    f"'{query}'\n"
                    f"Mische Aktien, ETFs und ggf. Bond ETFs. "
                    f"Fokus auf echtes Potenzial, nicht nur bekannte Namen. "
                    f"Schlage auch 1-2 weniger bekannte Werte vor die überraschen.")
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": AI_RADAR_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
        )
        raw = (getattr(resp, "output_text", "") or "").strip()
        # JSON aus Antwort extrahieren
        import re as _re, json as _json
        m = _re.search(r'\[.*\]', raw, _re.DOTALL)
        if not m:
            return []
        items = _json.loads(m.group(0))

        # Portfolio-Ticker laden → aus KI-Ergebnissen filtern
        try:
            import streamlit as _st2
            _port = load_portfolio()
            _portfolio_tickers = {
                _normalize_ticker((p.get("ticker") or "").upper())
                for pn in PORTFOLIO_NAMES
                for p in _port.get(pn, {}).get("positions", [])
                if p.get("ticker")
            }
        except Exception:
            _portfolio_tickers = set()

        result = []
        for item in items:
            tk   = (item.get("ticker") or "").strip().upper()
            nm   = (item.get("name")   or "").strip()
            tp   = (item.get("type")   or "Core Asset").strip()
            th   = (item.get("thesis") or "").strip()
            conf = (item.get("confidence") or "mittel").strip()
            if not tk or not nm:
                continue
            # Bereits im Portfolio → überspringen
            if _normalize_ticker(tk) in _portfolio_tickers:
                continue
            # Ticker validieren — kurzer Check ob Yahoo das kennt
            try:
                _info = yf.Ticker(tk).fast_info
                _price = getattr(_info, "last_price", None)
                if not _price:
                    continue  # Ticker existiert nicht
            except Exception:
                continue
            result.append((tk, nm, tp, th, conf))
            if len(result) >= n:
                break
        return result
    except Exception:
        return []


def calculate_depot_fit_score(ticker: str, mode: str, profile: dict) -> tuple:
    """Errechnet einen Depot-Fit-Score (1–10) basierend auf dem bestehenden Portfolio.
    Gibt (score, reason) zurück. Score = None wenn kein Portfolio vorhanden."""
    try:
        port = load_portfolio()
        all_pos = [p for pn in PORTFOLIO_NAMES
                   for p in port.get(pn, {}).get("positions", [])]
        if not all_pos:
            return None, ""

        total = len(all_pos)
        current_sector = (profile.get("sector") or "").strip()

        # Sektor-Überschneidung
        sector_counts = {}
        for p in all_pos:
            _ps = fetch_sector_cached(p.get("ticker","")) if p.get("ticker") else ""
            if _ps:
                sector_counts[_ps] = sector_counts.get(_ps, 0) + 1

        same_sector = sector_counts.get(current_sector, 0)
        sector_ratio = same_sector / total if total > 0 else 0

        # Mode-Balance
        ca_pnames = [pn for pn in PORTFOLIO_NAMES if "Compounder" in pn or "Core" in pn]
        hc_pnames = [pn for pn in PORTFOLIO_NAMES if "Champion" in pn or "HC" in pn]
        ca_count = sum(len(port.get(pn, {}).get("positions", [])) for pn in ca_pnames)
        hc_count = sum(len(port.get(pn, {}).get("positions", [])) for pn in hc_pnames)

        score = 6.0

        # Neue Sektor-Diversifikation = Bonus
        if current_sector and same_sector == 0:
            score += 2.5
            reason = f"Neue Branche im Depot ({current_sector})"
        elif sector_ratio > 0.25:
            score -= 2.0
            reason = f"{current_sector} bereits {sector_ratio*100:.0f}% des Depots"
        elif sector_ratio > 0.12:
            score -= 0.8
            reason = f"{current_sector} leicht übergewichtet"
        else:
            score += 0.5
            reason = "Sinnvolle Ergänzung zur Sektorverteilung"

        # Mode-Balance
        if mode == "Core Asset" and ca_count > hc_count * 2:
            score -= 1.0
        elif mode == "Hidden Champion" and hc_count < ca_count * 0.4:
            score += 1.5
        elif mode == "Hidden Champion" and hc_count > ca_count:
            score -= 0.5

        # Qualitäts-Bonus
        fs = st.session_state.get("fund_score")
        if fs and fs >= 7.5:
            score += 0.8
        elif fs and fs >= 6.0:
            score += 0.3

        score = round(max(1.0, min(10.0, score)), 1)
        return score, reason
    except Exception:
        return None, ""


def build_depot_fit(ticker, mode, profile, total_score, story_info, level="pro"):
    """Gibt eine Liste von (icon, text, color) Tuples zurück — Depot-Fit Einschätzung."""
    port    = load_portfolio()
    sector  = (profile.get("sector") or "").strip()
    bm      = (story_info.get("business_model") or "") if story_info else ""
    is_beginner = level == "beginner"
    lines   = []  # (icon, text, accent_color)

    # ── Alle Positionen sammeln ──────────────────────────────────────────────
    all_pos  = []
    ca_tickers, hc_tickers = set(), set()
    for pname, pdata in port.items():
        for pos in pdata.get("positions", []):
            all_pos.append(pos)
            tk_norm = _normalize_ticker((pos.get("ticker") or "").upper())
            if pname == "Quiet Compounder":
                ca_tickers.add(tk_norm)
            else:
                hc_tickers.add(tk_norm)

    n_total = len(all_pos)
    n_ca    = len(ca_tickers)
    n_hc    = len(hc_tickers)
    tk_norm = _normalize_ticker(ticker.upper())
    already_in_ca = tk_norm in ca_tickers
    already_in_hc = tk_norm in hc_tickers

    # ── Case A: Kein Portfolio vorhanden ────────────────────────────────────
    if n_total == 0:
        if mode == "Core Asset":
            if is_beginner:
                lines.append(("◆", "Diese Aktie ist als stabiles Langzeitinvestment eingestuft — "
                              "ein solider Einstiegsbaustein für ein erstes Depot.", "#3b82f6"))
            else:
                lines.append(("◆", f"Core Asset mit Velox-Score {total_score:.1f}/10 — "
                              "ideal als Anker für ein neu aufzubauendes Depot.", "#3b82f6"))
        else:
            if is_beginner:
                lines.append(("▸", "Diese Aktie ist ein versteckter Marktführer in seiner Nische — "
                              "als Beimischung in einem Depot oft eine starke Rendite-Quelle.", "#8b5cf6"))
            else:
                lines.append(("▸", f"Hidden Champion ({bm or 'Nischenmarktführer'}) — "
                              "als Satelliten-Position in einem Depot mit Core-Anker sehr geeignet.", "#8b5cf6"))
        lines.append(("→", "Du hast noch kein Depot bei Velox angelegt. Im Portfolio-Tab kannst du "
                      "deine Positionen hinterlegen — dann bekommst du eine individuelle Einschätzung "
                      "wie diese Aktie in dein persönliches Depot passt.", "#64748b"))
        return lines

    # ── Case B: Aktie bereits im Depot ──────────────────────────────────────
    if already_in_ca or already_in_hc:
        pname_found = "Quiet Compounder" if already_in_ca else "Hidden Champions"
        if is_beginner:
            lines.append(("✓", f"Diese Aktie hältst du bereits im Depot '{pname_found}'. "
                          "Ein Nachkauf ist möglich — prüfe ob der aktuelle Kurs attraktiv ist.", "#00C864"))
        else:
            lines.append(("✓", f"Bereits im Portfolio '{pname_found}'. Nachkauf-Entscheidung "
                          "basierend auf Timing und aktuellem Gewicht prüfen.", "#00C864"))
        return lines

    # ── Case C: Portfolio vorhanden, neue Position ───────────────────────────
    # Balance Core / HC
    balance_ok = (n_ca > 0 and n_hc > 0)
    ca_ratio   = n_ca / max(n_total, 1)

    if mode == "Core Asset":
        if n_ca == 0:
            msg = ("Das wäre dein erster Core Asset — ein stabiles Fundament fürs Depot."
                   if is_beginner else
                   f"Du hast noch keinen Core Asset. Diese Position schafft das stabile Fundament.")
            lines.append(("◆", msg, "#3b82f6"))
        elif ca_ratio > 0.75:
            msg = ("Dein Depot besteht schon zu einem großen Teil aus ähnlichen stabilen Aktien. "
                   "Prüfe ob eine weitere die Diversifikation wirklich verbessert."
                   if is_beginner else
                   f"{n_ca} von {n_total} Positionen sind bereits Core Assets ({ca_ratio:.0%}). "
                   "Weitere Konzentration prüfen — Diversifikation könnte sinnvoller sein.")
            lines.append(("⚖", msg, "#f59e0b"))
        else:
            msg = (f"Du hast bereits {n_ca} stabile Aktie{'n' if n_ca>1 else ''} im Depot. "
                   f"Diese würde als {n_ca+1}. Core Asset die Stabilität weiter stärken."
                   if is_beginner else
                   f"Core-Ratio aktuell {ca_ratio:.0%} ({n_ca}/{n_total}). "
                   f"Diese Position stärkt den stabilen Kern auf {(n_ca+1)/(n_total+1):.0%}.")
            lines.append(("✓", msg, "#00C864"))
    else:  # Hidden Champion
        if n_hc == 0:
            msg = ("Du hast noch keinen Hidden Champion im Depot — "
                   "diese Aktie könnte als Rendite-Treiber interessant sein."
                   if is_beginner else
                   f"Kein HC im Portfolio. Diese Position bringt als erste HC-Beimischung Wachstumspotenzial.")
            lines.append(("▸", msg, "#8b5cf6"))
        elif n_hc >= 4:
            msg = ("Du hast schon mehrere solcher Wachstumspositionen im Depot. "
                   "Prüfe ob eine weitere sinnvoll ist oder das Risiko zu hoch wird."
                   if is_beginner else
                   f"Mit {n_hc} HC-Positionen bereits gut aufgestellt. "
                   "Weitere HC-Konzentration nur wenn Überzeugung sehr hoch.")
            lines.append(("⚖", msg, "#f59e0b"))
        else:
            msg = (f"Du hast {n_hc} ähnliche Wachstumsaktie{'n' if n_hc>1 else ''} im Depot. "
                   f"Diese würde als {n_hc+1}. Hidden Champion das Renditepotenzial weiter erhöhen."
                   if is_beginner else
                   f"HC-Bestand: {n_hc}/{n_total} Positionen. "
                   f"Diese ergänzt das HC-Profil mit '{bm or sector}'-Exposition.")
            lines.append(("✓", msg, "#00C864"))

    # ── Echter Sektor-Vergleich mit yfinance Cache ───────────────────────────
    if sector and n_total >= 2:
        # Sektoren der Portfolio-Positionen live nachschlagen (gecacht)
        _port_sectors = []
        _port_sector_tickers = []
        for _p in all_pos:
            _ptk = (_p.get("ticker") or "").strip()
            if _ptk:
                _psec = fetch_sector_cached(_ptk)
                if _psec:
                    _port_sectors.append(_psec)
                    if _psec.lower() == sector.lower():
                        _port_sector_tickers.append(_ptk.upper())

        _same_count = len(_port_sector_tickers)
        if _same_count >= 2:
            _tk_list = ", ".join(_port_sector_tickers[:3])
            sec_txt = (f"Du hast bereits {_same_count} {sector}-Aktien im Depot ({_tk_list}). "
                       "Eine weitere erhöht die Sektor-Konzentration."
                       if is_beginner else
                       f"Sektor-Klumpen: {_same_count}× '{sector}' im Portfolio ({_tk_list}). "
                       "Diversifikation in anderen Sektoren prüfen.")
            lines.append(("·", sec_txt, "#f59e0b"))
        elif _same_count == 0 and len(_port_sectors) >= 2:
            sec_txt = (f"Kein anderes '{sector}'-Investment im Depot — "
                       "diese Aktie schließt eine Sektor-Lücke."
                       if is_beginner else
                       f"Sektor '{sector}' noch nicht im Portfolio repräsentiert → echte Diversifikation.")
            lines.append(("✓", sec_txt, "#00C864"))

    return lines


# ══════════════════════════════════════════════════════════════════════════════
# CSS — Bloomberg-Style
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<style>
/* ═══════════════════════════════════════════════════════════════
   VELOX — Design System v9.0
   Alle Farben via Streamlit CSS-Variablen: theme-agnostisch.
   Typografie-Skala:
     xs  : 0.72rem  (labels, badges)
     sm  : 0.84rem  (hints, meta)
     base: 0.95rem  (body)
     lg  : 1.1rem   (emphasis)
     xl  : 2.8rem   (score numbers)
═══════════════════════════════════════════════════════════════ */

/* ── Basis ── */
.main .block-container { font-size: 0.95rem; }
.main .block-container p,
.main .block-container li { font-size: 0.95rem !important; line-height: 1.6; }

/* ── Cards ── */
.ace-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.16);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.65rem;
}

/* ── Score Cards ── */
.vx-score-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.16);
    border-radius: 10px;
    padding: 1rem 1.2rem 0.9rem 1.2rem;
    margin-bottom: 0.65rem;
    position: relative;
    overflow: hidden;
}
.vx-score-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    border-radius: 10px 0 0 10px;
}
.vx-score-card.vx-green::before { background: #00C864; }
.vx-score-card.vx-orange::before { background: #FFA500; }
.vx-score-card.vx-red::before { background: #FF4444; }

.vx-score-label {
    font-size: 0.72rem;
    font-family: 'Space Grotesk', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-color);
    opacity: 0.45;
    margin-bottom: 0.3rem;
}
.vx-score-number {
    font-size: 2.8rem;
    font-weight: 700;
    font-family: 'Space Grotesk', monospace;
    line-height: 1;
    letter-spacing: -0.02em;
}
.vx-score-denom {
    font-size: 0.9rem;
    font-weight: 400;
    color: var(--text-color);
    opacity: 0.3;
    margin-left: 2px;
}
.vx-progress-track {
    height: 3px;
    background: rgba(128,128,128,0.15);
    border-radius: 2px;
    margin: 0.55rem 0 0.6rem 0;
    overflow: hidden;
}
.vx-progress-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.4s ease;
}
.vx-hint {
    font-size: 0.84rem;
    line-height: 1.55;
    color: var(--text-color);
    opacity: 0.65;
}

/* ── Action Banner ── */
.ace-go   { background:rgba(0,200,100,0.08);  border-left:3px solid #00C864; border-radius:8px; padding:0.9rem 1.3rem; margin-bottom:0.9rem; }
.ace-wait { background:rgba(255,165,0,0.08);  border-left:3px solid #FFA500; border-radius:8px; padding:0.9rem 1.3rem; margin-bottom:0.9rem; }
.ace-stop { background:rgba(255,60,60,0.08);  border-left:3px solid #FF4444; border-radius:8px; padding:0.9rem 1.3rem; margin-bottom:0.9rem; }

/* ── Trigger chips ── */
.trig-go   { background:rgba(0,200,100,0.07); border:1px solid rgba(0,200,100,0.22); border-radius:7px; padding:0.65rem 1rem; margin-bottom:0.45rem; font-size:0.92rem; line-height:1.55; }
.trig-wait { background:rgba(255,165,0,0.07); border:1px solid rgba(255,165,0,0.22);  border-radius:7px; padding:0.65rem 1rem; margin-bottom:0.45rem; font-size:0.92rem; line-height:1.55; }
.trig-stop { background:rgba(255,60,60,0.07); border:1px solid rgba(255,60,60,0.22);  border-radius:7px; padding:0.65rem 1rem; margin-bottom:0.45rem; font-size:0.92rem; line-height:1.55; }

/* ── Placeholder ── */
.ace-placeholder {
    background: var(--secondary-background-color);
    border: 1px dashed rgba(128,128,128,0.2);
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
    font-size: 0.88rem;
    color: var(--text-color);
    opacity: 0.32;
    margin-bottom: 0.65rem;
}

/* ── Section label ── */
.ace-section {
    font-size: 0.72rem;
    font-family: 'Space Grotesk', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.13em;
    margin-bottom: 0.55rem;
    margin-top: 0.15rem;
    color: var(--text-color);
    opacity: 0.42;
}

/* ── Level Radio als Segmented Pill ── */
div[data-testid="stRadio"] { margin-top: 0 !important; }
div[data-testid="stRadio"] > label { display: none !important; }
div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 30px;
    padding: 3px;
    gap: 2px;
    width: fit-content;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label {
    display: flex !important;
    align-items: center;
    padding: 6px 18px;
    border-radius: 26px;
    font-size: 0.72rem !important;
    font-weight: 600;
    letter-spacing: 0.06em;
    cursor: pointer;
    transition: all 0.2s ease;
    color: var(--text-color);
    opacity: 0.42;
    white-space: nowrap;
}
/* Einsteiger = grün */
div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(1):has(input:checked) {
    background: rgba(0,200,100,0.13);
    color: #00C864 !important;
    opacity: 1;
    box-shadow: 0 1px 8px rgba(0,200,100,0.22);
}
/* Fortgeschritten = orange */
div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(2):has(input:checked) {
    background: rgba(255,165,0,0.13);
    color: #FFA500 !important;
    opacity: 1;
    box-shadow: 0 1px 8px rgba(255,165,0,0.22);
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:last-child p {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    margin: 0 !important;
}

/* ── News-Refresh Button: aussehen wie ace-section Label ── */
button[data-testid="baseButton-secondary"][kind="secondary"]:has(+ *),
div[data-testid="stButton"]:has(button[key="btn_news_refresh"]) button,
button[key="btn_news_refresh"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    font-size: 0.72rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    text-transform: uppercase !important;
    letter-spacing: 0.13em !important;
    color: var(--text-color) !important;
    opacity: 0.42 !important;
    cursor: pointer !important;
    box-shadow: none !important;
    min-height: unset !important;
    height: auto !important;
    line-height: 1 !important;
}
button[key="btn_news_refresh"]:hover {
    opacity: 0.75 !important;
    background: transparent !important;
}

/* ── Velox Radar Teaser Button — hinter der Card verstecken ── */
div[data-testid="stButton"]:has(button[key="btn_radar_toggle"]) {
    margin-top: -0.5rem !important;
}
div[data-testid="stButton"]:has(button[key="btn_radar_toggle"]) button {
    background: transparent !important;
    border: none !important;
    color: transparent !important;
    font-size: 0 !important;
    padding: 0.35rem !important;
    box-shadow: none !important;
    cursor: pointer !important;
    height: 2rem !important;
    min-height: unset !important;
}

/* ── Neue Aktie Button — nur via JS-Klasse, kein breiter Selektor ── */
@keyframes vx-neue-pulse {
  0%,100% { outline: 2px solid rgba(16,185,129,0);    outline-offset: 0px; }
  50%      { outline: 2px solid rgba(16,185,129,0.65); outline-offset: 3px; }
}
button.vx-neue-aktie {
    background: rgba(0,200,100,0.06) !important;
    border: 1px solid rgba(0,200,100,0.4) !important;
    color: #00C864 !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    animation: vx-neue-pulse 2.2s ease-in-out infinite !important;
}
button.vx-neue-aktie:hover {
    background: rgba(0,200,100,0.12) !important;
    animation-play-state: paused !important;
    transform: scale(1.02) !important;
}

/* ── Haupt-Tab aktiv: Velox-Farbe statt Streamlit-Rot ── */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--text-color) !important;
    opacity: 0.85;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] p {
    color: var(--text-color) !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}
.stTabs [data-baseweb="tab"][aria-selected="false"] p {
    opacity: 0.45 !important;
}

/* ── Radar Mode-Radio (3 Optionen) — alle aktiv = Amber ── */
div[data-testid="stRadio"]:has(label:nth-child(3))
  > div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(245,158,11,0.14) !important;
    color: #f59e0b !important;
    box-shadow: 0 1px 8px rgba(245,158,11,0.25) !important;
    opacity: 1 !important;
}

/* ── Radar Vollanalyse Button — andocken an Card, kein Gap ── */
div[data-testid="stButton"]:has(button[key*="rc_"]) {
    margin-top: -0.45rem !important;
}
div[data-testid="stButton"]:has(button[key*="rc_"]) button {
    border-radius: 0 0 14px 14px !important;
    border-top: none !important;
    border-color: rgba(128,128,128,0.15) !important;
    background: transparent !important;
    color: var(--text-color) !important;
    opacity: 0.6 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    height: 2.4rem !important;
    min-height: unset !important;
    transition: opacity 0.15s, background 0.15s !important;
}
div[data-testid="stButton"]:has(button[key*="rc_"]) button:hover {
    opacity: 1 !important;
    background: rgba(16,185,129,0.06) !important;
    color: #10b981 !important;
}

/* ── KI-Radar "Analysieren" Button — Premium Amber ── */
div[data-testid="stButton"]:has(button[key="ki_go"]) button {
    background: linear-gradient(135deg, rgba(245,158,11,0.18), rgba(251,191,36,0.12)) !important;
    border: 1.5px solid rgba(245,158,11,0.5) !important;
    color: #f59e0b !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    font-size: 0.82rem !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 2px 8px rgba(245,158,11,0.2) !important;
}
div[data-testid="stButton"]:has(button[key="ki_go"]) button:hover {
    background: linear-gradient(135deg, rgba(245,158,11,0.28), rgba(251,191,36,0.2)) !important;
    box-shadow: 0 4px 14px rgba(245,158,11,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Radar Theme "Öffnen" Button — Teil der Kachel ── */
button.vx-theme-open {
    background: transparent !important;
    border: 1px solid rgba(128,128,128,0.15) !important;
    border-top: none !important;
    border-radius: 0 0 14px 14px !important;
    color: var(--text-color) !important;
    opacity: 0.55 !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    padding: 0.5rem !important;
    margin-top: -0.5rem !important;
    height: 2.4rem !important;
    min-height: unset !important;
    transition: background 0.15s, opacity 0.15s !important;
}
button.vx-theme-open:hover {
    background: rgba(245,158,11,0.08) !important;
    color: #f59e0b !important;
    opacity: 1 !important;
    border-color: rgba(245,158,11,0.3) !important;
}

/* ── Velox Radar Watchlist Button — hinter der Brand-Card ── */
div[data-testid="stButton"]:has(button[key*="wl_radar_"]) {
    margin-top: -0.4rem !important;
}
div[data-testid="stButton"]:has(button[key*="wl_radar_"]) button {
    background: transparent !important;
    border: none !important;
    color: transparent !important;
    font-size: 0 !important;
    padding: 0.3rem !important;
    box-shadow: none !important;
    cursor: pointer !important;
    height: 1.8rem !important;
    min-height: unset !important;
}

/* ── Ins Portfolio Button ── */
button.vx-ins-portfolio {
    background: rgba(16,185,129,0.10) !important;
    border: 1px solid rgba(16,185,129,0.45) !important;
    color: #10b981 !important;
    font-weight: 600 !important;
    transition: background 0.15s ease !important;
}
button.vx-ins-portfolio:hover {
    background: rgba(16,185,129,0.18) !important;
}
button.vx-ins-portfolio:active { transform: scale(0.98) !important; }

/* ── Watchlist Cards ── */
.vx-wl-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.14);
    border-radius: 14px;
    padding: 1.1rem 1.2rem 0.85rem 1.2rem;
    margin-bottom: 0.9rem;
    position: relative;
    overflow: hidden;
}
.vx-wl-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3px;
    border-radius: 14px 14px 0 0;
}
.vx-wl-green::before  { background: #10b981; box-shadow: 0 0 12px rgba(16,185,129,0.3); }
.vx-wl-orange::before { background: #f59e0b; }
.vx-wl-red::before    { background: #ef4444; }
.vx-wl-gray::before   { background: rgba(128,128,128,0.3); }

.vx-wl-score-bar {
    height: 3px;
    background: rgba(128,128,128,0.12);
    border-radius: 2px;
    overflow: hidden;
    margin-top: 0.2rem;
}
.vx-wl-action-btn button {
    font-size: 0.72rem !important;
    padding: 0.3rem 0.6rem !important;
    min-height: unset !important;
    height: 2.2rem !important;
}

/* ── Plotly Modebar Styling ── */
.modebar {
    background: transparent !important;
}
.modebar-btn path {
    fill: var(--text-color) !important;
    opacity: 0.3 !important;
}
.modebar-btn:hover path {
    opacity: 0.7 !important;
}
.modebar-btn.active path {
    opacity: 0.8 !important;
    fill: #10b981 !important;
}

/* ── Plotly Chart Cards ── */
[data-testid="stPlotlyChart"] {
    background: var(--secondary-background-color) !important;
    border: 1px solid rgba(128,128,128,0.12) !important;
    border-radius: 12px !important;
    overflow: hidden;
    padding: 0.25rem 0.1rem 0 0.1rem;
    margin-bottom: 0.4rem;
}

/* ── Expander Restyling — minimal, kein Eingriff in Streamlit-Internals ── */
[data-testid="stExpander"] {
    border: 1px solid rgba(128,128,128,0.12) !important;
    border-radius: 10px !important;
    background: var(--secondary-background-color) !important;
    margin-top: 0.4rem !important;
}
/* Nur den sichtbaren Text im Header verkleinern */
[data-testid="stExpander"] .streamlit-expanderHeader p,
[data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] + div p,
[data-testid="stExpander"] summary > div p {
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    opacity: 0.55 !important;
    margin: 0 !important;
}
/* Content-Bereich */
[data-testid="stExpanderDetails"] {
    padding: 0 1rem 0.75rem 1rem !important;
    font-size: 0.88rem !important;
    line-height: 1.65 !important;
}

/* ── Level Toggle ── */
.vx-level-bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.13);
    border-radius: 12px;
    padding: 0.55rem 0.8rem;
    margin-bottom: 1rem;
}
.vx-level-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--text-color);
    opacity: 0.4;
    white-space: nowrap;
    flex-shrink: 0;
}
.vx-level-active {
    background: rgba(16,185,129,0.12) !important;
    border-color: rgba(16,185,129,0.35) !important;
    color: #10b981 !important;
}

/* ── Beginner Glossar-Bullet ── */
.vx-detail-bullet {
    display: flex;
    gap: 0.5rem;
    padding: 0.45rem 0;
    border-bottom: 1px solid rgba(128,128,128,0.07);
    font-size: 0.87rem;
    line-height: 1.55;
    color: var(--text-color);
}
.vx-detail-bullet:last-child { border-bottom: none; }
.vx-detail-dot {
    flex-shrink: 0;
    width: 5px; height: 5px;
    border-radius: 50%;
    margin-top: 0.55rem;
    background: rgba(128,128,128,0.4);
}

/* ── Legacy aliases (Kompatibilität) ── */
.ace-score-lbl { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em; color:var(--text-color); opacity:0.45; margin-bottom:3px; }
.ace-score-num { font-size:2.8rem; font-weight:700; font-family:'Space Grotesk',monospace; line-height:1; }
.ace-hint      { font-size:0.84rem; margin-top:0.2rem; line-height:1.55; color:var(--text-color); opacity:0.65; }
.ace-green { color:#00C864; } .ace-orange { color:#FFA500; } .ace-red { color:#FF4444; }

</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Helpers — UI
# ══════════════════════════════════════════════════════════════════════════════
def score_color_cls(s):
    if s is None: return "ace-orange"
    return "ace-green" if s >= 6.5 else "ace-red" if s < 5.0 else "ace-orange"

def score_color_hex(s):
    if s is None: return "#FFA500"
    return "#00C864" if s >= 6.5 else "#FF4444" if s < 5.0 else "#FFA500"

def render_score_card(label, score, hint, details, key, level="pro"):
    hex_c = score_color_hex(score)
    cls_v = "vx-green" if (score or 0) >= 6.5 else "vx-red" if (score or 0) < 5.0 else "vx-orange"
    val   = f"{score:.1f}" if score else "—"
    pct   = int((score or 0) * 10)
    st.markdown(
        f'<div class="vx-score-card {cls_v}">'
        f'  <div class="vx-score-label">{label}</div>'
        f'  <div class="vx-score-number" style="color:{hex_c};">'
        f'    {val}<span class="vx-score-denom">/10</span>'
        f'  </div>'
        f'  <div class="vx-progress-track">'
        f'    <div class="vx-progress-fill" style="width:{pct}%;background:{hex_c};"></div>'
        f'  </div>'
        f'  <div class="vx-hint">{hint}</div>'
        f'</div>',
        unsafe_allow_html=True)
    if details:
        if level == "beginner":
            exp_label = "Was bedeutet das?"
            translated = [beginner_translate(d) for d in details]
        else:
            exp_label = "Details"
            translated = details
        with st.expander(exp_label, expanded=False):
            bullets_html = "".join(
                f'<div class="vx-detail-bullet">'
                f'<div class="vx-detail-dot"></div>'
                f'<div>{d}</div>'
                f'</div>'
                for d in translated
            )
            st.markdown(bullets_html, unsafe_allow_html=True)

def render_action_banner(action, total, why):
    go_kw  = ["Einstieg", "Nachkauf", "Halten +"]
    stop_kw= ["kein Einstieg", "streichen", "Kein Einstieg"]
    if any(k in action for k in go_kw):
        cls = "ace-go"
    elif any(k in action for k in stop_kw):
        cls = "ace-stop"
    else:
        cls = "ace-wait"
    why_txt = " · ".join(why[:2]) if why else ""
    st.markdown(
        f'<div class="{cls}">'
        f'<div style="font-size:0.72rem;font-family:\'Space Grotesk\',sans-serif;'
        f'color:var(--text-color);opacity:0.45;text-transform:uppercase;letter-spacing:0.13em;">'
        f'Empfehlung · {total:.1f}/10</div>'
        f'<div style="font-size:1.45rem;font-weight:700;font-family:\'Space Grotesk\',sans-serif;'
        f'margin:0.2rem 0 0.15rem 0;letter-spacing:-0.01em;">{action}</div>'
        f'<div style="font-size:0.84rem;color:var(--text-color);opacity:0.6;line-height:1.5;">{why_txt}</div>'
        f'</div>', unsafe_allow_html=True)

def trigger_cls(t):
    t_low = t.lower()
    go_w  = ["aktiv:", "kreuzte signal nach oben", "macd bullisch", "gut abgekühlt", "solide mitte",
             "rückenwind", "bestätigung", "aufwärts", "halten", "stützt entry"]
    stop_w= ["überkauft", "bärisch", "verlust-kontrolle", "kein nachkauf", "dreht negativ",
              "gegenwind", "nicht aktuell", "schwach"]
    if any(w in t_low for w in go_w):  return "trig-go"
    if any(w in t_low for w in stop_w): return "trig-stop"
    return "trig-wait"

def render_triggers(triggers):
    for t in triggers:
        st.markdown(f'<div class="{trigger_cls(t)}">• {t}</div>', unsafe_allow_html=True)

def beginner_hint_fund(score):
    if score is None: return "Noch keine Fundamentalanalyse gestartet."
    if score >= 7: return "Das Unternehmen steht finanziell sehr solide da — attraktive Kennzahlen."
    if score >= 6: return "Ordentliche Fundamentaldaten, kleinere Schwachstellen sind vorhanden."
    if score >= 5: return "Gemischtes Bild — manche Kennzahlen passen, andere sind erhöht."
    return "Die Fundamentaldaten sind schwach oder ambitioniert bewertet — Vorsicht."

def beginner_hint_timing(score):
    if score is None: return "Noch kein Chart analysiert."
    if score >= 7: return "Guter Zeitpunkt — der Chart zeigt ein sauberes Entry-Fenster."
    if score >= 6: return "Timing ist ok, aber nicht perfekt. Etwas Geduld kann helfen."
    if score >= 5: return "Unruhiges Bild im Chart — besser auf Bestätigung warten."
    return "Schlechtes Timing — der Chart signalisiert Druck oder Schwäche."

def timing_summary_text(score, reasons: list, level: str = "pro") -> str:
    """2-Zeiler: Zeile 1 = was der Chart zeigt, Zeile 2 = was das bedeutet."""
    if score is None: return "Noch kein Chart analysiert."
    # Schlüsselsignale aus reasons extrahieren
    _r = " ".join(reasons).lower() if reasons else ""
    # Trend
    if "ma50 > ma200" in _r or "rückenwind" in _r:         _trend = "aufwärts"
    elif "ma50 < ma200" in _r or "gegenwind" in _r:        _trend = "abwärts"
    else:                                                   _trend = "neutral"
    # MACD
    if "bullische kreuzung" in _r:                         _macd = "dreht bullisch"
    elif "bärische kreuzung" in _r:                        _macd = "dreht bärisch"
    elif "bullisch" in _r and "macd" in _r:                _macd = "bullisch"
    elif "bärisch" in _r and "macd" in _r:                 _macd = "bärisch"
    else:                                                   _macd = None
    # MA-Nähe
    if "entry-fenster" in _r and "ma20" in _r:             _ma = "nahe MA20 — Entry-Fenster"
    elif "weit über ma20" in _r:                           _ma = "überdehnt über MA20"
    elif "unter ma20" in _r:                               _ma = "unter MA20"
    else:                                                   _ma = None
    # Konsolidierung
    if "sammelt sich" in _r:                               _cons = "sammelt sich"
    elif "konsolidierung eher breit" in _r:                _cons = "unruhig konsolidiert"
    else:                                                   _cons = None

    if level == "beginner":
        # Klartextsprache für Einsteiger
        if score >= 7.5:
            line1 = "Der Chart sieht gut aus — ruhige Bewegung, kein Überkauf."
            line2 = "Ein guter Moment zum Einsteigen. Gestaffelt kaufen ist trotzdem klüger."
        elif score >= 6.5:
            line1 = "Der Chart ist in Ordnung — kein Alarmsignal, aber auch kein perfekter Einstieg."
            line2 = "Wer einsteigen will, kann es tun. Geduld bis zur nächsten Ruhephase zahlt sich aber aus."
        elif score >= 5.5:
            line1 = "Der Chart zeigt ein gemischtes Bild — mal rauf, mal runter."
            line2 = "Besser warten, bis sich der Kurs beruhigt hat. Kein Druck."
        elif score >= 4.5:
            line1 = "Der Chart ist gerade unruhig — der Kurs schwankt stärker als normal."
            line2 = "Einstieg jetzt ist riskanter. Lieber abwarten und beobachten."
        else:
            line1 = "Der Chart zeigt deutliche Schwäche — Abverkaufsdruck vorhanden."
            line2 = "Jetzt einzusteigen wäre schlechtes Timing. Erst wenn sich das Bild aufhellt."
    else:
        # Fortgeschritten: technische Sprache mit echten Signalen
        parts1 = []
        if _trend != "neutral": parts1.append(f"Trend {_trend}")
        if _ma:                  parts1.append(_ma)
        if _macd:                parts1.append(f"MACD {_macd}")
        if _cons:                parts1.append(_cons)
        if not parts1:
            parts1 = ["Gemischte Signale"]
        line1 = " · ".join(parts1[:3]) + "."
        if score >= 7.5:
            line2 = "Entry-Fenster ist offen — kein Momentum-Jagen, gestaffelter Kauf ist ideal."
        elif score >= 6.5:
            line2 = "Timing solide. Entry möglich, aber Bestätigung über MA20 abwarten."
        elif score >= 5.5:
            line2 = "Chart neutral — kein klares Signal. Geduld oder kleinen Starter."
        elif score >= 4.5:
            line2 = "Erhöhter Druck im Chart. Gestaffelt nur bei klarer Verbesserung."
        else:
            line2 = "Schlechtes Timing — Distribution/Abgabedruck sichtbar. Abwarten."
    return f"{line1}<br><span style='opacity:0.65;font-size:0.9em;'>{line2}</span>"

def beginner_hint_story(score):
    if score is None: return "Kein Profil geladen."
    if score >= 7: return "Das Geschäftsmodell passt sehr gut zur gewählten Strategie."
    if score >= 5.5: return "Modell passt teilweise — einige Aspekte stimmen, andere weniger."
    return "Das Geschäftsmodell passt nicht ideal zur gewählten Strategie."

# ── Einsteiger-Übersetzungen ──────────────────────────────────────────────────
def beginner_translate(detail: str) -> str:
    """Übersetzt technische Kennzahlen-Bullets in verständliche Einsteiger-Sprache."""
    import re
    d = detail.strip()

    # Beta / Volatilität
    if "Beta" in d:
        m = re.search(r"Beta\s*([\d.,]+)", d)
        val = m.group(1) if m else "?"
        if any(w in d.lower() for w in ["volatil", "zu hoch", "hoch"]):
            return (f"Kursschwankung (Beta {val}): Diese Aktie schwankt stärker als der Markt — "
                    "das bedeutet mehr Risiko, aber auch mehr Chancen.")
        return (f"Kursschwankung (Beta {val}): Die Aktie bewegt sich ähnlich wie der "
                "Gesamtmarkt — normale Schwankungsbreite.")

    # KGV
    if "KGV" in d:
        m = re.search(r"KGV\s*([\d.,]+)", d)
        val = m.group(1) if m else "?"
        if any(w in d.lower() for w in ["sehr teuer", "hoch"]):
            return (f"Preis-Gewinn-Verhältnis (KGV {val}): Für jeden Euro Gewinn zahlen "
                    "Anleger viel — die Aktie ist hoch bewertet.")
        if any(w in d.lower() for w in ["günstig", "niedrig", "moderat"]):
            return (f"Preis-Gewinn-Verhältnis (KGV {val}): Die Aktie ist günstig bewertet "
                    "im Verhältnis zu ihrem Gewinn — positives Signal.")
        return (f"Preis-Gewinn-Verhältnis (KGV {val}): Bewertung liegt im normalen Bereich.")

    # PEG
    if d.lstrip().startswith("PEG") or "PEG " in d[:15]:
        m = re.search(r"PEG\s*([\d.,]+)", d)
        val = m.group(1) if m else "?"
        if any(w in d.lower() for w in ["gut", "vertretbar", "fair"]):
            return (f"Wachstums-Bewertung (PEG {val}): Der Preis ist fair für das "
                    "Wachstum, das das Unternehmen liefert.")
        return (f"Wachstums-Bewertung (PEG {val}): Gemessen am Wachstum erscheint "
                "die Bewertung etwas hoch.")

    # KUV
    if "KUV" in d:
        m = re.search(r"KUV\s*([\d.,]+)", d)
        val = m.group(1) if m else "?"
        if any(w in d.lower() for w in ["angenehm", "günstig", "niedrig"]):
            return (f"Preis-Umsatz-Verhältnis (KUV {val}): Günstig bewertet gemessen "
                    "am Jahresumsatz — gutes Zeichen.")
        if any(w in d.lower() for w in ["hoch", "teuer"]):
            return (f"Preis-Umsatz-Verhältnis (KUV {val}): Gemessen am Umsatz ist "
                    "die Aktie teuer — Vorsicht bei schwachen Margen.")
        return f"Preis-Umsatz-Verhältnis (KUV {val}): Bewertung im normalen Bereich."

    # KBV
    if "KBV" in d:
        m = re.search(r"KBV\s*([\d.,]+)", d)
        val = m.group(1) if m else "?"
        if any(w in d.lower() for w in ["sehr hoch", "hoch"]):
            return (f"Preis-Buchwert-Verhältnis (KBV {val}): Anleger zahlen ein "
                    "Vielfaches des Buchwertes — typisch für starke Marken und Plattformen.")
        return (f"Preis-Buchwert-Verhältnis (KBV {val}): Bewertung zum Eigenkapital "
                "liegt im normalen Bereich.")

    # Dividende
    if "Dividende" in d or "Div." in d:
        m = re.search(r"([\d.,]+)\s*%", d)
        val = m.group(1) if m else "?"
        if any(w in d.lower() for w in ["sehr hoch"]):
            return (f"Dividende ({val}%): Das Unternehmen schüttet sehr viel aus — "
                    "prüfen, ob das langfristig nachhaltig ist.")
        if any(w in d.lower() for w in ["solide", "gut", "ordentlich"]):
            return (f"Dividende ({val}%): Das Unternehmen zahlt eine solide Dividende — "
                    "du bekommst regelmäßig einen Teil des Gewinns ausgezahlt.")
        if any(w in d.lower() for w in ["keine", "kein"]):
            return ("Keine Dividende: Das Unternehmen zahlt keine Ausschüttung — "
                    "reinvestiert die Gewinne lieber ins eigene Wachstum.")
        return f"Dividende ({val}%): Das Unternehmen beteiligt seine Aktionäre am Gewinn."

    # Datenlage
    if "Datenlage" in d:
        m = re.search(r"(\d+)\s*/\s*(\d+)", d)
        if m:
            has, tot = int(m.group(1)), int(m.group(2))
            if has >= tot - 1:
                return (f"Datenverfügbarkeit ({has}/{tot}): Fast alle Kennzahlen sind "
                        "abrufbar — die Analyse steht auf einer soliden Basis.")
            return (f"Datenverfügbarkeit ({has}/{tot}): Nicht alle Kennzahlen sind "
                    "verfügbar — die Analyse ist weniger vollständig.")
        return "Datenverfügbarkeit: Basis für die Analyse."

    # ── Timing / Chart-Bullets ────────────────────────────────────────────────
    d_lower = d.lower()
    if "trendstruktur" in d_lower:
        if "gesunder" in d_lower or "höheres tief" in d_lower:
            return "Kurstrend: Die Aktie läuft nach einem kleinen Rücksetzer wieder nach oben — gutes Zeichen."
        if "angeschlagen" in d_lower or "unter ma" in d_lower:
            return "Kurstrend: Die Aktie kämpft gerade — der Kurs liegt unterhalb wichtiger Durchschnittswerte."
        return "Kurstrend: " + d.split("→")[-1].strip()
    if "rsi" in d_lower:
        m = re.search(r"RSI[:\s]*([\d.]+)", d)
        val = m.group(1) if m else "?"
        if "abgekühlt" in d_lower or "brauchbarer" in d_lower:
            return f"Kaufdruck-Indikator (RSI {val}): Die Aktie ist gerade nicht überhitzt — guter Ausgangspunkt."
        if "heißgelaufen" in d_lower or "überkauft" in d_lower:
            return f"Kaufdruck-Indikator (RSI {val}): Viele Anleger haben bereits gekauft — kurzfristig etwas überhitzt."
        if "schwach" in d_lower:
            return f"Kaufdruck-Indikator (RSI {val}): Wenig Kaufinteresse gerade — Kurs könnte noch etwas fallen."
        return f"Kaufdruck-Indikator (RSI {val}): Im normalen Bereich."
    if "macd" in d_lower:
        if "bullisch" in d_lower or "überkreuzung" in d_lower:
            return "Trendwechsel-Signal: Ein technisches Kaufsignal wurde ausgelöst — positiv."
        if "bärisch" in d_lower:
            return "Trendwechsel-Signal: Ein Warnsignal zeigt sich — Vorsicht mit Neueinsteigen."
        return "Trendwechsel-Signal: Neutral, kein klares Signal."
    if "ma200" in d_lower or "200" in d and "trend" in d_lower:
        if "intakt" in d_lower or "aufwärts" in d_lower:
            return "Langfristiger Trend: Die Aktie befindet sich in einem langfristigen Aufwärtstrend — stabiles Umfeld."
        if "darunter" in d_lower or "unterhalb" in d_lower:
            return "Langfristiger Trend: Die Aktie liegt unter ihrem Langfristdurchschnitt — schwächeres Umfeld."
    if "ausbruch" in d_lower:
        return "Kursausbruch: Die Aktie hat ein Kursziel nach oben durchbrochen — kann weitere Käufer anziehen."
    if "kerze" in d_lower:
        if "grün" in d_lower or "bestätigung" in d_lower:
            return "Tageskurs: Heute schloss die Aktie positiv — kleines grünes Kaufsignal."
        return "Tageskurs: " + d.split("→")[-1].strip() if "→" in d else d

    # ── Story-Gründe ──────────────────────────────────────────────────────────
    story_map = [
        ("defensiver charakter",     "Das Unternehmen ist defensiv aufgestellt — ideal für ein stabiles Langzeitinvestment."),
        ("kapitalintensität",        "Das Unternehmen braucht viel Kapital — kann Gewinne, aber auch Risiken verstärken."),
        ("skalierbar",               "Das Geschäftsmodell wächst effizient — mehr Umsatz braucht kaum mehr Kosten."),
        ("wiederkehrende einnahmen", "Das Unternehmen hat planbare, regelmäßige Einnahmen — gut für Stabilität."),
        ("pricing power",            "Das Unternehmen kann Preise erhöhen, ohne Kunden zu verlieren — starke Marktposition."),
        ("regulierung",              "Staatliche Regulierung sorgt für planbarere Einnahmen — gut für Stabilität."),
        ("nischenstärke",            "Das Unternehmen beherrscht eine spezielle Nische — schwer angreifbar."),
        ("margen",                   "Die Gewinnspannen zeigen, wie profitabel das Unternehmen wirklich ist."),
        ("moat",                     "Das Unternehmen hat einen klaren Wettbewerbsvorteil — schwer zu kopieren."),
        ("kapital light",            "Das Unternehmen wächst ohne viel Kapital zu brauchen — sehr effizient."),
        ("nischenmonopol",           "Das Unternehmen dominiert seine Nische ohne nennenswerte Konkurrenz."),
        ("core-fit",                 "Das Geschäftsmodell passt gut zu einer langfristigen, stabilen Strategie."),
        ("hc-fit",                   "Das Unternehmen zeigt Merkmale eines versteckten Marktführers."),
    ]
    for keyword, translation in story_map:
        if keyword in d_lower:
            return translation

    # ── Entry-Trigger Übersetzungen ──────────────────────────────────────────
    # Trendkontext
    if "trendkontext" in d_lower:
        if "rückenwind" in d_lower or "ma50" in d_lower and "ma200" in d_lower and ">" in d:
            return "Langfristiger Trend: Der Kurs ist langfristig im Aufwärtstrend — gutes Umfeld für einen Einstieg."
        if "negativ" in d_lower or "<" in d and "ma50" in d_lower:
            return "Langfristiger Trend: Der Kurs ist langfristig eher schwach. Größeren Kauf lieber noch etwas abwarten."
        return "Langfristiger Trend: Keine klare Richtung erkennbar — Geduld ist angebracht."
    # Entry-Zone
    if "entry-zone aktiv" in d_lower or "einstiegsfenster ist jetzt offen" in d_lower:
        return "Jetzt ist ein guter Moment — der Kurs ist nahe seinem Durchschnittspreis. Gestaffelt einsteigen ist ideal."
    if "warte auf rücksetzer" in d_lower or "weit über" in d_lower and "ma20" in d_lower:
        return "Noch etwas warten — der Kurs ist gerade stark gestiegen. Ein günstigerer Einstieg kommt wahrscheinlich noch."
    if "leicht über ma20" in d_lower or "kleine erste tranche" in d_lower:
        return "Ein kleiner Erstkauf ist möglich, aber nicht der perfekte Zeitpunkt. Lieber auf einen Rücksetzer warten."
    if "unter ma20" in d_lower and "warte auf tagesschluss" in d_lower:
        return "Noch abwarten — der Kurs ist unter seinem Durchschnitt. Erst kaufen, wenn er sich wieder erholt."
    # RSI Trigger
    if "rsi-filter" in d_lower and "überkauft" in d_lower:
        return "Abkühlung abwarten — die Aktie ist gerade sehr gefragt. Besser einsteigen wenn sich der Andrang legt."
    if "rsi-timing" in d_lower and ("abgekühlt" in d_lower or "gut abgekühlt" in d_lower):
        return "Guter Moment — die Aktie ist weder überhitzt noch am Boden. Jetzt einsteigen macht Sinn."
    if "rsi" in d_lower and "schwach" in d_lower and "stabilisierung" in d_lower:
        return "Noch Geduld — die Aktie hat zuletzt nachgegeben. Erst kaufen wenn sie zwei grüne Tage in Folge zeigt."
    # MACD Trigger
    if "macd-kreuzung bullisch" in d_lower:
        return "Frisches Kaufsignal — ein technischer Indikator hat gerade nach oben gedreht. Zeitnah handeln sinnvoll."
    if "macd-kreuzung bärisch" in d_lower:
        return "Kein guter Zeitpunkt — ein technischer Indikator hat nach unten gedreht. Entry lieber verschieben."
    if "macd bullisch" in d_lower:
        return "Momentum zeigt nach oben — unterstützt einen Einstieg."
    if "macd bärisch" in d_lower:
        return "Momentum zeigt nach unten — lieber auf bessere Signale warten bevor du kaufst."
    # 52W-Position
    if "52w-position" in d_lower:
        if "nahe jahrestief" in d_lower or "mögliches schnäppchen" in d_lower:
            return "Jahrestief-Nähe: Die Aktie ist nahe ihrem günstigsten Preis der letzten 52 Wochen — mögliches Schnäppchen. Erst Stabilisierung abwarten."
        if "nahe jahreshoch" in d_lower or "wenig luft" in d_lower:
            return "Nahe Jahreshoch: Die Aktie ist nahe ihrem Jahreshoch — wenig Puffer nach oben. Nur bei sehr starkem Trend einsteigen."
        if "solide mitte" in d_lower or "ausgewogen" in d_lower:
            return "Die Aktie ist in der Mitte ihrer Jahrespreisspanne — ausgewogenes Chance/Risiko-Verhältnis."
        # Extrahiere die Prozentzahl
        import re as _re
        _m = _re.search(r"(\d+)%", d)
        _pos = _m.group(1) if _m else "?"
        return f"Jahrespreisposition ({_pos}% der Jahresspanne): Im mittleren Bereich — kein extremes Hoch oder Tief."
    # Portfolioposition-Trigger
    if "nachkauf-disziplin" in d_lower:
        return "Du liegst gut im Plus — beim Nachkauf lieber auf einen Rücksetzer warten, nicht der Stärke hinterherrennen."
    if "verlust-kontrolle" in d_lower:
        return "Die Position ist im Minus — kein emotionaler Nachkauf jetzt. Erst warten bis das Chart-Bild sich verbessert."
    if "break-even" in d_lower or "break-even-zone" in d_lower:
        return "Die Position ist ungefähr bei deinem Einstiegspreis — Geduld. Nachkauf erst wenn klare positive Signale kommen."

    return d  # Fallback: Original zurückgeben

# ══════════════════════════════════════════════════════════════════════════════
# Session State
# ══════════════════════════════════════════════════════════════════════════════
for k, v in [("fund_score",None),("fund_reasons",[]),("timing_score",None),("timing_reasons",[]),
             ("story_score",None),("story_reasons",[]),("story_info",None),("chart_df",None),
             ("chart_bg",[]),("last_action",""),("last_why",[]),("ace_long_fazit",""),
             ("ace_long_key",""),("red_flags",[]),("entry_triggers",[]),("risk_hints",[]),
             ("show_radar",False),("auto_run_fund",False),
             ("vr_sim_limit",3),("vr_rad_limit",3),
             ("user_level","beginner"),("news_rv",0)]:
    if k not in st.session_state: st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# Ticker Tape
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def fetch_ticker_tape() -> list:
    """Lädt Live-Kurse für den Ticker-Tape. 5-Minuten-Cache."""
    items = [
        ("^GSPC",  "S&P 500"),
        ("^IXIC",  "Nasdaq"),
        ("^STOXX50E", "Euro Stoxx 50"),
        ("^FTSE",  "FTSE 100"),
        ("^GDAXI", "DAX"),
        ("GC=F",   "Gold"),
        ("CL=F",   "Crude Oil"),
        ("^TNX",   "US 10Y"),
        ("EURUSD=X","EUR/USD"),
        ("BTC-USD", "Bitcoin"),
    ]
    results = []
    for sym, label in items:
        try:
            info = yf.Ticker(sym).info or {}
            price = safe_float(info.get("regularMarketPrice") or info.get("currentPrice"))
            prev  = safe_float(info.get("regularMarketPreviousClose"))
            if price and prev and prev > 0:
                chg_pct = (price - prev) / prev * 100
                results.append({"label": label, "price": price, "chg": chg_pct})
        except Exception:
            pass
    return results

def render_ticker_tape():
    tape = fetch_ticker_tape()
    if not tape:
        return
    # Baue Chips HTML
    chips = ""
    for t in tape:
        color = "#22c55e" if t["chg"] >= 0 else "#ef4444"
        arrow = "▲" if t["chg"] >= 0 else "▼"
        sign  = "+" if t["chg"] >= 0 else ""
        chips += (
            f'<span style="display:inline-flex;align-items:center;gap:0.4rem;'
            f'background:var(--secondary-background-color);'
            f'border:1px solid rgba(128,128,128,0.15);'
            f'border-radius:6px;padding:0.25rem 0.75rem;white-space:nowrap;">'
            f'<span style="font-size:0.78rem;font-weight:600;'
            f'font-family:\'Space Grotesk\',sans-serif;color:var(--text-color);">'
            f'{t["label"]}</span>'
            f'<span style="font-size:0.78rem;font-family:monospace;color:var(--text-color);opacity:0.8;">'
            f'{t["price"]:,.2f}</span>'
            f'<span style="font-size:0.72rem;color:{color};font-weight:600;">'
            f'{arrow} {sign}{t["chg"]:.2f}%</span>'
            f'</span>'
        )
    # Doppeln für nahtloses Scrollen
    doubled = chips + "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" + chips
    st.markdown(f"""
<style>
@keyframes velox-scroll {{
  0%   {{ transform: translateX(0); }}
  100% {{ transform: translateX(-50%); }}
}}
.velox-tape-wrap {{
  overflow: hidden;
  width: 100%;
  padding: 0.4rem 0 0.6rem 0;
  mask-image: linear-gradient(to right, transparent 0%, black 4%, black 96%, transparent 100%);
  -webkit-mask-image: linear-gradient(to right, transparent 0%, black 4%, black 96%, transparent 100%);
}}
.velox-tape-inner {{
  display: inline-flex;
  gap: 0.5rem;
  animation: velox-scroll 40s linear infinite;
  will-change: transform;
}}
.velox-tape-wrap:hover .velox-tape-inner {{
  animation-play-state: paused;
}}
</style>
<div class="velox-tape-wrap">
  <div class="velox-tape-inner">{doubled}</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════
_hc, _lvc, _rc = st.columns([3.5, 3.5, 1.4])
with _hc:
    st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
<div style="padding:0.5rem 0 1rem 0;">
  <div style="font-family:'Space Grotesk',sans-serif;font-size:2.5rem;font-weight:700;
    letter-spacing:-0.03em;line-height:1;color:var(--text-color);">Velox</div>
  <div style="font-family:'Space Grotesk',sans-serif;font-size:0.66rem;
    letter-spacing:0.22em;text-transform:uppercase;color:var(--text-color);
    opacity:0.35;margin-top:0.28rem;">Stock Check
    <span style="opacity:0.55;font-size:0.56rem;letter-spacing:0.1em;">v9.0</span></div>
</div>
""", unsafe_allow_html=True)
with _lvc:
    # Vertikale Ausrichtung: gleicher Abstand wie Logo-Padding
    st.markdown('<div style="height:0.85rem;"></div>', unsafe_allow_html=True)
    _lvl_cur = st.session_state.get("user_level", "beginner")
    _lvl_radio = st.radio(
        "Level", ["Einsteiger", "Fortgeschritten"],
        horizontal=True,
        label_visibility="collapsed",
        index=0 if _lvl_cur == "beginner" else 1,
        key="hdr_level_radio",
    )
    st.session_state["user_level"] = "beginner" if _lvl_radio == "Einsteiger" else "pro"
with _rc:
    st.markdown('<div style="height:0.85rem;"></div>', unsafe_allow_html=True)
    # Top Bar: immer "＋ Neue Aktie" — "Portfolio einrichten" nur im Portfolio-Tab selbst
    if st.button("＋ Neue Aktie", key="reset_top", use_container_width=True):
        for k, v in [
            ("fund_score",None),("fund_reasons",[]),
            ("timing_score",None),("timing_reasons",[]),
            ("story_score",None),("story_reasons",[]),("story_info",None),
            ("chart_df",None),("chart_bg",[]),
            ("last_action",""),("last_why",[]),
            ("ace_long_fazit",""),("ace_long_key",""),
            ("red_flags",[]),("entry_triggers",[]),("risk_hints",[]),
            ("ace_search_results",[]),("ace_search_q",""),
            ("ace_selected_ticker",""),("ace_selected_name",""),("ace_selected_isin",""),
            ("ace_direct_ticker",""),("ace_yf_metrics",{}),("ace_yf_ticker",""),
            ("ace_mode_idx",0), ("show_radar",False),
            ("ace_ext_metrics",{}),
        ]:
            st.session_state[k] = v
        # Widget-Keys explizit löschen damit Felder wirklich leer erscheinen
        for _wk in ("ace_search_input", "ace_direct_ticker"):
            st.session_state.pop(_wk, None)
        st.rerun()

# Ticker Tape — nach Header, vor Tabs
render_ticker_tape()

# Button-Styling per JS (components.v1.html führt JS zuverlässig aus)
st_components.html("""
<script>
(function() {
  function applyStyle() {
    var btns = window.parent.document.querySelectorAll('button');
    btns.forEach(function(b) {
      var t = b.textContent.trim();
      if (t.indexOf('Neue Aktie') !== -1) {
        b.classList.add('vx-neue-aktie');
      }
      if (t === '✓ Jetzt ins Portfolio') {
        b.classList.add('vx-ins-portfolio');
      }
      if (t === 'Öffnen') {
        b.classList.add('vx-theme-open');
      }
    });
  }
  applyStyle();
  new MutationObserver(applyStyle).observe(
    window.parent.document.body, { childList: true, subtree: true }
  );
  // Styles + Keyframe einmalig in Parent-Doc injizieren
  var doc = window.parent.document;
  if (!doc.getElementById('vx-injected-styles')) {
    var s = doc.createElement('style');
    s.id = 'vx-injected-styles';
    s.textContent =
      '@keyframes vxNeuePulse{' +
        '0%,100%{outline:2px solid rgba(0,200,100,0);outline-offset:0px;}' +
        '50%{outline:2px solid rgba(0,200,100,0.65);outline-offset:3px;}}' +
      'button.vx-neue-aktie{' +
        'background:rgba(0,200,100,0.06)!important;' +
        'border:1px solid rgba(0,200,100,0.4)!important;' +
        'color:#00C864!important;font-weight:600!important;' +
        'letter-spacing:0.04em!important;' +
        'animation:vxNeuePulse 2.2s ease-in-out infinite!important;}' +
      'button.vx-neue-aktie:hover{' +
        'background:rgba(0,200,100,0.12)!important;' +
        'animation-play-state:paused!important;}';
    doc.head.appendChild(s);
  }
})();
</script>
""", height=0)

# ── Automatischer Tab-Wechsel zur Analyse ────────────────────────────────────
_switch_target = None
if st.session_state.get("_auto_switch_to_analyse"):
    st.session_state.pop("_auto_switch_to_analyse", None)
    _switch_target = "Analyse"
if st.session_state.get("_auto_switch_to_radar"):
    st.session_state.pop("_auto_switch_to_radar", None)
    _switch_target = "◎ Radar"

if _switch_target:
    _js_target = _switch_target.replace("'", "\\'")
    st_components.html(f"""
<script>
(function() {{
  var target = '{_js_target}';
  function switchTab() {{
    var tabs = window.parent.document.querySelectorAll('[data-testid="stTab"]');
    for (var i = 0; i < tabs.length; i++) {{
      if (tabs[i].textContent.trim() === target) {{
        tabs[i].click();
        window.parent.scrollTo({{top: 0, behavior: 'smooth'}});
        return true;
      }}
    }}
    return false;
  }}
  if (!switchTab()) {{ setTimeout(switchTab, 150); }}
}})();
</script>
""", height=0)

# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════
tab_analyse, tab_watchlist, tab_radar, tab_portfolio = st.tabs(
    ["Analyse", "Watchlist", "◎ Radar", "Portfolio"]
)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Analyse
# ──────────────────────────────────────────────────────────────────────────────
with tab_analyse:
    st.session_state["_active_tab"] = "Analyse"

    # ── Auto-Snapshot: Watchlist-Vollanalyse zurückspeichern ─────────────────
    _wl_save = st.session_state.get("wl_analyse_save_back")
    if _wl_save and st.session_state.get("fund_score") is not None:
        _fs_sb = st.session_state.get("fund_score")
        _ts_sb = st.session_state.get("timing_score")
        _ss_sb = st.session_state.get("story_score")
        _tot_sb = overall_score(_wl_save["mode"], _fs_sb, _ts_sb, _ss_sb)
        save_snapshot_to_watchlist({
            "ticker":       _wl_save["ticker"],
            "name":         _wl_save["name"],
            "mode":         _wl_save["mode"],
            "saved_at":     datetime.now().isoformat(),
            "fund_score":   round(_fs_sb, 2) if _fs_sb else None,
            "timing_score": round(_ts_sb, 2) if _ts_sb else None,
            "story_score":  round(_ss_sb, 2) if _ss_sb else None,
            "total_score":  round(_tot_sb, 2) if _tot_sb else None,
            "action":       st.session_state.get("last_action", ""),
            "triggers":     st.session_state.get("entry_triggers", []),
            "risks":        st.session_state.get("risk_hints", []),
            "metrics":      {}, "red_flags": st.session_state.get("red_flags", []),
        })
        st.session_state.pop("wl_analyse_save_back", None)

    # ── A1: Welcome-Lightbox — einmalig beim ersten Besuch ───────────────────
    if not st.session_state.get("welcome_shown"):
        st.markdown(
            '<div style="'
            'background:linear-gradient(135deg,'
            'rgba(16,185,129,0.10) 0%,rgba(59,130,246,0.06) 50%,rgba(139,92,246,0.08) 100%);'
            'border:1px solid rgba(16,185,129,0.25);border-radius:22px;'
            'padding:2.2rem 2.4rem 1.8rem 2.4rem;margin-bottom:1.5rem;'
            'position:relative;overflow:hidden;text-align:center;">'
            # Glow spots
            '<div style="position:absolute;top:-30%;left:-10%;width:300px;height:300px;'
            'background:radial-gradient(circle,rgba(16,185,129,0.10) 0%,transparent 65%);'
            'pointer-events:none;"></div>'
            '<div style="position:absolute;bottom:-30%;right:-8%;width:240px;height:240px;'
            'background:radial-gradient(circle,rgba(139,92,246,0.08) 0%,transparent 65%);'
            'pointer-events:none;"></div>'
            # Logo / brand
            '<div style="font-size:0.58rem;font-weight:900;letter-spacing:0.28em;'
            'text-transform:uppercase;color:#10b981;margin-bottom:1.1rem;'
            'opacity:0.85;">◎ Velox</div>'
            # Headline
            '<div style="font-size:2rem;font-weight:800;letter-spacing:-0.03em;'
            'line-height:1.2;color:var(--text-color);margin-bottom:0.7rem;">'
            'Hey, schön<br>dass du da bist!</div>'
            # Sub
            '<div style="font-size:0.92rem;color:var(--text-color);opacity:0.52;'
            'line-height:1.7;max-width:520px;margin:0 auto 1.6rem auto;">'
            'Velox analysiert Aktien mit echten Scores — Fundament, Timing, Story.<br>'
            'Starte mit einer Aktiensuche oder erkunde den Radar für neue Ideen.'
            '</div>'
            '</div>',
            unsafe_allow_html=True)
        _wlc1, _wlc2, _wlc3 = st.columns([1, 2, 1])
        with _wlc2:
            if st.button("Los geht's →", key="welcome_dismiss",
                         use_container_width=True, type="primary"):
                st.session_state["welcome_shown"] = True
                st.rerun()
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    # ─── LEFT: Input ──────────────────────────────────────────────────────────
    with left:
        st.markdown('<div class="ace-section">Eingabe</div>', unsafe_allow_html=True)

        # ── Aktiensuche ───────────────────────────────────────────────────────
        _srch_col, _srch_btn = st.columns([3, 1], vertical_alignment="bottom")
        with _srch_col:
            _srch_q = st.text_input("Aktie suchen (Name oder Ticker)",
                                     value=st.session_state.get("ace_search_q", ""),
                                     placeholder="z.B. ASML, Royal Gold, Nvidia…",
                                     key="ace_search_input",
                                     label_visibility="visible")
        with _srch_btn:
            _do_search = st.button("Suchen", key="btn_search",
                                   use_container_width=True)

        if _do_search and _srch_q:
            st.session_state["ace_search_q"] = _srch_q
            with st.spinner("Suche bei Yahoo…"):
                try:
                    _sr = yf.Search(_srch_q, max_results=8).quotes or []
                except Exception:
                    _sr = []
            st.session_state["ace_search_results"] = _sr
            # Bestes Portfolio-Match vorselektieren (Priorität: ISIN > Ticker > Name)
            _best_sym = None; _best_prio = 99
            for _r in _sr:
                _sym    = _r.get("symbol", "")
                _isin_r = _r.get("isin", "")
                _yname  = _r.get("shortname") or _r.get("longname") or ""
                _, _hit, _reason = find_in_portfolio(_sym, _isin_r, _yname)
                if _hit:
                    _prio = {"ISIN": 1, "Ticker exakt": 2}.get(_reason.split(" ")[0], 3)
                    if _prio < _best_prio:
                        _best_prio = _prio; _best_sym = _sym
            if _best_sym:
                st.session_state["ace_selected_ticker"] = _best_sym

        # ── Ergebnisse anzeigen ───────────────────────────────────────────────
        _sr_list = st.session_state.get("ace_search_results", [])
        if _sr_list:
            _opts = []; _opt_tickers = []; _opt_recommended = []
            for _r in _sr_list:
                _sym    = _r.get("symbol", "")
                _name   = (_r.get("shortname") or _r.get("longname") or "")[:38]
                _exc    = _r.get("exchange", "")
                _isin_r = _r.get("isin", "")
                _yname  = _r.get("shortname") or _r.get("longname") or ""
                _, _pf_hit, _match_reason = find_in_portfolio(_sym, _isin_r, _yname)
                if _pf_hit:
                    _badge = f"  · Im Portfolio ({_match_reason})"
                else:
                    _badge = ""
                _opts.append(f"{_sym}  —  {_name}  ({_exc}){_badge}")
                _opt_tickers.append(_sym)
                _opt_recommended.append(bool(_pf_hit))

            # Zeige Hinweis wenn ein Treffer im PF ist
            _pf_match_idx = next((i for i, r in enumerate(_opt_recommended) if r), None)
            if _pf_match_idx is not None:
                _pf_sym = _opt_tickers[_pf_match_idx]
                _pf_exc = (_sr_list[_pf_match_idx].get("exchange") or "").upper()
                st.markdown(
                    f'<div style="font-size:0.78rem;color:#00C864;opacity:0.85;'
                    f'margin-bottom:0.25rem;padding-left:2px;">'
                    f'Empfehlung: <strong>{_pf_sym}</strong> ({_pf_exc}) — '
                    f'dieser Ticker entspricht deiner Portfolio-Position.</div>',
                    unsafe_allow_html=True)

            _cur_sel = st.session_state.get("ace_selected_ticker", _opt_tickers[0] if _opt_tickers else "")
            _cur_idx = _opt_tickers.index(_cur_sel) if _cur_sel in _opt_tickers else 0
            _chosen  = st.selectbox("Treffer", _opts, index=_cur_idx,
                                     key="ace_result_select",
                                     label_visibility="collapsed")
            _chosen_ticker = _opt_tickers[_opts.index(_chosen)]
            # Namen + ISIN des gewählten Ergebnisses für späteres Portfolio-Matching merken
            _chosen_idx = _opts.index(_chosen)
            _chosen_result = _sr_list[_chosen_idx] if _chosen_idx < len(_sr_list) else {}
            st.session_state["ace_selected_name"] = (
                _chosen_result.get("shortname") or _chosen_result.get("longname") or "")
            st.session_state["ace_selected_isin"] = _chosen_result.get("isin") or ""
            if _chosen_ticker != st.session_state.get("ace_selected_ticker"):
                st.session_state["ace_selected_ticker"] = _chosen_ticker
                st.session_state.pop("ace_yf_metrics", None)

        # ── Aktiver Ticker ────────────────────────────────────────────────────
        # ace_selected_ticker ist die Single Source of Truth
        _sel_tk = st.session_state.get("ace_selected_ticker", "")
        ticker  = _sel_tk  # Default: immer aus session state
        if not _sr_list:
            _direct = st.text_input("Oder Ticker direkt eingeben",
                                     value=_sel_tk,
                                     key="ace_direct_ticker",
                                     placeholder="z.B. RGLD, ASML.AS…").strip().upper()
            # Nur überschreiben wenn User aktiv etwas anderes eingetippt hat
            if _direct and _direct != _sel_tk:
                ticker = _direct
                st.session_state["ace_selected_ticker"] = _direct
                st.session_state.pop("ace_yf_metrics",  None)
                st.session_state.pop("ace_ext_metrics", None)
            else:
                ticker = _sel_tk  # Aus Radar vorgeladen → behalten

        # ── Strategie-Auswahl ──────────────────────────────────────────────────
        _mode_opts = ["Core Asset", "Hidden Champion", "Ich weiß es noch nicht"]
        _mode_idx  = st.session_state.get("ace_mode_idx", 0)
        mode_raw   = st.selectbox("Strategie-Typ", _mode_opts, index=_mode_idx, key="ace_mode_select")
        st.session_state["ace_mode_idx"] = _mode_opts.index(mode_raw)

        if mode_raw == "Ich weiß es noch nicht":
            st.markdown(
                '<div class="ace-card" style="font-size:0.82rem;line-height:1.6;">'
                '<div style="font-weight:600;margin-bottom:0.4rem;">Core Asset vs. Hidden Champion</div>'
                '<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;">'
                '<div><div style="color:#60a5fa;font-weight:600;margin-bottom:0.25rem;">Core Asset</div>'
                'Große, etablierte Unternehmen. Planbare Erträge, breite Marktstellung, oft Dividende. '
                'Gut als stabiles Fundament im Portfolio. Beispiele: MSFT, Nestlé, Visa.</div>'
                '<div><div style="color:#f59e0b;font-weight:600;margin-bottom:0.25rem;">Hidden Champion</div>'
                'Spezialisierter Nischenwert. Oft Small/Mid-Cap, B2B-Fokus, kaum öffentlich bekannt '
                '— aber Marktführer in ihrer Nische. Hohes Re-Rating-Potenzial. '
                'Beispiele: BESI, Rational AG, Diploma.</div>'
                '</div>'
                '<div style="margin-top:0.5rem;opacity:0.6;font-size:0.77rem;">'
                'Starte die Analyse — das System empfiehlt dann automatisch den passenden Typ.</div>'
                '</div>', unsafe_allow_html=True)
            # Effektiven Modus aus gespeicherter Story-Analyse ableiten
            _si_auto = st.session_state.get("story_info")
            if _si_auto and isinstance(_si_auto, dict):
                _cf = _si_auto.get("core_fit", 5); _hf = _si_auto.get("hc_fit", 5)
                mode = "Core Asset" if _cf >= _hf else "Hidden Champion"
                st.markdown(
                    f'<div style="margin-top:0.4rem;font-size:0.8rem;'
                    f'border-left:3px solid {"#60a5fa" if mode=="Core Asset" else "#f59e0b"};'
                    f'padding-left:0.5rem;opacity:0.85;">'
                    f'System-Empfehlung: <strong>{mode}</strong> '
                    f'(Core-Fit: {_cf:.1f} · HC-Fit: {_hf:.1f})</div>',
                    unsafe_allow_html=True)
            else:
                mode = "Core Asset"  # Default bis Analyse gelaufen
        else:
            mode = mode_raw
        # Yahoo-Profil immer laden — kein sichtbarer Toggle nötig
        use_yp = True
        ypb    = {"ok": False, "profile": {}, "errors": []}
        if ticker:
            ypb = fetch_yahoo_profile(ticker)
        sug_name  = (ypb.get("profile") or {}).get("name") or ""
        asset_name = sug_name or ticker  # kein Text-Input mehr — wird still gesetzt

        # ── Aktueller Kurs ────────────────────────────────────────────────────
        if ticker:
            _px = fetch_price_now(ticker)
            if _px.get("ok"):
                _px_eur  = _px["price_eur"]
                _px_chg  = _px.get("chg_pct")
                _px_abs  = _px.get("chg_abs")
                _px_exc  = _px.get("exchange","")
                _px_fn   = _px.get("fx_note","")
                _px_up   = (_px_chg or 0) >= 0
                _px_col  = "#10b981" if _px_up else "#ef4444"
                _px_arr  = "▲" if _px_up else "▼"
                _px_sign = "+" if _px_up else ""
                _chg_str = (f"{_px_arr} {_px_sign}{_px_chg:.2f}%"
                            f"  ({_px_sign}{_px_abs:+.2f}€)" if _px_chg is not None and _px_abs is not None
                            else "")
                st.markdown(
                    f'<div style="background:var(--secondary-background-color);'
                    f'border:1px solid rgba(128,128,128,0.14);border-radius:12px;'
                    f'padding:0.75rem 1rem 0.65rem 1rem;margin:0.4rem 0 0.6rem 0;">'
                    f'<div style="font-size:0.58rem;letter-spacing:0.14em;text-transform:uppercase;'
                    f'color:var(--text-color);opacity:0.38;margin-bottom:0.3rem;">'
                    f'{ticker.upper()}'
                    + (f' · {asset_name}' if asset_name and asset_name != ticker else '')
                    + '</div>'
                    f'<div style="display:flex;align-items:baseline;gap:0.55rem;flex-wrap:wrap;">'
                    f'<span style="font-family:\'Space Grotesk\',monospace;font-size:2rem;'
                    f'font-weight:700;color:{_px_col};letter-spacing:-0.02em;line-height:1;">'
                    f'€&thinsp;{_px_eur:,.2f}</span>'
                    f'<span style="font-size:0.82rem;font-weight:600;color:{_px_col};">'
                    f'{_chg_str}</span>'
                    f'</div>'
                    + f'<div style="font-size:0.58rem;color:var(--text-color);opacity:0.32;'
                      f'margin-top:0.25rem;display:flex;align-items:center;gap:4px;">'
                      f'<span style="width:5px;height:5px;border-radius:50%;'
                      f'background:#10b981;display:inline-block;flex-shrink:0;'
                      f'box-shadow:0 0 4px #10b98188;"></span>'
                      f'Live'
                      + (f" · {_px_fn}" if _px_fn else "")
                      + (f" · {_px_exc}" if _px_exc else "")
                      + '</div>'
                    + '</div>',
                    unsafe_allow_html=True)

        # ── Firmen-Briefing (Beschreibung + Nachrichtenlage) ─────────────────
        if ticker and ypb.get("ok"):
            _prof    = ypb.get("profile", {})
            _bf_name = _prof.get("name", asset_name or ticker)
            _bf_sec  = _prof.get("sector", "")
            _bf_ind  = _prof.get("industry", "")
            _bf_api  = (st.secrets.get("OPENAI_API_KEY") or
                        os.environ.get("OPENAI_API_KEY") or "") if OPENAI_AVAILABLE else ""
            _brief   = get_company_brief_de(ticker, _bf_name, _bf_sec, _bf_ind, _bf_api)

            # News-Sentiment für Badge
            _news_bf = fetch_news(ticker, _refresh=st.session_state.get("news_rv", 0))
            _sent_bf = score_news_sentiment([n["title"] for n in _news_bf[:6]]) if _news_bf else None
            _sent_lbl = _sent_bf["label"] if _sent_bf else None
            _sent_clr = _sent_bf["color"] if _sent_bf else None

            if _brief or _sent_lbl:
                _bf_parts = []
                if _brief:
                    _bf_parts.append(
                        f'<div style="font-size:0.83rem;color:var(--text-color);'
                        f'opacity:0.75;line-height:1.6;margin-bottom:{"0.4rem" if _sent_lbl else "0"};">'
                        f'{_brief}</div>')
                if _sent_lbl:
                    _bf_parts.append(
                        f'<div style="display:flex;align-items:center;gap:0.4rem;">'
                        f'<span style="font-size:0.6rem;letter-spacing:0.1em;'
                        f'text-transform:uppercase;color:var(--text-color);opacity:0.35;">Nachrichtenlage:</span>'
                        f'<span style="font-size:0.72rem;font-weight:600;color:{_sent_clr};">'
                        f'{_sent_lbl}</span></div>')
                st.markdown(
                    '<div style="background:var(--secondary-background-color);'
                    'border:1px solid rgba(128,128,128,0.12);border-radius:10px;'
                    'padding:0.7rem 0.9rem;margin-bottom:0.5rem;">'
                    + "".join(_bf_parts) +
                    '</div>',
                    unsafe_allow_html=True)

        # Portfolio-Kontext — alle verfügbaren Keys für maximale Trefferquote
        _sel_name = st.session_state.get("ace_selected_name", "")
        _sel_isin = st.session_state.get("ace_selected_isin", "")
        # Fallback: Namen aus Yahoo-Profil wenn geladen
        if not _sel_name and ypb.get("ok"):
            _sel_name = (ypb.get("profile") or {}).get("name") or ""
        _pf_name, _pf_pos, _pf_match_reason = find_in_portfolio(ticker, _sel_isin, _sel_name)
        if _pf_pos:
            _pf_cv    = _pf_pos.get("current_value") or 0
            _pf_inv   = _pf_pos.get("invested") or 0
            _pf_pl    = _pf_cv - _pf_inv
            _pf_pct   = _pf_pos.get("perf_since_buy_pct") or 0
            _pf_col   = "#00C864" if _pf_pct >= 0 else "#FF4444"
            _pf_sign  = "+" if _pf_pct >= 0 else ""
            _pf_shares= f"{_pf_pos['shares']:.4f} Stk." if _pf_pos.get("shares") else "—"
            _pf_avg   = f"{_pf_pos['avg_price']:.2f} €" if _pf_pos.get("avg_price") else "—"
            st.markdown(
                f'<div class="ace-card" style="border-left:3px solid #00C864;margin-bottom:0.6rem;">'
                f'<div class="ace-score-lbl" style="margin-bottom:0.4rem;">Im Portfolio — {_pf_name}'
                f'{"  ·  " + _pf_match_reason if _pf_match_reason else ""}</div>'
                f'<div style="display:flex;gap:2rem;flex-wrap:wrap;align-items:baseline;">'
                f'<div><div class="ace-score-lbl">Anteile</div>'
                f'<div style="font-weight:700;font-size:1.05rem;">{_pf_shares}</div></div>'
                f'<div><div class="ace-score-lbl">Ø Kaufkurs</div>'
                f'<div style="font-weight:700;font-size:1.05rem;">{_pf_avg}</div></div>'
                f'<div><div class="ace-score-lbl">Aktueller Wert</div>'
                f'<div style="font-weight:700;font-size:1.05rem;">{_pf_cv:,.2f} €</div></div>'
                f'<div><div class="ace-score-lbl">P&amp;L</div>'
                f'<div style="font-weight:700;font-size:1.05rem;color:{_pf_col};">'
                f'{_pf_sign}{_pf_pl:,.2f} € ({_pf_sign}{_pf_pct:.1f}%)</div></div>'
                f'</div></div>', unsafe_allow_html=True)

        # Portfolio-Daten: wenn aus PF → direkt übernehmen, kein manuelles Formular
        buy_price = portfolio_total = position_value = None
        if _pf_pos:
            buy_price      = float(_pf_pos.get("avg_price")    or 0.0)
            position_value = float(_pf_pos.get("current_value")or 0.0)
            portfolio_total= float(_pf_pos.get("_depot_total") or 0.0)
            has_position   = True
        else:
            has_position = st.checkbox("Bereits im Portfolio?", value=False,
                                       key="chk_in_portfolio")
            if has_position:
                ca, cb = st.columns(2)
                with ca: buy_price      = st.number_input("Kaufkurs (€)", min_value=0.0, value=0.0, step=0.5)
                with cb: portfolio_total= st.number_input("Depotgröße (€)", min_value=0.0, value=0.0, step=100.0)
                position_value = st.number_input("Positionswert (€)", min_value=0.0, value=0.0, step=50.0)

        st.divider()
        st.markdown('<div class="ace-section">Fundamentaldaten</div>', unsafe_allow_html=True)
        fhc, fac = st.columns([0.6, 0.4])
        with fac: auto_clicked = st.button("Kennzahlen laden", key="btn_auto", use_container_width=True)

        # Persistente Cache: nur cleanen wenn Ticker wirklich geändert
        # (nicht cleanen wenn ticker leer — verhindert versehentliches Löschen)
        _cached_tk = st.session_state.get("ace_yf_ticker", "")
        if ticker and _cached_tk and _cached_tk != ticker:
            st.session_state.pop("ace_yf_metrics",  None)
            st.session_state.pop("ace_ext_metrics", None)
            st.session_state["ace_yf_ticker"] = ticker
        elif ticker and not _cached_tk:
            st.session_state["ace_yf_ticker"] = ticker

        if auto_clicked and ticker:
            with st.spinner("Lade von Yahoo…"):
                _fetched = fetch_yahoo_metrics(ticker)
                _fetched_ext = fetch_extended_metrics(ticker)
            if _fetched:
                st.session_state["ace_yf_metrics"]  = _fetched
                st.session_state["ace_ext_metrics"] = _fetched_ext
                st.session_state["ace_yf_ticker"]   = ticker
            else:
                st.warning("Yahoo liefert keine Fundamentaldaten für diesen Ticker.")

        yf_m       = st.session_state.get("ace_yf_metrics", {})
        auto_loaded = bool(yf_m) and st.session_state.get("ace_yf_ticker") == ticker
        if auto_loaded:
            cur = yf_m.get("currency", "")
            with fhc:
                st.markdown(
                    f'<div style="font-size:0.82rem;color:#00C864;padding-top:0.4rem;">'
                    f'✓ Yahoo-Daten geladen{(" · " + cur) if cur else ""}</div>',
                    unsafe_allow_html=True)

        def fv(key, fb): return fmt_v(yf_m[key]) if (auto_loaded and yf_m.get(key) is not None) else fb

        _cm1, _cm2 = st.columns(2)
        with _cm1:
            mcap   = st.text_input("Marktkapitalisierung (Mrd)",
                value=fv("mcap", ""),
                placeholder="z.B. 37,54",
                help="Gesamtwert aller ausstehenden Aktien in Milliarden €.")
        with _cm2:
            shares = st.text_input("Anzahl Aktien (Mio)",
                value=fv("shares", ""),
                placeholder="z.B. 102,37",
                help="Anzahl der insgesamt ausgegebenen Aktien in Millionen.")
        c52a, c52b = st.columns(2)
        with c52a: high52 = st.text_input("52W Hoch",
            value=fv("high52", ""),
            placeholder="z.B. 259,11",
            help="Höchster Kurs der letzten 52 Wochen (1 Jahr). "
                 "Liegt der aktuelle Kurs nahe dran, ist die Aktie auf Hochstand.")
        with c52b: low52  = st.text_input("52W Tief",
            value=fv("low52", ""),
            placeholder="z.B. 120,94",
            help="Niedrigster Kurs der letzten 52 Wochen. "
                 "Liegt der Kurs nahe am Tief, kann das eine Einstiegschance sein — "
                 "oder ein Warnsignal.")
        cfa, cfb = st.columns(2)
        with cfa:
            div_yield = st.text_input("Dividende (%)",
                value=fv("div_yield", ""),
                placeholder="z.B. 0,70",
                help="Jährliche Dividende im Verhältnis zum aktuellen Kurs in Prozent. "
                     "Höher = mehr laufende Ausschüttung.")
            pe        = st.text_input("KGV",
                value=fv("pe", ""),
                placeholder="z.B. 36,17",
                help="Kurs-Gewinn-Verhältnis: Aktienkurs ÷ Gewinn pro Aktie. "
                     "Zeigt wie viel Anleger bereit sind pro € Gewinn zu zahlen. "
                     "Niedriger KGV = günstiger bewertet (Faustregel: unter 15 günstig, "
                     "über 30 teuer — je nach Branche).")
            pb        = st.text_input("KBV",
                value=fv("pb", ""),
                placeholder="z.B. 5,10",
                help="Kurs-Buchwert-Verhältnis: Marktwert ÷ Buchwert des Eigenkapitals. "
                     "Unter 1 = Aktie günstiger als Substanzwert. "
                     "Wachstumsaktien haben oft KBV > 5.")
        with cfb:
            beta = st.text_input("Beta",
                value=fv("beta", ""),
                placeholder="z.B. 0,65",
                help="Maß für die Kursschwankung im Vergleich zum Markt. "
                     "Beta 1,0 = bewegt sich wie der Markt. "
                     "Beta 0,5 = halb so schwankungsreich. "
                     "Beta 1,5 = 50% volatiler als der Markt.")
            peg  = st.text_input("PEG",
                value=fv("peg", ""),
                placeholder="z.B. 5,37",
                help="Price/Earnings-to-Growth: KGV ÷ Gewinnwachstum. "
                     "Berücksichtigt das Wachstum — unter 1 gilt als attraktiv, "
                     "über 2 als teuer.")
            ps   = st.text_input("KUV",
                value=fv("ps", ""),
                placeholder="z.B. 20,23",
                help="Kurs-Umsatz-Verhältnis: Marktwert ÷ Jahresumsatz. "
                     "Hilfreich bei Unternehmen ohne Gewinn (z.B. Wachstumsaktien). "
                     "Unter 1 gilt als günstig.")

        metrics = {
            "mcap": to_float(mcap), "shares": to_float(shares),
            "high52": to_float(high52), "low52": to_float(low52),
            "div_yield": to_float(div_yield), "beta": to_float(beta),
            "pe": to_float(pe), "peg": to_float(peg),
            "pb": to_float(pb), "ps": to_float(ps),
        }
        # Extended metrics (FCF, Debt, Margen, Wachstum) einmischen
        _ext = st.session_state.get("ace_ext_metrics") or {}
        if _ext and st.session_state.get("ace_yf_ticker") == ticker:
            metrics.update({k: v for k, v in _ext.items() if v is not None})

        profile = {"name": asset_name or ticker, "sector": "", "industry": "",
                   "summary": "", "country": "", "currency": "", "exchange": ""}
        if ypb.get("profile"): profile = ypb["profile"]
        if asset_name: profile["name"] = asset_name

        # Yahoo-Profil Fehler nur bei Bedarf anzeigen (kein Expander)
        for _yp_err in (ypb.get("errors") or []):
            if "Dünnes Profil" not in _yp_err:  # stilles Fallback, nicht störend
                st.caption(f"⚠ {_yp_err}")

        # ── Aktuelle News ──────────────────────────────────────────────────────
        if ticker:
            st.divider()
            n_col1, n_col2, n_col3 = st.columns([2.5, 0.55, 1])
            with n_col1:
                # Header-Button: klicken = News neu laden
                if st.button("↻  Aktuelle News", key="btn_news_refresh",
                             help="News neu abrufen"):
                    st.session_state["news_rv"] = st.session_state.get("news_rv", 0) + 1
                    st.rerun()
            with n_col2:
                # Letzter Refresh-Zeitpunkt als kleiner Hinweis
                _news_rv = st.session_state.get("news_rv", 0)
                if _news_rv > 0:
                    st.markdown(
                        f'<div style="font-size:0.6rem;color:var(--text-color);opacity:0.35;'
                        f'padding-top:0.6rem;">#{_news_rv}</div>',
                        unsafe_allow_html=True)
            with n_col3:
                _api_key = (st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") or "") if OPENAI_AVAILABLE else ""
                de_toggle = st.toggle("Auf Deutsch", value=False, key="news_de",
                                      disabled=not bool(_api_key))
            news_items = fetch_news(ticker, _refresh=st.session_state.get("news_rv", 0))
            if news_items:
                titles = tuple(n["title"] for n in news_items)
                if de_toggle and _api_key:
                    with st.spinner("Übersetze…"):
                        titles = tuple(translate_headlines(titles, _api_key))

                # ── Stimmungsbild ──────────────────────────────────────────
                _sent = score_news_sentiment(list(titles))
                _gpt_summary = ""
                if _api_key:
                    _gpt_summary = gpt_news_summary(list(titles), _api_key)

                _dot_counts = (
                    f'<span style="color:#00C864;">{_sent["pos"]} positiv</span>'
                    f'<span style="opacity:0.35;margin:0 0.3rem;">·</span>'
                    f'<span style="color:#FF4444;">{_sent["neg"]} negativ</span>'
                    f'<span style="opacity:0.35;margin:0 0.3rem;">·</span>'
                    f'<span style="opacity:0.5;">{_sent["neu"]} neutral</span>'
                )
                st.markdown(
                    f'<div class="ace-card" style="border-left:3px solid {_sent["color"]};'
                    f'padding:0.75rem 1rem;margin-bottom:0.6rem;">'
                    f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:{"0.4rem" if _gpt_summary else "0"};">'
                    f'<span style="font-size:0.78rem;font-weight:600;color:{_sent["color"]};">'
                    f'{_sent["label"]}</span>'
                    f'<span style="font-size:0.75rem;opacity:0.45;">{_dot_counts}</span>'
                    f'</div>'
                    + (f'<div style="font-size:0.84rem;opacity:0.7;line-height:1.55;">{_gpt_summary}</div>' if _gpt_summary else "")
                    + '</div>', unsafe_allow_html=True)

                # ── News-Karten (nach Datum sortiert) ─────────────────────
                for i, n in enumerate(news_items):
                    title    = titles[i] if i < len(titles) else n["title"]
                    date_pub = f"{n['date']}  ·  {n['publisher']}" if n['publisher'] else n['date']
                    link_open  = f'<a href="{n["url"]}" target="_blank" style="color:inherit;text-decoration:none;">' if n["url"] else ""
                    link_close = "</a>" if n["url"] else ""
                    st.markdown(
                        f'<div class="ace-card" style="padding:0.65rem 1rem;margin-bottom:0.35rem;">'
                        f'<div style="font-size:0.72rem;color:var(--text-color);opacity:0.4;margin-bottom:0.2rem;">{date_pub}</div>'
                        f'<div style="font-size:0.92rem;line-height:1.45;">{link_open}{title}{link_close}</div>'
                        f'</div>', unsafe_allow_html=True)
            else:
                st.caption("Keine aktuellen News gefunden.")

    # ─── RIGHT: Analysis ──────────────────────────────────────────────────────
    with right:

        # ── Fundament & Business ───────────────────────────────────────────────
        st.markdown('<div class="ace-section">Fundament &amp; Business</div>', unsafe_allow_html=True)
        run_fund = st.button("▶ Starten", key="btn_fund", use_container_width=True)
        # Auto-Run wenn aus Velox Radar kommend
        if st.session_state.get("auto_run_fund") and not run_fund:
            run_fund = True
            st.session_state["auto_run_fund"] = False
        if run_fund:
            fs, fr = (score_core_fundamentals if mode == "Core Asset" else score_hc_fundamentals)(metrics)
            # Relative Bewertung zum Sektor
            _sector = profile.get("sector", "")
            _rel_delta, _rel_reasons = score_relative_valuation(metrics, _sector)
            if _rel_delta != 0 and _rel_reasons:
                fs = clip_score((fs or 5.0) + _rel_delta)
                fr = fr + _rel_reasons
            si = classify_business_profile(profile, metrics)
            ss = (si["core_fit"] if mode == "Core Asset" else si["hc_fit"]) if si else None
            sr = (si["core_reasons"] if mode == "Core Asset" else si["hc_reasons"]) if si else []
            st.session_state.fund_score  = fs; st.session_state.fund_reasons = fr
            st.session_state.story_score = ss; st.session_state.story_reasons = sr
            st.session_state.story_info  = si; st.session_state.ace_long_fazit = ""

        if st.session_state.fund_score is not None:
            _ul = st.session_state.get("user_level", "pro")
            _fund_label  = ("Unternehmensqualität"  if _ul == "beginner" else "Fundamentalscore")
            _story_label = ("Passt zum Depot?"       if _ul == "beginner" else "Business / Story-Fit")
            fc1, fc2 = st.columns(2)
            with fc1:
                render_score_card(_fund_label, st.session_state.fund_score,
                                  beginner_hint_fund(st.session_state.fund_score),
                                  st.session_state.fund_reasons, "fund", level=_ul)
            with fc2:
                if st.session_state.story_score is not None:
                    render_score_card(_story_label, st.session_state.story_score,
                                      beginner_hint_story(st.session_state.story_score),
                                      st.session_state.story_reasons, "story", level=_ul)

            if st.session_state.story_info:
                info = st.session_state.story_info
                _bm_detected = info.get("business_model","—")
                _cf_v = info.get("core_fit", 5); _hf_v = info.get("hc_fit", 5)
                _auto_type = "Core Asset" if _cf_v >= _hf_v else "Hidden Champion"
                _ul_bm = st.session_state.get("user_level", "pro")
                # Einsteiger: einfachere Begriffe
                _CHAR_DE = {
                    "skalierbar":              "Wächst ohne viel mehr Kosten",
                    "pricing power":           "Kann Preise erhöhen",
                    "Nischenstärke":           "Starke Nischenposition",
                    "Capital Light":           "Braucht wenig Kapital",
                    "wiederkehrende Einnahmen":"Planbare Einnahmen",
                    "defensiv":                "Stabil in Krisenzeiten",
                    "kapitalintensiv":         "Hoher Kapitalbedarf",
                    "Regulierung":             "Regulierter Markt",
                    "Marktführer":             "Führend im Markt",
                }
                _chars_raw = info.get("characteristics", [])
                if _ul_bm == "beginner":
                    _chars_disp = [_CHAR_DE.get(c, c) for c in _chars_raw]
                    _prof_label = "Wie verdient das Unternehmen?"
                    _type_label = "Strategie-Typ: " + (
                        "Stabiler Langzeitinvest" if _auto_type == "Core Asset"
                        else "Versteckter Marktführer")
                    _fit_label  = None  # Fit-Zahlen ausblenden
                else:
                    _chars_disp = _chars_raw
                    _prof_label = "Geschäftsprofil"
                    _type_label = f"Typ: {_auto_type}"
                    _fit_label  = f"Core-Fit {_cf_v:.1f} · HC-Fit {_hf_v:.1f}"

                _fit_span = (
                    f'<span style="font-size:0.76rem;padding:0.15rem 0.5rem;border-radius:4px;'
                    f'background:rgba(96,165,250,0.08);color:var(--text-color);opacity:0.55;">'
                    f'{_fit_label}</span>'
                ) if _fit_label else ""

                # A2/A3: Kurztext (Summary) für mehr Story
                _bm_summary = (info.get("summary") or "").strip()
                # Einsteiger: erste 2 Sätze zeigen; Pro: volle Kurzinfo
                if _ul_bm == "beginner":
                    # A2: Einfache Story — "Wie verdient das Unternehmen?"
                    # Mapping BM → verständliche Erklärung
                    _bm_story_map = {
                        "Software / SaaS":       "Das Unternehmen verkauft Software-Abonnements — Kunden zahlen monatlich oder jährlich für den Zugang.",
                        "Royalty / Streaming":   "Das Unternehmen bekommt einen Anteil an jedem Verkauf oder jeder Produktion — ohne selbst die teure Arbeit zu machen.",
                        "Nischen-B2B":           "Das Unternehmen beliefert andere Firmen mit spezialisierten Produkten oder Dienstleistungen — die Kunden kommen immer wieder.",
                        "Asset Light":           "Das Unternehmen wächst, ohne viel eigene Anlagen zu brauchen — die Gewinne bleiben deshalb besonders hoch.",
                        "Recurring Revenue":     "Das Unternehmen hat Daueraufträge oder Abonnements — die Einnahmen kommen regelmäßig und planbar.",
                        "Qualitäts-/Markenmoat": "Das Unternehmen lebt von einer starken Marke — Kunden zahlen mehr, weil sie dem Namen vertrauen.",
                        "Rohstoff-sensitiv":     "Das Unternehmen verdient mit natürlichen Rohstoffen — Gold, Öl, Chemikalien. Kurse schwanken mit dem Markt.",
                        "Zyklisch/Industrie":    "Das Unternehmen verkauft an Industrie und Wirtschaft — läuft gut wenn die Konjunktur brummt.",
                        "Defensiv":              "Das Unternehmen verkauft Dinge die Menschen immer brauchen — unabhängig davon wie gut die Wirtschaft läuft.",
                        "Reguliert/Monopol":     "Das Unternehmen operiert mit Genehmigung des Staates — Konkurrenz ist schwierig, Einnahmen sind stabil.",
                        "Kapitalintensiv":       "Das Unternehmen braucht viel Geld für Maschinen und Anlagen — dafür ist es schwer zu kopieren.",
                    }
                    _bm_story = _bm_story_map.get(_bm_detected, "")
                    if not _bm_story and _bm_summary:
                        # Fallback: erste Sentence vom Yahoo-Summary
                        _bm_story = _bm_summary.split(".")[0].strip() + "."
                    st.markdown(
                        f'<div class="ace-card">'
                        f'<div class="ace-score-lbl">{_prof_label}</div>'
                        f'<div style="font-size:1.0rem;font-weight:700;margin:0.25rem 0 0.4rem 0;'
                        f'color:var(--text-color);">{_bm_detected}</div>'
                        + (f'<div style="font-size:0.83rem;color:var(--text-color);opacity:0.62;'
                           f'line-height:1.65;margin-bottom:0.5rem;">{_bm_story}</div>'
                           if _bm_story else '')
                        + f'<div style="display:flex;gap:0.35rem;flex-wrap:wrap;margin-bottom:0.45rem;">'
                        + "".join(f'<span style="font-size:0.62rem;padding:2px 8px;border-radius:20px;'
                                  f'background:rgba(16,185,129,0.1);color:#10b981;font-weight:600;">'
                                  f'{c}</span>' for c in _chars_disp[:4])
                        + f'</div>'
                        f'<div style="margin-top:0.3rem;">'
                        f'<span style="font-size:0.72rem;padding:0.15rem 0.6rem;border-radius:4px;'
                        f'background:{"rgba(96,165,250,0.13)" if _auto_type=="Core Asset" else "rgba(245,158,11,0.13)"};'
                        f'color:{"#93c5fd" if _auto_type=="Core Asset" else "#fcd34d"};font-weight:600;">'
                        f'{_type_label}</span>'
                        f'</div>'
                        f'</div>', unsafe_allow_html=True)
                else:
                    # A3: Premium-Design für Fortgeschritten
                    _type_color = "#3b82f6" if _auto_type == "Core Asset" else "#8b5cf6"
                    _type_bg    = "rgba(59,130,246,0.1)" if _auto_type == "Core Asset" else "rgba(139,92,246,0.1)"
                    st.markdown(
                        f'<div style="background:var(--secondary-background-color);'
                        f'border:1px solid rgba(128,128,128,0.12);border-radius:14px;'
                        f'border-left:3px solid {_type_color};'
                        f'padding:1rem 1.1rem;margin-bottom:0.6rem;">'
                        # Header row
                        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;'
                        f'margin-bottom:0.5rem;">'
                        f'<div>'
                        f'<div style="font-size:0.5rem;font-weight:700;letter-spacing:0.16em;'
                        f'text-transform:uppercase;color:{_type_color};margin-bottom:0.2rem;">Geschäftsprofil</div>'
                        f'<div style="font-size:0.95rem;font-weight:700;color:var(--text-color);">'
                        f'{_bm_detected}</div>'
                        f'</div>'
                        f'<span style="font-size:0.62rem;padding:3px 10px;border-radius:20px;'
                        f'background:{_type_bg};color:{_type_color};font-weight:700;'
                        f'letter-spacing:0.06em;white-space:nowrap;">{_auto_type}</span>'
                        f'</div>'
                        # Summary snippet
                        + (f'<div style="font-size:0.78rem;color:var(--text-color);opacity:0.55;'
                           f'line-height:1.6;margin-bottom:0.55rem;'
                           f'border-left:2px solid rgba(128,128,128,0.15);padding-left:0.6rem;">'
                           f'{_bm_summary[:180] + "…" if len(_bm_summary) > 180 else _bm_summary}'
                           f'</div>' if _bm_summary else '')
                        # Characteristics pills
                        + f'<div style="display:flex;gap:0.35rem;flex-wrap:wrap;margin-bottom:0.5rem;">'
                        + "".join(f'<span style="font-size:0.6rem;padding:2px 8px;border-radius:20px;'
                                  f'background:rgba(128,128,128,0.1);color:var(--text-color);'
                                  f'opacity:0.65;font-weight:600;">{c}</span>' for c in _chars_disp)
                        + f'</div>'
                        # Fit scores
                        + (f'<div style="font-size:0.68rem;color:var(--text-color);opacity:0.4;'
                           f'border-top:1px solid rgba(128,128,128,0.1);padding-top:0.4rem;">'
                           f'{_fit_label}</div>' if _fit_label else '')
                        + f'</div>', unsafe_allow_html=True)

                if info.get("strengths") or info.get("risks"):
                    _exp_lbl = ("Was spricht dafür — und was dagegen?"
                                if _ul_bm == "beginner"
                                else "Stärken & Risiken des Geschäftsmodells")
                    with st.expander(_exp_lbl, expanded=False):
                        if _ul_bm == "beginner":
                            for x in info.get("strengths", []):
                                st.markdown(
                                    f'<div class="vx-detail-bullet">'
                                    f'<div style="color:#10b981;flex-shrink:0;">✓</div>'
                                    f'<div>{beginner_translate(x)}</div></div>',
                                    unsafe_allow_html=True)
                            for x in info.get("risks", []):
                                st.markdown(
                                    f'<div class="vx-detail-bullet">'
                                    f'<div style="color:#ef4444;flex-shrink:0;">✗</div>'
                                    f'<div>{beginner_translate(x)}</div></div>',
                                    unsafe_allow_html=True)
                        else:
                            _str_col, _risk_col = st.columns(2)
                            with _str_col:
                                for x in info.get("strengths", []):
                                    st.markdown(
                                        f'<div class="vx-detail-bullet">'
                                        f'<div style="color:#10b981;flex-shrink:0;">+</div>'
                                        f'<div style="font-size:0.84rem;">{x}</div></div>',
                                        unsafe_allow_html=True)
                            with _risk_col:
                                for x in info.get("risks", []):
                                    st.markdown(
                                        f'<div class="vx-detail-bullet">'
                                        f'<div style="color:#ef4444;flex-shrink:0;">–</div>'
                                        f'<div style="font-size:0.84rem;">{x}</div></div>',
                                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="ace-placeholder">Kennzahlen laden &amp; Starten<br>'
                        '<span style="font-size:0.78rem;opacity:0.6;">Qualität · Bewertung · Geschäftsmodell</span></div>',
                        unsafe_allow_html=True)

        st.divider()

        # ── Chart / Timing ─────────────────────────────────────────────────────
        st.markdown('<div class="ace-section">Chart · Timing</div>', unsafe_allow_html=True)
        run_chart = st.button("▶ Starten", key="btn_chart", use_container_width=True)
        if run_chart:
            out = chart_check_shortterm(ticker)
            if out is None:
                st.session_state.timing_score = None; st.session_state.timing_reasons = []
                st.session_state.chart_df = None;     st.session_state.chart_bg = []
                st.error("Keine Kursdaten. Ticker prüfen (z.B. .DE für DE-Aktien).")
            else:
                df, ts, tr, bg = out
                st.session_state.chart_df      = df
                st.session_state.timing_score  = ts
                st.session_state.timing_reasons= tr
                st.session_state.chart_bg      = bg
                st.session_state.ace_long_fazit= ""

        if st.session_state.timing_score is not None:
            _ul = st.session_state.get("user_level", "pro")
            _timing_label = ("Guter Einstiegszeitpunkt?" if _ul == "beginner" else "Timing-Score")
            render_score_card(_timing_label, st.session_state.timing_score,
                              timing_summary_text(st.session_state.timing_score,
                                                  st.session_state.timing_reasons, _ul),
                              st.session_state.timing_reasons, "timing", level=_ul)

            _cdf = st.session_state.chart_df.tail(150).copy()

            # ── Theme-aware Plotly-Farben ──────────────────────────────────────
            try:
                _ch_dark = st.get_option("theme.base") == "dark"
            except Exception:
                _ch_dark = False
            _ch_font   = "#aaaaaa"   if _ch_dark else "#666666"
            _ch_grid   = "rgba(255,255,255,0.06)" if _ch_dark else "rgba(0,0,0,0.06)"
            _ch_zerol  = "rgba(255,255,255,0.12)" if _ch_dark else "rgba(0,0,0,0.12)"
            _ch_bg     = "rgba(0,0,0,0)"
            _ch_border = "rgba(255,255,255,0.06)" if _ch_dark else "rgba(0,0,0,0.06)"

            if PLOTLY_AVAILABLE and not _cdf.empty:
                # ── Preischart: Candlestick + MAs + Volumen ────────────────────
                _has_ohlcv = all(c in _cdf.columns for c in ["Open","High","Low","Close","Volume"])
                _fig_price = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.76, 0.24], vertical_spacing=0.02
                )
                if _has_ohlcv:
                    _fig_price.add_trace(go.Candlestick(
                        x=_cdf.index,
                        open=_cdf["Open"], high=_cdf["High"],
                        low=_cdf["Low"],   close=_cdf["Close"],
                        name="Kurs",
                        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                        increasing_fillcolor="rgba(38,166,154,0.25)",
                        decreasing_fillcolor="rgba(239,83,80,0.25)",
                        line_width=1, showlegend=False
                    ), row=1, col=1)
                else:
                    # Gradient-Fill unter dem Kurs
                    _close_s = _cdf["Close"].dropna()
                    _fig_price.add_trace(go.Scatter(
                        x=_close_s.index, y=_close_s, name="Kurs",
                        line=dict(color="#26a69a", width=2),
                        fill="tozeroy",
                        fillcolor="rgba(38,166,154,0.08)",
                        showlegend=False
                    ), row=1, col=1)

                # Moving averages
                _ma_cfg = [
                    ("MA20",  "#f7c948", 1.3, "MA 20"),
                    ("MA50",  "#42a5f5", 1.3, "MA 50"),
                    ("MA200", "#ff7043", 1.3, "MA 200"),
                ]
                for _mc, _mcolor, _mw, _mlbl in _ma_cfg:
                    if _mc in _cdf.columns:
                        _ms = _cdf[_mc].dropna()
                        if not _ms.empty:
                            _fig_price.add_trace(go.Scatter(
                                x=_ms.index, y=_ms, name=_mlbl,
                                line=dict(color=_mcolor, width=_mw, dash="solid"),
                                opacity=0.8
                            ), row=1, col=1)

                # Volume bars
                if _has_ohlcv and "Volume" in _cdf.columns:
                    _vol_colors = [
                        "rgba(38,166,154,0.35)" if c >= o else "rgba(239,83,80,0.35)"
                        for c, o in zip(_cdf["Close"], _cdf["Open"])
                    ]
                    _fig_price.add_trace(go.Bar(
                        x=_cdf.index, y=_cdf["Volume"], name="Vol",
                        marker_color=_vol_colors, showlegend=False
                    ), row=2, col=1)

                _fig_price.update_layout(
                    height=300, margin=dict(l=4, r=4, t=8, b=4),
                    paper_bgcolor=_ch_bg, plot_bgcolor=_ch_bg,
                    font=dict(color=_ch_font, size=11,
                              family="-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.01,
                                xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                                font=dict(size=10), traceorder="normal"),
                    xaxis_rangeslider_visible=False,
                    xaxis2=dict(showgrid=False, zeroline=False,
                                tickfont=dict(size=9, color=_ch_font)),
                    yaxis=dict(gridcolor=_ch_grid, zeroline=False,
                               tickfont=dict(size=10, color=_ch_font),
                               showgrid=True),
                    yaxis2=dict(gridcolor=_ch_grid, zeroline=False,
                                showticklabels=False),
                    hovermode="x unified",
                    hoverlabel=dict(bgcolor="rgba(20,20,30,0.88)" if _ch_dark else "rgba(255,255,255,0.95)",
                                   font_size=11, font_color=_ch_font),
                    modebar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        color=_ch_font,
                        activecolor="#10b981",
                    ),
                )
                # ── Fibonacci Retracement — nur die 3 Schlüssel-Level ──────
                try:
                    _fib_high = float(_cdf["High"].max())
                    _fib_low  = float(_cdf["Low"].min())
                    _fib_rng  = _fib_high - _fib_low
                    if _fib_rng > 0:
                        # Nur 38.2 / 50 / 61.8 — die aussagekräftigsten Level
                        _fib_levels = [
                            (0.382, "38.2%", "rgba(52,211,153,0.45)"),
                            (0.500, "50.0%", "rgba(251,191,36,0.45)"),
                            (0.618, "61.8%", "rgba(249,115,22,0.45)"),
                        ]
                        for _fl, _flbl, _fcol in _fib_levels:
                            _fib_price = _fib_low + _fib_rng * (1 - _fl)
                            _fig_price.add_hline(
                                y=_fib_price,
                                line=dict(color=_fcol, width=0.75, dash="dot"),
                                annotation_text=f"{_flbl}  {_fib_price:.2f}",
                                annotation_position="right",
                                annotation_font_size=8,
                                annotation_font_color=_ch_font,
                                row=1, col=1
                            )
                except Exception:
                    pass

                _fig_price.update_xaxes(showgrid=False, zeroline=False,
                                        tickfont=dict(size=9, color=_ch_font))
                st.plotly_chart(_fig_price, use_container_width=True,
                                config={
                                    "displayModeBar": True,
                                    "displaylogo": False,
                                    "scrollZoom": True,
                                    "modeBarButtonsToRemove": [
                                        "select2d", "lasso2d",
                                        "hoverClosestCartesian",
                                        "hoverCompareCartesian",
                                        "toggleSpikelines",
                                    ],
                                    "toImageButtonOptions": {
                                        "format": "png",
                                        "filename": f"velox_{ticker}_chart",
                                        "scale": 2,
                                    },
                                })

                # ── RSI + MACD nebeneinander ───────────────────────────────────
                rc1, rc2 = st.columns(2)
                with rc1:
                    st.markdown(
                        '<span style="font-size:0.72rem;opacity:0.5;font-weight:600;'
                        'letter-spacing:0.08em;text-transform:uppercase;">RSI 14</span>'
                        '<span title="Relative Strength Index: misst ob die Aktie überkauft (>70) '
                        'oder überverkauft (<30) ist. 30–70 = neutraler Bereich." '
                        'style="cursor:help;font-size:0.68rem;opacity:0.35;margin-left:5px;">ⓘ</span>',
                        unsafe_allow_html=True)
                    if "RSI14" in _cdf.columns:
                        _rsi = _cdf[["RSI14"]].dropna()
                        if not _rsi.empty:
                            _fig_rsi = go.Figure()
                            _fig_rsi.add_hrect(y0=70, y1=100,
                                               fillcolor="#ef5350", opacity=0.07, line_width=0)
                            _fig_rsi.add_hrect(y0=0, y1=30,
                                               fillcolor="#26a69a", opacity=0.07, line_width=0)
                            _fig_rsi.add_hline(y=70,
                                               line=dict(color="#ef5350", width=0.8, dash="dot"))
                            _fig_rsi.add_hline(y=30,
                                               line=dict(color="#26a69a", width=0.8, dash="dot"))
                            _fig_rsi.add_trace(go.Scatter(
                                x=_rsi.index, y=_rsi["RSI14"],
                                line=dict(color="#ce93d8", width=1.8),
                                fill="tozeroy", fillcolor="rgba(206,147,216,0.06)",
                                showlegend=False
                            ))
                            _fig_rsi.update_layout(
                                height=155, margin=dict(l=4, r=4, t=4, b=4),
                                paper_bgcolor=_ch_bg, plot_bgcolor=_ch_bg,
                                font=dict(color=_ch_font, size=10),
                                yaxis=dict(range=[0,100], gridcolor=_ch_grid,
                                           zeroline=False, tickvals=[30,50,70],
                                           tickfont=dict(size=9, color=_ch_font)),
                                xaxis=dict(showgrid=False, zeroline=False,
                                           tickfont=dict(size=9, color=_ch_font)),
                                hovermode="x unified",
                                modebar=dict(bgcolor="rgba(0,0,0,0)",
                                             color=_ch_font, activecolor="#10b981"),
                            )
                            st.plotly_chart(_fig_rsi, use_container_width=True,
                                            config={"displayModeBar": False})

                with rc2:
                    st.markdown(
                        '<span style="font-size:0.72rem;opacity:0.5;font-weight:600;'
                        'letter-spacing:0.08em;text-transform:uppercase;">MACD 12/26/9</span>'
                        '<span title="Moving Average Convergence Divergence: Kreuzt die MACD-Linie '
                        'die Signallinie nach oben → bullisches Signal." '
                        'style="cursor:help;font-size:0.68rem;opacity:0.35;margin-left:5px;">ⓘ</span>',
                        unsafe_allow_html=True)
                    _macd_avail = all(c in _cdf.columns
                                      for c in ["MACD","MACD_Signal","MACD_Hist"])
                    if _macd_avail:
                        _mdf = _cdf[["MACD","MACD_Signal","MACD_Hist"]].dropna()
                        if not _mdf.empty:
                            _hist_colors = [
                                "rgba(38,166,154,0.55)" if v >= 0 else "rgba(239,83,80,0.55)"
                                for v in _mdf["MACD_Hist"]
                            ]
                            _fig_macd = go.Figure()
                            _fig_macd.add_trace(go.Bar(
                                x=_mdf.index, y=_mdf["MACD_Hist"],
                                marker_color=_hist_colors, showlegend=False
                            ))
                            _fig_macd.add_trace(go.Scatter(
                                x=_mdf.index, y=_mdf["MACD"], name="MACD",
                                line=dict(color="#42a5f5", width=1.6)
                            ))
                            _fig_macd.add_trace(go.Scatter(
                                x=_mdf.index, y=_mdf["MACD_Signal"], name="Signal",
                                line=dict(color="#ff9800", width=1.6)
                            ))
                            _fig_macd.add_hline(y=0,
                                                line=dict(color=_ch_zerol, width=0.8))
                            _fig_macd.update_layout(
                                height=155, margin=dict(l=4, r=4, t=4, b=4),
                                paper_bgcolor=_ch_bg, plot_bgcolor=_ch_bg,
                                font=dict(color=_ch_font, size=10),
                                legend=dict(orientation="h", yanchor="bottom", y=1.01,
                                            xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                                            font=dict(size=9)),
                                yaxis=dict(gridcolor=_ch_grid, zeroline=False,
                                           tickfont=dict(size=9, color=_ch_font)),
                                xaxis=dict(showgrid=False, zeroline=False,
                                           tickfont=dict(size=9, color=_ch_font)),
                                hovermode="x unified",
                                modebar=dict(bgcolor="rgba(0,0,0,0)",
                                             color=_ch_font, activecolor="#10b981"),
                            )
                            st.plotly_chart(_fig_macd, use_container_width=True,
                                            config={"displayModeBar": False})
            else:
                # Fallback: Plotly nicht verfügbar
                if not PLOTLY_AVAILABLE:
                    st.warning("Plotly nicht installiert — `pip install plotly` in der Konsole ausführen oder requirements.txt prüfen.")
                _col_close = [c for c in ["Close","MA20","MA50","MA200"] if c in _cdf.columns]
                pdf = _cdf[_col_close].dropna(how="all")
                if not pdf.empty: st.line_chart(pdf, height=200)
                rc1, rc2 = st.columns(2)
                with rc1:
                    if "RSI14" in _cdf.columns:
                        rdf = _cdf[["RSI14"]].dropna()
                        if not rdf.empty: st.line_chart(rdf, height=120)
                with rc2:
                    _mc = [c for c in ["MACD","MACD_Signal"] if c in _cdf.columns]
                    if _mc:
                        mdf = _cdf[_mc].dropna()
                        if not mdf.empty: st.line_chart(mdf, height=120)

            # ── 52W-Range als visueller Balken ────────────────────────────────
            h52v = metrics.get("high52"); l52v = metrics.get("low52")
            lp_now = current_price_from_df(st.session_state.chart_df)
            if h52v and l52v and lp_now and h52v > l52v:
                pos52v = max(0.0, min(100.0, (lp_now - l52v) / (h52v - l52v) * 100))
                _rng_color = ("#10b981" if pos52v <= 35
                              else "#f59e0b" if pos52v <= 65
                              else "#ef4444")
                st.markdown(
                    f'<div style="margin-top:0.55rem;">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:0.62rem;color:var(--text-color);opacity:0.42;margin-bottom:4px;">'
                    f'<span>52W-Tief {l52v:.2f}</span>'
                    f'<span style="font-weight:600;color:{_rng_color};">'
                    f'Jetzt bei {pos52v:.0f}% der Range</span>'
                    f'<span>52W-Hoch {h52v:.2f}</span></div>'
                    f'<div style="height:4px;background:rgba(128,128,128,0.15);'
                    f'border-radius:3px;position:relative;">'
                    f'<div style="position:absolute;left:0;top:0;height:100%;'
                    f'width:{pos52v:.1f}%;background:linear-gradient(90deg,'
                    f'{_rng_color}88,{_rng_color});border-radius:3px;"></div>'
                    f'<div style="position:absolute;top:-3px;'
                    f'left:calc({pos52v:.1f}% - 5px);width:10px;height:10px;'
                    f'background:{_rng_color};border-radius:50%;'
                    f'box-shadow:0 0 6px {_rng_color}88;"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True)

            if st.session_state.chart_bg:
                st.caption(" · ".join(st.session_state.chart_bg))

            # ── Entry-Trigger & Risiken (am Chart andocken) ───────────────────
            _lp_trig  = current_price_from_df(st.session_state.chart_df)
            _bpv_trig = float(buy_price) if has_position and buy_price and buy_price > 0 else None
            _triggers = build_entry_triggers(mode, metrics, st.session_state.timing_score,
                                             st.session_state.chart_df, has_position,
                                             _bpv_trig, _lp_trig)
            st.session_state.entry_triggers = _triggers
            _risks = build_risk_hints(mode, metrics, st.session_state.story_info,
                                      st.session_state.fund_score,
                                      st.session_state.timing_score,
                                      st.session_state.chart_df)
            st.session_state.risk_hints = _risks

            if _triggers:
                _trig_ul = st.session_state.get("user_level", "pro")
                _trig_label = ("Wann und wie kaufen?" if _trig_ul == "beginner"
                               else "Entry-Trigger — Wann und wie handeln?")
                with st.expander(_trig_label, expanded=False):
                    if _trig_ul == "beginner":
                        st.markdown(
                            '<div class="ace-hint" style="margin-bottom:0.55rem;">'
                            'Diese Hinweise sagen dir, <strong>wann</strong> ein guter Moment '
                            'zum Kaufen oder Nachkaufen wäre.</div>',
                            unsafe_allow_html=True)
                        # Trigger in Einsteiger-Sprache übersetzen
                        for _trg in _triggers:
                            _trg_plain = beginner_translate(_trg)
                            st.markdown(
                                f'<div class="{trigger_cls(_trg)}">• {_trg_plain}</div>',
                                unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '<div class="ace-hint" style="margin-bottom:0.55rem;">'
                            '<span style="color:#00C864;">●</span> Handeln &nbsp;'
                            '<span style="color:#FFA500;">●</span> Abwarten &nbsp;'
                            '<span style="color:#FF4444;">●</span> Vorsicht</div>',
                            unsafe_allow_html=True)
                        render_triggers(_triggers)

            if _risks:
                _risk_label = ("Was spricht gegen einen Einstieg?"
                               if st.session_state.get("user_level") == "beginner"
                               else "Was könnte die These zerreißen?")
                with st.expander(_risk_label, expanded=False):
                    for r in _risks:
                        st.markdown(f'<div class="trig-stop">• {r}</div>',
                                    unsafe_allow_html=True)
        else:
            st.markdown('<div class="ace-placeholder">Chart starten<br>'
                        '<span style="font-size:0.78rem;opacity:0.6;">Timing · RSI · MACD · Fibonacci · 52W</span></div>',
                        unsafe_allow_html=True)

        st.divider()

        # ── Fazit & Handlung ───────────────────────────────────────────────────
        st.markdown('<div class="ace-section">Fazit &amp; Handlung</div>', unsafe_allow_html=True)
        total = overall_score(mode, st.session_state.fund_score,
                              st.session_state.timing_score, st.session_state.story_score)

        if total is not None:
            lp  = current_price_from_df(st.session_state.chart_df)
            bpv = float(buy_price)      if has_position and buy_price      and buy_price > 0      else None
            ptv = float(portfolio_total)if has_position and portfolio_total and portfolio_total > 0 else None
            pvv = float(position_value) if has_position and position_value  and position_value > 0  else None

            fazit_md, action, why = build_fazit(
                mode, st.session_state.fund_score, st.session_state.timing_score,
                st.session_state.story_score, st.session_state.story_info, total,
                has_position, bpv, ptv, pvv, lp)
            st.session_state.last_action = action
            st.session_state.last_why    = why

            # Action Banner — Einsteiger bekommt vereinfachte Sprache
            _ul_action = st.session_state.get("user_level", "pro")
            if _ul_action == "beginner":
                _action_map = {
                    "Einstieg/Nachkauf möglich (gestaffelt).": "Interessant — Einstieg prüfen.",
                    "Einstieg möglich (gestaffelt).":          "Interessant — Einstieg prüfen.",
                    "Halten + Nachkauf möglich (gestaffelt).": "Du bist gut dabei — Nachkauf möglich.",
                    "Halten (Position läuft).":                "Aktie läuft gut — einfach halten.",
                    "Beobachten / kein Einstieg.":            "Noch abwarten — kein guter Zeitpunkt.",
                    "Eher streichen / Watchlist.":             "Kritisch — besser auf die Watchlist setzen.",
                    "Kein Einstieg.":                          "Jetzt nicht einsteigen — Vorsicht.",
                    "Kein Einstieg aktuell.":                  "Jetzt nicht einsteigen — Vorsicht.",
                }
                _display_action = _action_map.get(action, action)
                _why_simplified = [w.split("→")[-1].strip() if "→" in w else w for w in why]
            else:
                _display_action = action
                _why_simplified = why
            render_action_banner(_display_action, total, _why_simplified)

            # ── Depot-Fit Card ────────────────────────────────────────────────
            if not has_position:
                _df_lines = build_depot_fit(
                    ticker, mode, profile, total,
                    st.session_state.story_info,
                    level=st.session_state.get("user_level", "pro"))
                if _df_lines:
                    _df_html = (
                        '<div style="background:var(--secondary-background-color);'
                        'border:1px solid rgba(128,128,128,0.14);border-radius:10px;'
                        'padding:0.85rem 1.1rem;margin-bottom:0.75rem;">'
                        '<div style="font-size:0.62rem;letter-spacing:0.16em;'
                        'text-transform:uppercase;color:var(--text-color);'
                        'opacity:0.4;margin-bottom:0.55rem;">Depot-Fit</div>'
                    )
                    for icon, txt, color in _df_lines:
                        _df_html += (
                            f'<div style="display:flex;align-items:flex-start;gap:0.55rem;'
                            f'margin-bottom:0.45rem;">'
                            f'<span style="color:{color};flex-shrink:0;font-size:0.95rem;'
                            f'margin-top:1px;">{icon}</span>'
                            f'<span style="font-size:0.85rem;line-height:1.6;'
                            f'color:var(--text-color);">{txt}</span></div>'
                        )
                    _df_html += '</div>'
                    st.markdown(_df_html, unsafe_allow_html=True)

            # Depot-Fit Score berechnen
            _dfs, _dfr = calculate_depot_fit_score(ticker, mode, profile)

            # Scores kompakt — mit optionalem Depot-Fit
            _ul_fazit = st.session_state.get("user_level", "pro")
            _sp_labels = (
                ["Qualität", "Zeitpunkt", "Strategie", "Gesamt"]
                if _ul_fazit == "beginner" else
                ["Fund", "Timing", "Story", "Gesamt"]
            )
            # 5 Spalten wenn Depot-Fit vorhanden, sonst 4
            if _dfs is not None:
                sc0, sc1, sc2, sc3, sc4 = st.columns(5)
                with sc0:
                    _dfs_cls = score_color_cls(_dfs)
                    _dfs_lbl = "Depot-Fit" if _ul_fazit == "pro" else "Passt ins Depot?"
                    st.markdown(
                        f'<div class="ace-score-lbl">{_dfs_lbl}</div>'
                        f'<div class="ace-score-num {_dfs_cls}">{_dfs:.1f}</div>'
                        + (f'<div style="font-size:0.6rem;color:var(--text-color);'
                           f'opacity:0.4;margin-top:0.15rem;line-height:1.4;">'
                           f'{_dfr[:40]}</div>' if _dfr else ''),
                        unsafe_allow_html=True)
            else:
                sc1, sc2, sc3, sc4 = st.columns(4)

            score_pairs = [
                (sc1, _sp_labels[0], st.session_state.fund_score),
                (sc2, _sp_labels[1], st.session_state.timing_score),
                (sc3, _sp_labels[2], st.session_state.story_score),
                (sc4, _sp_labels[3], total),
            ]
            for col, lbl, val in score_pairs:
                with col:
                    cls = score_color_cls(val)
                    v   = f"{val:.1f}" if val else "—"
                    st.markdown(f'<div class="ace-score-lbl">{lbl}</div>'
                                f'<div class="ace-score-num {cls}">{v}</div>',
                                unsafe_allow_html=True)

            # ── Ace Fazit ─────────────────────────────────────────────────────
            st.divider()
            snap_key = str({"t": ticker, "m": mode, "fs": st.session_state.fund_score,
                            "ts": st.session_state.timing_score, "ss": st.session_state.story_score,
                            "tot": total, "act": action})
            perf2 = (lp / bpv) - 1.0 if (has_position and bpv and lp) else None
            wt2   = compute_position_weight(ptv, pvv)

            # Ace-Card Header
            st.markdown(
                '<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.55rem;">'
                '<span style="font-size:1.05rem;">✦</span>'
                '<span style="font-size:0.65rem;font-weight:700;letter-spacing:0.18em;'
                'text-transform:uppercase;color:var(--text-color);opacity:0.5;">Ace · KI-Analyse</span>'
                '</div>',
                unsafe_allow_html=True)

            _fazit_ul  = st.session_state.get("user_level", "pro")
            _fazit_btn = ("▶  Ace erklärt es dir" if _fazit_ul == "beginner"
                          else "▶  Vollständiges Fazit generieren")
            if st.button(_fazit_btn, use_container_width=True, key="btn_ace_fazit"):
                if not (st.session_state.ace_long_fazit and st.session_state.ace_long_key == snap_key):
                    _triggers_for_ace = build_entry_triggers(mode, metrics, st.session_state.timing_score,
                                                              st.session_state.chart_df, has_position, bpv, lp)
                    _risks_for_ace    = build_risk_hints(mode, metrics, st.session_state.story_info,
                                                          st.session_state.fund_score, st.session_state.timing_score,
                                                          st.session_state.chart_df)
                    snap = {
                        "asset":     {"name": asset_name or profile.get("name") or ticker, "ticker": ticker},
                        "einordnung": mode,
                        "profile":   profile if st.session_state.story_info else None,
                        "fundament": {"kennzahlen": metrics, "score": st.session_state.fund_score,
                                      "beobachtungen": st.session_state.fund_reasons},
                        "business_story": {"score": st.session_state.story_score,
                                           "beobachtungen": st.session_state.story_reasons,
                                           "profil": st.session_state.story_info} if st.session_state.story_info else None,
                        "timing":    {"score": st.session_state.timing_score,
                                      "beobachtungen": st.session_state.timing_reasons,
                                      "kontext": st.session_state.chart_bg},
                        "fazit":     {"gesamt_score": total, "handlung": action, "warum": why},
                        "entry_trigger": _triggers_for_ace, "risiken": _risks_for_ace,
                        "portfolio": {"im_portfolio": has_position, "kaufkurs": bpv, "aktueller_kurs": lp,
                                      "seit_kauf_perf": perf2, "depotgroesse": ptv,
                                      "positionswert": pvv, "positionsgewicht": wt2},
                        "red_flags": st.session_state.red_flags,
                    }
                    _ace_model = "gpt-4.1-mini"
                    try:
                        with st.spinner("Ace schreibt…"):
                            txt = ace_fazit(snap, model=_ace_model,
                                            user_level=st.session_state.get("user_level","pro"))
                        st.session_state.ace_long_fazit = txt
                        st.session_state.ace_long_key   = snap_key
                    except Exception as e:
                        st.session_state.ace_long_fazit = ""
                        st.session_state.ace_long_key   = ""
                        st.error(str(e))

            if st.session_state.ace_long_fazit:
                st.markdown(
                    f'<div class="ace-card" style="line-height:1.75;font-size:0.95rem;'
                    f'margin-top:0.6rem;border-left:3px solid rgba(128,128,128,0.2);">'
                    f'{st.session_state.ace_long_fazit}</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="font-size:0.82rem;color:var(--text-color);opacity:0.42;'
                    'line-height:1.6;margin-top:0.35rem;padding-left:0.1rem;">'
                    'Ace fasst alle Scores zusammen und gibt eine persönliche Einschätzung — '
                    'wie im Gespräch mit einem erfahrenen Investor.</div>',
                    unsafe_allow_html=True)
                if not OPENAI_AVAILABLE:
                    st.caption("⚠ openai SDK nicht installiert — in requirements.txt ergänzen.")


            # Red Flags — inline unterhalb Action Banner, keine eigene Sektion
            red_flags = build_red_flags(mode, metrics, st.session_state.timing_score,
                                        st.session_state.story_score, has_position, ptv, pvv)
            st.session_state.red_flags = red_flags
            _ul_rf = st.session_state.get("user_level", "pro")
            if red_flags:
                _rf_html = ""
                for f in red_flags:
                    if _ul_rf == "beginner":
                        # Vereinfachte Red-Flag Texte
                        _f_txt = f
                        if "Bewertungs-Flag" in f or "KGV" in f or "KBV" in f:
                            _f_txt = "Die Aktie ist aktuell hoch bewertet — prüfe, ob der Preis noch fair ist."
                        elif "Stabilitäts-Flag" in f or "Beta" in f:
                            _f_txt = "Diese Aktie schwankt stark — nichts für schwache Nerven."
                        elif "Timing-Flag" in f:
                            _f_txt = "Der Chart zeigt gerade kein sauberes Einstiegsbild — besser abwarten."
                        elif "Business-Flag" in f:
                            _f_txt = "Das Geschäftsmodell passt nicht optimal zur gewählten Strategie."
                        elif "Portfolio-Flag" in f or "Gewicht" in f:
                            _f_txt = "Diese Position ist bereits groß in deinem Depot — Risiko beachten."
                        elif "Fundamentalrisiko" in f:
                            _f_txt = "Die Zahlen des Unternehmens sind nicht stark genug für einen sicheren Einstieg."
                        _rf_html += (
                            f'<div style="display:flex;align-items:flex-start;gap:0.5rem;'
                            f'padding:0.45rem 0.7rem;margin-bottom:0.3rem;'
                            f'background:rgba(239,68,68,0.07);border-radius:7px;'
                            f'border-left:2px solid rgba(239,68,68,0.4);'
                            f'font-size:0.83rem;line-height:1.5;color:var(--text-color);">'
                            f'<span style="color:#ef4444;flex-shrink:0;margin-top:1px;">▲</span>'
                            f'<span>{_f_txt}</span></div>'
                        )
                    else:
                        _rf_html += (
                            f'<div style="display:flex;align-items:flex-start;gap:0.5rem;'
                            f'padding:0.4rem 0.7rem;margin-bottom:0.25rem;'
                            f'background:rgba(239,68,68,0.06);border-radius:7px;'
                            f'border-left:2px solid rgba(239,68,68,0.35);'
                            f'font-size:0.82rem;line-height:1.5;color:var(--text-color);">'
                            f'<span style="color:#ef4444;flex-shrink:0;margin-top:1px;">▲</span>'
                            f'<span>{f}</span></div>'
                        )
                st.markdown(_rf_html, unsafe_allow_html=True)

            # ── Watchlist speichern ────────────────────────────────────────────
            st.divider()
            _wl_h1, _wl_h2 = st.columns([5, 1])
            with _wl_h1:
                snap_note = st.text_input(
                    "Notiz zur Watchlist",
                    placeholder="z.B. 'Warte auf Q2' · 'Entry unter 130€'",
                    label_visibility="collapsed",
                    key="snap_note_input")
            with _wl_h2:
                if st.button("＋", key="btn_watchlist", use_container_width=True,
                             help="Auf Watchlist speichern"):
                    perf_v = round((lp / bpv) - 1.0, 4) if (has_position and bpv and lp) else None
                    wt_v   = compute_position_weight(ptv, pvv)
                    entry  = {
                        "ticker": ticker,
                        "name":   asset_name or profile.get("name") or ticker,
                        "mode":   mode,
                        "saved_at": datetime.now().isoformat(),
                        "fund_score":   round(st.session_state.fund_score,   2) if st.session_state.fund_score   else None,
                        "timing_score": round(st.session_state.timing_score, 2) if st.session_state.timing_score else None,
                        "story_score":  round(st.session_state.story_score,  2) if st.session_state.story_score  else None,
                        "total_score":  round(total, 2),
                        "action":   action,
                        "triggers": st.session_state.get("entry_triggers", []),
                        "risks":    st.session_state.get("risk_hints", []),
                        "metrics":  {k: v for k, v in metrics.items() if v is not None},
                        "red_flags": red_flags,
                        "perf_since_buy":  perf_v,
                        "position_weight": round(wt_v, 4) if wt_v else None,
                    }
                    save_snapshot_to_watchlist(entry, notes=snap_note)
                    st.success(f"✓ {ticker} auf der Watchlist gespeichert.")
                    st.rerun()

        else:
            st.markdown('<div class="ace-placeholder">Fundament und/oder Chart starten,<br>um Fazit &amp; Handlung zu berechnen.</div>',
                        unsafe_allow_html=True)

        # ── Velox Radar Teaser ─────────────────────────────────────────────────
        if st.session_state.story_info:
            _radar_open = st.session_state.get("show_radar", False)
            st.markdown('<div style="margin-top:1.2rem;"></div>', unsafe_allow_html=True)
            if _radar_open:
                # Schließen-Button — dezent
                _rc1, _rc2, _rc3 = st.columns([1, 3, 1])
                with _rc2:
                    if st.button("▲ Velox Radar schließen", key="btn_radar_toggle",
                                 use_container_width=True):
                        st.session_state["show_radar"] = False
                        st.rerun()
            else:
                # Premium Teaser Card — ein einziges Element, kein gestapelter Button
                st.markdown(
                    '<div style="'
                    'background:linear-gradient(135deg,'
                    'rgba(245,158,11,0.10) 0%,rgba(245,158,11,0.04) 60%,rgba(251,191,36,0.07) 100%);'
                    'border:1px solid rgba(245,158,11,0.30);'
                    'border-radius:14px;padding:1rem 1.2rem;'
                    'position:relative;overflow:hidden;cursor:pointer;'
                    'transition:border-color 0.2s,box-shadow 0.2s;">'
                    # Subtiler Glanz
                    '<div style="position:absolute;top:-40%;right:-10%;width:55%;height:180%;'
                    'background:radial-gradient(ellipse,rgba(251,191,36,0.12) 0%,transparent 65%);'
                    'pointer-events:none;"></div>'
                    # Icon + Label
                    '<div style="display:flex;align-items:center;'
                    'justify-content:space-between;margin-bottom:0.5rem;">'
                    '<div style="display:flex;align-items:center;gap:0.5rem;">'
                    '<span style="font-size:1.1rem;line-height:1;">◎</span>'
                    '<span style="font-size:0.68rem;font-weight:800;letter-spacing:0.18em;'
                    'text-transform:uppercase;color:#f59e0b;">Velox Radar</span>'
                    '</div>'
                    '<span style="font-size:0.9rem;color:#f59e0b;opacity:0.6;">→</span>'
                    '</div>'
                    # Beschreibung
                    '<div style="font-size:0.84rem;line-height:1.55;'
                    'color:var(--text-color);opacity:0.65;">'
                    'Vergleichbare Aktien &amp; Entdeckungsideen — '
                    'mit echten Velox-Scores vorberechnet.'
                    '</div>'
                    # Mini-Tags
                    '<div style="display:flex;gap:0.4rem;flex-wrap:wrap;margin-top:0.6rem;">'
                    '<span style="font-size:0.6rem;padding:2px 8px;border-radius:20px;'
                    'background:rgba(245,158,11,0.12);color:#f59e0b;'
                    'border:1px solid rgba(245,158,11,0.25);">Vergleichswerte</span>'
                    '<span style="font-size:0.6rem;padding:2px 8px;border-radius:20px;'
                    'background:rgba(245,158,11,0.12);color:#f59e0b;'
                    'border:1px solid rgba(245,158,11,0.25);">Entdeckungsideen</span>'
                    '<span style="font-size:0.6rem;padding:2px 8px;border-radius:20px;'
                    'background:rgba(245,158,11,0.12);color:#f59e0b;'
                    'border:1px solid rgba(245,158,11,0.25);">Live Scores</span>'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True)
                # Unsichtbarer voller Streamlit-Button über der Card
                if st.button("◎  Velox Radar öffnen", key="btn_radar_toggle",
                             use_container_width=True, type="secondary"):
                    st.session_state["show_radar"] = True
                    st.rerun()

    # ── Velox Radar Cards — volle Breite unterhalb der Analyse ────────────────
    if st.session_state.get("show_radar") and st.session_state.get("story_info"):
        _vr_info   = st.session_state.story_info
        _vr_bm     = _vr_info.get("business_model", "—")
        _vr_tags   = _vr_info.get("tags", [])
        _vr_cf     = _vr_info.get("core_fit", 5)
        _vr_hf     = _vr_info.get("hc_fit",  5)
        _vr_dtype  = "Core Asset" if _vr_cf >= _vr_hf else "Hidden Champion"
        _vr_mode   = st.session_state.get("ace_mode_select", _vr_dtype)
        if _vr_mode not in ("Core Asset", "Hidden Champion"): _vr_mode = _vr_dtype
        _vr_ticker = st.session_state.get("ace_selected_ticker", "")

        # Theme-Erkennung für Card-Farben
        # st.get_option gibt None für Default-Theme (= light) zurück
        try:
            _vr_dark = st.get_option("theme.base") == "dark"
        except Exception:
            _vr_dark = False

        st.divider()

        # ── Header ─────────────────────────────────────────────────────────────
        _hdr_c1, _hdr_c2 = st.columns([3, 1])
        with _hdr_c1:
            st.markdown(
                '<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700;800&display=swap" rel="stylesheet">'
                '<div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:0.2rem;">'
                '<div style="width:4px;height:2rem;border-radius:3px;flex-shrink:0;'
                'background:linear-gradient(180deg,#f59e0b,#fbbf24);"></div>'
                '<div style="font-family:\'Space Grotesk\',sans-serif;font-size:1.5rem;'
                'font-weight:800;letter-spacing:-0.01em;color:#f59e0b;">Velox Radar</div>'
                '</div>',
                unsafe_allow_html=True)
            st.caption(f"**{_vr_ticker}** · {_vr_bm} · {_vr_mode} — Scores via Yahoo Finance")
        with _hdr_c2:
            if st.button("✕ Schließen", key="btn_radar_close2", use_container_width=True):
                st.session_state["show_radar"] = False
                st.rerun()

        # ── Vergleichbare Werte ────────────────────────────────────────────────
        _vr_sl    = st.session_state.get("vr_sim_limit", 3)
        _vr_similar = get_similar_stocks(_vr_bm, _vr_mode, limit=_vr_sl)
        _max_sim  = len(get_similar_stocks(_vr_bm, _vr_mode, limit=99))

        if _vr_similar:
            _vr_hc    = _vr_mode == "Hidden Champion"
            _vr_badge = "#fcd34d" if _vr_hc else "#93c5fd"
            _vr_bdgbg = "rgba(245,158,11,0.15)" if _vr_hc else "rgba(96,165,250,0.15)"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.5rem;'
                f'margin:1.4rem 0 0.75rem 0;">'
                f'<span style="font-size:0.78rem;font-weight:700;letter-spacing:0.06em;'
                f'opacity:0.45;text-transform:uppercase;">Vergleichbare Werte</span>'
                f'<span style="font-size:0.67rem;padding:0.1rem 0.45rem;border-radius:4px;'
                f'background:{_vr_bdgbg};color:{_vr_badge};">'
                f'{"Hidden Champion" if _vr_hc else "Core Asset"}</span>'
                f'</div>',
                unsafe_allow_html=True)
            _sim_cols = st.columns(min(len(_vr_similar), 3), gap="medium")
            for _vsi, (tk, name, why) in enumerate(_vr_similar):
                with _sim_cols[_vsi % 3]:
                    with st.spinner(f"Lade {tk}…"):
                        _vr_data = _card_full_data(tk, _vr_mode)
                    render_radar_card(tk, name, why, _vr_data, f"s{_vsi}", _vr_mode, _vr_dark)
            if _vr_sl < _max_sim:
                _mc1, _mc2, _mc3 = st.columns([1, 2, 1])
                with _mc2:
                    if st.button("+ Weitere Vergleichswerte", key="btn_more_sim",
                                 use_container_width=True):
                        st.session_state["vr_sim_limit"] = _vr_sl + 3
                        st.rerun()

        # ── Entdeckungsideen ───────────────────────────────────────────────────
        _vr_rl    = st.session_state.get("vr_rad_limit", 3)
        _vr_radar = get_radar_stocks(_vr_tags, _vr_ticker, limit=_vr_rl)
        _max_rad  = len(get_radar_stocks(_vr_tags, _vr_ticker, limit=99))

        if _vr_radar:
            st.markdown(
                '<div style="display:flex;align-items:center;gap:0.5rem;'
                'margin:1.8rem 0 0.75rem 0;">'
                '<div style="width:3px;height:1rem;border-radius:2px;flex-shrink:0;'
                'background:linear-gradient(180deg,#f59e0b,#fbbf24);"></div>'
                '<span style="font-size:0.78rem;font-weight:700;letter-spacing:0.06em;'
                'color:#f59e0b;text-transform:uppercase;opacity:0.9;">Entdeckungsideen</span>'
                '<span style="font-size:0.67rem;opacity:0.35;margin-left:0.25rem;">'
                '— nicht-offensichtlich, basierend auf deinem Profil</span>'
                '</div>',
                unsafe_allow_html=True)
            _rad_cols = st.columns(min(len(_vr_radar), 3), gap="medium")
            for _vri, (tk, name, why) in enumerate(_vr_radar):
                with _rad_cols[_vri % 3]:
                    with st.spinner(f"Lade {tk}…"):
                        _vr_data = _card_full_data(tk, _vr_mode)
                    render_radar_card(tk, name, why, _vr_data, f"r{_vri}", _vr_mode, _vr_dark)
            if _vr_rl < _max_rad:
                _rc1, _rc2, _rc3 = st.columns([1, 2, 1])
                with _rc2:
                    if st.button("+ Weitere Entdeckungsideen", key="btn_more_rad",
                                 use_container_width=True):
                        st.session_state["vr_rad_limit"] = _vr_rl + 3
                        st.rerun()

        st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Watchlist
# ──────────────────────────────────────────────────────────────────────────────
with tab_watchlist:
    st.session_state["_active_tab"] = "Watchlist"
    wl = load_watchlist()

    # ── Sprung aus Watchlist in Analyse-Tab ───────────────────────────────────
    if st.session_state.get("wl_jump_to_analyse"):
        _jt = st.session_state.pop("wl_jump_to_analyse")
        st.info(f"**{_jt}** wurde geladen → wechsle zum **Analyse**-Tab oben.")

    if not wl:
        st.markdown(
            '<div style="text-align:center;padding:3rem 1rem;">'
            '<div style="font-size:2rem;margin-bottom:0.8rem;"></div>'
            '<div style="font-size:1.1rem;font-weight:600;margin-bottom:0.4rem;">'
            'Deine Watchlist ist leer</div>'
            '<div style="font-size:0.88rem;color:var(--text-color);opacity:0.5;line-height:1.6;">'
            'Analysiere eine Aktie im Analyse-Tab und speichere sie mit dem '
            '<strong>＋</strong>-Button auf die Watchlist.</div>'
            '</div>',
            unsafe_allow_html=True)
    else:
        # ── Header ───────────────────────────────────────────────────────────
        st.markdown(
            f'<div style="font-size:0.65rem;letter-spacing:0.16em;text-transform:uppercase;'
            f'color:var(--text-color);opacity:0.4;margin-bottom:0.8rem;">'
            f'{len(wl)} Aktie{"n" if len(wl)!=1 else ""} auf der Watchlist</div>',
            unsafe_allow_html=True)

        _wl_open = st.session_state.get("wl_open_key", None)

        # ── Filter & Sort ─────────────────────────────────────────────────────
        _fh1, _fh2, _fh3 = st.columns([2, 2, 2])
        with _fh1:
            _wl_mode_filter = st.selectbox(
                "Modus", ["Alle", "Core Asset", "Hidden Champion"],
                key="wl_filter_mode", label_visibility="collapsed")
        with _fh2:
            _wl_sort = st.selectbox(
                "Sortierung", ["Score ↓", "Score ↑", "Neueste zuerst", "Name A–Z"],
                key="wl_sort", label_visibility="collapsed")
        with _fh3:
            st.markdown(
                f'<div style="font-size:0.65rem;color:var(--text-color);opacity:0.35;'
                f'padding-top:0.65rem;text-align:right;">'
                f'{len(wl)} Aktie{"n" if len(wl)!=1 else ""}</div>',
                unsafe_allow_html=True)

        # Filtern
        wl_filtered = [it for it in wl
                       if _wl_mode_filter == "Alle" or it.get("mode","") == _wl_mode_filter]
        # Sortieren
        def _wl_sort_key(it):
            ls = (it.get("snapshots") or [{}])[-1]
            if _wl_sort == "Score ↓":   return -(ls.get("total_score") or 0)
            if _wl_sort == "Score ↑":   return  (ls.get("total_score") or 0)
            if _wl_sort == "Name A–Z":  return  (it.get("name") or it.get("ticker","")).lower()
            return 0  # Neueste zuerst — reverse beim Rendern
        if _wl_sort == "Neueste zuerst":
            wl_display = list(reversed(wl_filtered))
        else:
            wl_display = sorted(wl_filtered, key=_wl_sort_key)

        if not wl_display:
            st.caption("Keine Einträge für diesen Filter.")

        # ── Kachel-Grid (2 Spalten) — Detail in derselben Spalte ──────────────
        wl_rev = wl_display
        for _row_i in range(0, len(wl_rev), 2):
            _row_items = wl_rev[_row_i:_row_i+2]
            _cols = st.columns(2)

            for _ci, item in enumerate(_row_items):
                col = _cols[_ci]
                tk   = item.get("ticker", "?")
                mode = item.get("mode", "Core Asset")
                name = item.get("name", tk)
                snaps = item.get("snapshots", [])
                ls    = snaps[-1] if snaps else {}
                prev  = snaps[-2] if len(snaps) >= 2 else None
                _key  = f"{tk}_{mode}"
                _is_open = (_wl_open == _key)

                total_s = ls.get("total_score")
                note    = item.get("notes", "")
                qk      = st.session_state.get(f"wl_qk_{_key}", {})

                _sc  = ("#10b981" if (total_s or 0) >= 6.5
                        else "#f59e0b" if (total_s or 0) >= 5.0
                        else "#ef4444" if total_s else "rgba(128,128,128,0.35)")
                _border = f"2px solid {_sc}" if _is_open else "1px solid rgba(128,128,128,0.14)"
                _shadow = f"0 0 0 2px {_sc}22" if _is_open else "none"

                px      = fetch_price_now(tk)
                _px_ok  = px.get("ok")
                _px_eur = px.get("price_eur", 0)
                _px_chg = px.get("chg_pct")
                _px_up  = (_px_chg or 0) >= 0
                _px_col = "#10b981" if _px_up else "#ef4444"
                _px_arr = "▲" if _px_up else "▼"

                _trend_str = ""
                if prev and total_s and prev.get("total_score"):
                    _d = total_s - prev["total_score"]
                    _trend_str = (f'<span style="color:#10b981;font-size:0.68rem;font-weight:600;">↑ +{_d:.1f}</span>'
                                  if _d > 0.05 else
                                  f'<span style="color:#ef4444;font-size:0.68rem;font-weight:600;">↓ {_d:.1f}</span>'
                                  if _d < -0.05 else
                                  '<span style="font-size:0.68rem;opacity:0.35;">→</span>')

                _mode_c  = "#3b82f6" if "Core" in mode else "#8b5cf6"
                _mode_bg = "rgba(59,130,246,0.09)" if "Core" in mode else "rgba(139,92,246,0.09)"
                _mo_lbl  = "Core" if "Core" in mode else "HC"
                _bar_pct = int((total_s or 0) * 10)

                with col:
                    # W2: Kurzbeschreibung laden (gecacht)
                    _wl_desc = fetch_short_desc(tk)
                    # ── Kompakte Kachel ───────────────────────────────────────
                    st.markdown(
                        f'<div style="background:var(--secondary-background-color);'
                        f'border:{_border};border-radius:14px;'
                        f'box-shadow:{_shadow};'
                        f'padding:0.9rem 1rem 0.75rem 1rem;'
                        f'position:relative;overflow:hidden;">'
                        f'<div style="position:absolute;top:0;left:0;right:0;height:2.5px;'
                        f'background:{_sc};border-radius:14px 14px 0 0;"></div>'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;'
                        f'margin-top:0.1rem;margin-bottom:0.4rem;">'
                        f'<div style="display:flex;align-items:center;gap:0.4rem;">'
                        f'<span style="font-size:0.45rem;font-weight:800;letter-spacing:0.14em;'
                        f'text-transform:uppercase;color:{_mode_c};background:{_mode_bg};'
                        f'border-radius:20px;padding:2px 6px;">{_mo_lbl}</span>'
                        f'<span style="font-size:1.05rem;font-weight:800;color:var(--text-color);'
                        f'letter-spacing:0.04em;">{tk}</span>'
                        f'</div>'
                        + (f'<div style="text-align:right;">'
                           f'<div style="font-size:0.92rem;font-weight:700;color:{_px_col};">€{_px_eur:,.0f}</div>'
                           f'<div style="font-size:0.6rem;color:{_px_col};opacity:0.8;">'
                           f'{_px_arr}{abs(_px_chg or 0):.1f}%</div></div>' if _px_ok else '')
                        + f'</div>'
                        f'<div style="font-size:0.78rem;font-weight:600;color:var(--text-color);'
                        f'margin-bottom:0.12rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
                        f'{name[:32] + "…" if len(name) > 32 else name}</div>'
                        + (f'<div style="font-size:0.63rem;color:var(--text-color);opacity:0.38;'
                           f'margin-bottom:0.45rem;line-height:1.45;'
                           f'overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;">'
                           f'{_wl_desc}</div>' if _wl_desc else '<div style="margin-bottom:0.45rem;"></div>')
                        + f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.25rem;">'
                        f'<span style="font-size:1.35rem;font-weight:800;color:{_sc};line-height:1;">'
                        f'{f"{total_s:.1f}" if total_s else "—"}</span>'
                        f'<div style="flex:1;">'
                        f'<div style="height:3px;background:rgba(128,128,128,0.12);border-radius:2px;">'
                        f'<div style="width:{_bar_pct}%;height:100%;background:{_sc};border-radius:2px;"></div>'
                        f'</div></div>'
                        f'{_trend_str}'
                        f'</div>'
                        + (f'<div style="font-size:0.65rem;color:var(--text-color);opacity:0.38;'
                           f'font-style:italic;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
                           f'· {note[:35]}{"…" if len(note) > 35 else ""}</div>' if note else '')
                        + f'</div>',
                        unsafe_allow_html=True)

                    _btn_lbl = "▲ Schließen" if _is_open else "✎ Eintrag bearbeiten"
                    if st.button(_btn_lbl, key=f"wl_tog_{_key}",
                                 use_container_width=True):
                        st.session_state["wl_open_key"] = None if _is_open else _key
                        st.rerun()

                    # ── Detail-Panel — in derselben Spalte ───────────────────
                    if _is_open:
                        _dqk = st.session_state.get(f"wl_qk_{_key}", {})

                        st.markdown(
                            f'<div style="background:var(--secondary-background-color);'
                            f'border:1px solid {_sc}33;border-radius:10px;'
                            f'padding:0.9rem 1rem;margin-top:0.4rem;">'
                            f'<div style="font-size:0.58rem;letter-spacing:0.14em;text-transform:uppercase;'
                            f'color:{_sc};margin-bottom:0.6rem;font-weight:700;">'
                            f'▸ {tk} · Details</div>',
                            unsafe_allow_html=True)

                        # Sub-Scores
                        _sd1, _sd2 = st.columns(2)
                        _sd3, _sd4 = st.columns(2)
                        for _sc2, _sl, _sv in [
                            (_sd1, "Qualität",  ls.get("fund_score")),
                            (_sd2, "Timing",    ls.get("timing_score")),
                            (_sd3, "Strategie", ls.get("story_score")),
                            (_sd4, "Gesamt",    ls.get("total_score")),
                        ]:
                            _vc = ("#10b981" if (_sv or 0) >= 6.5
                                   else "#f59e0b" if (_sv or 0) >= 5.0
                                   else "#ef4444" if _sv else "rgba(128,128,128,0.3)")
                            with _sc2:
                                st.markdown(
                                    f'<div style="font-size:0.56rem;text-transform:uppercase;'
                                    f'letter-spacing:0.1em;color:var(--text-color);opacity:0.38;">{_sl}</div>'
                                    f'<div style="font-size:1.2rem;font-weight:800;color:{_vc};margin-bottom:0.4rem;">'
                                    f'{f"{_sv:.1f}" if _sv else "—"}</div>',
                                    unsafe_allow_html=True)

                        if ls.get("action"):
                            st.markdown(
                                f'<div style="font-size:0.78rem;color:var(--text-color);'
                                f'opacity:0.55;margin-bottom:0.4rem;">→ {ls["action"]}</div>',
                                unsafe_allow_html=True)

                        # W3: Depot-Fit Badge
                        _wl_sector = fetch_sector_cached(tk)
                        _wl_dfs, _wl_dfr = calculate_depot_fit_score(
                            tk, mode, {"sector": _wl_sector})
                        if _wl_dfs is not None:
                            _wl_dfc = ("#10b981" if _wl_dfs >= 7 else
                                       "#f59e0b" if _wl_dfs >= 5 else "#ef4444")
                            st.markdown(
                                f'<div style="font-size:0.68rem;padding:0.28rem 0.6rem;'
                                f'background:{_wl_dfc}14;border-radius:7px;'
                                f'border-left:2px solid {_wl_dfc};margin-bottom:0.4rem;'
                                f'display:flex;align-items:center;gap:0.4rem;">'
                                f'<span style="font-size:0.55rem;font-weight:700;letter-spacing:0.1em;'
                                f'text-transform:uppercase;color:{_wl_dfc};">Depot-Fit</span>'
                                f'<span style="font-weight:800;color:{_wl_dfc};">{_wl_dfs:.1f}</span>'
                                + (f'<span style="color:var(--text-color);opacity:0.45;">'
                                   f'· {_wl_dfr}</span>' if _wl_dfr else '')
                                + '</div>',
                                unsafe_allow_html=True)

                        if _dqk.get("ok"):
                            _qc = ("#10b981" if (_dqk.get("fund") or 0) >= 6.5 else
                                   "#f59e0b" if (_dqk.get("fund") or 0) >= 5.0 else "#ef4444")
                            st.markdown(
                                f'<div style="font-size:0.72rem;padding:0.3rem 0.55rem;'
                                f'background:rgba(16,185,129,0.07);border-radius:7px;'
                                f'border-left:2px solid {_qc};margin-bottom:0.4rem;">'
                                + (f'↻ Kurzcheck {_dqk["ts"]}: <strong style="color:{_qc};">{_dqk["fund"]:.1f}</strong>' if _dqk.get("fund") else '↻ Kurzcheck: Daten nicht verfügbar')
                                + '</div>',
                                unsafe_allow_html=True)

                        st.markdown('</div>', unsafe_allow_html=True)

                        # Aktionen
                        _a1, _a2 = st.columns(2)
                        _a3, _a4 = st.columns(2)
                        with _a1:
                            if st.button("↻ Kurzcheck", key=f"wl_chk_{_key}", use_container_width=True):
                                with st.spinner("Lädt…"):
                                    st.session_state[f"wl_qk_{_key}"] = watchlist_quick_check(tk, mode)
                                st.rerun()
                        with _a2:
                            if st.button("▶ Vollanalyse", key=f"wl_anal_{_key}", use_container_width=True):
                                st.session_state["ace_selected_ticker"] = tk
                                st.session_state["ace_search_q"]        = name
                                st.session_state["wl_jump_to_analyse"]  = name or tk
                                st.session_state["_auto_switch_to_analyse"] = True
                                st.session_state["wl_open_key"]         = None
                                # Merken: nach Analyse Snapshot in Watchlist speichern
                                st.session_state["wl_analyse_save_back"] = {
                                    "ticker": tk, "mode": mode, "name": name}
                                for _k2 in ("fund_score","timing_score","story_score","story_info",
                                           "chart_df","ace_long_fazit","ace_yf_metrics","ace_ext_metrics",
                                           "ace_direct_ticker","ace_search_input"):
                                    st.session_state.pop(_k2, None)
                                st.session_state["auto_run_fund"] = True
                                st.rerun()
                        with _a3:
                            if st.button("✓ Gekauft", key=f"wl_buy_{_key}", use_container_width=True):
                                _bk = f"wl_kaufen_{_key}"
                                st.session_state[_bk] = not st.session_state.get(_bk, False)
                                st.rerun()
                        with _a4:
                            if st.button("✕ Entfernen", key=f"wl_rm_{_key}", use_container_width=True):
                                remove_from_watchlist(tk, mode)
                                st.session_state["wl_open_key"] = None
                                st.session_state.pop(f"wl_qk_{_key}", None)
                                st.rerun()

                        # Gekauft-Form
                        if st.session_state.get(f"wl_kaufen_{_key}"):
                            _gk1, _gk2 = st.columns(2)
                            with _gk1:
                                _buy_shares = st.number_input("Anteile", min_value=0.0001,
                                                               step=1.0, value=1.0,
                                                               key=f"wl_sh_{_key}")
                            with _gk2:
                                _buy_price = st.number_input("Kaufkurs €", min_value=0.01,
                                                              value=float(f"{_px_eur:.2f}") if _px_ok else 100.0,
                                                              key=f"wl_pr_{_key}")
                            _buy_pname = st.selectbox("Portfolio", PORTFOLIO_NAMES, key=f"wl_pn_{_key}")
                            if st.button("✓ Jetzt ins Portfolio", key=f"wl_dosave_{_key}",
                                         use_container_width=True):
                                _port = load_portfolio()
                                _port[_buy_pname]["positions"].append({
                                    "ticker": tk, "name": name,
                                    "shares": _buy_shares, "avg_price": _buy_price,
                                    "invested": round(_buy_shares * _buy_price, 2),
                                    "current_value": round(_buy_shares * (_px_eur if _px_ok else _buy_price), 2),
                                    "current_price": _px_eur if _px_ok else _buy_price,
                                    "notes": f"Aus Watchlist · {mode}",
                                })
                                save_portfolio(_port)
                                st.session_state.pop(f"wl_kaufen_{_key}", None)
                                st.session_state["wl_open_key"] = None
                                st.success(f"✓ {tk} gespeichert.")
                                st.rerun()

                        # ── Velox Radar Brand-Button ─────────────────────────
                        st.markdown(
                            '<div style="margin-top:0.6rem;'
                            'background:linear-gradient(135deg,'
                            'rgba(245,158,11,0.10) 0%,rgba(245,158,11,0.04) 100%);'
                            'border:1px solid rgba(245,158,11,0.32);'
                            'border-radius:10px;padding:0.7rem 0.9rem;'
                            'position:relative;overflow:hidden;">'
                            '<div style="position:absolute;top:-30%;right:-5%;width:40%;height:160%;'
                            'background:radial-gradient(ellipse,rgba(251,191,36,0.10) 0%,transparent 70%);'
                            'pointer-events:none;"></div>'
                            '<div style="display:flex;justify-content:space-between;align-items:center;">'
                            '<div>'
                            '<div style="display:flex;align-items:center;gap:0.35rem;margin-bottom:0.2rem;">'
                            '<span style="font-size:0.8rem;color:#f59e0b;">◎</span>'
                            '<span style="font-size:0.6rem;font-weight:800;letter-spacing:0.18em;'
                            'text-transform:uppercase;color:#f59e0b;">Velox Radar</span>'
                            '</div>'
                            '<div style="font-size:0.72rem;color:var(--text-color);opacity:0.5;">'
                            'Ähnliche Aktien entdecken</div>'
                            '</div>'
                            '<span style="color:#f59e0b;font-size:0.9rem;opacity:0.7;">→</span>'
                            '</div></div>',
                            unsafe_allow_html=True)
                        if st.button("◎ Velox Radar öffnen",
                                     key=f"wl_radar_{_key}",
                                     use_container_width=True):
                            st.session_state["radar_from_ticker"]       = tk
                            st.session_state["radar_from_name"]         = name
                            st.session_state["radar_mode"]              = mode
                            st.session_state["_auto_switch_to_radar"]   = True
                            st.session_state["wl_open_key"]             = None
                            st.rerun()

                        # ── Snapshot-Verlauf ──────────────────────────────────
                        if len(snaps) >= 2:
                            with st.expander(f"Score-Verlauf ({len(snaps)} Snapshots)",
                                             expanded=False):
                                # Mini-Sparkline als Text-Visualisierung
                                _spark_items = []
                                for _s in snaps[-6:]:  # max 6 anzeigen
                                    _sd = (_s.get("saved_at") or "")[:10]
                                    _st = _s.get("total_score")
                                    _sa = (_s.get("action") or "")[:30]
                                    _sc3 = ("#10b981" if (_st or 0) >= 6.5
                                            else "#f59e0b" if (_st or 0) >= 5.0
                                            else "#ef4444" if _st else "#888")
                                    _spark_items.append(
                                        f'<div style="display:flex;align-items:center;'
                                        f'gap:0.5rem;padding:0.3rem 0;'
                                        f'border-bottom:1px solid rgba(128,128,128,0.07);">'
                                        f'<span style="font-size:0.65rem;color:var(--text-color);'
                                        f'opacity:0.38;min-width:70px;">{_sd}</span>'
                                        f'<span style="font-size:0.9rem;font-weight:700;'
                                        f'color:{_sc3};min-width:32px;">'
                                        f'{f"{_st:.1f}" if _st else "—"}</span>'
                                        f'<span style="font-size:0.68rem;color:var(--text-color);'
                                        f'opacity:0.45;">{_sa}</span>'
                                        f'</div>'
                                    )
                                st.markdown(
                                    '<div style="font-size:0.72rem;">'
                                    + "".join(_spark_items)
                                    + '</div>',
                                    unsafe_allow_html=True)

                        # ── Notiz ─────────────────────────────────────────────
                        with st.expander("Notiz", expanded=False):
                            _cn = item.get("notes", "")
                            _nn = st.text_area("Notiz", value=_cn, key=f"note_{_key}",
                                               height=55, label_visibility="collapsed",
                                               placeholder="'Warte auf Q2' · 'Entry unter 130€'")
                            if st.button("Speichern", key=f"sn_{_key}", use_container_width=True):
                                update_watchlist_notes(tk, mode, _nn)
                                st.success("✓"); st.rerun()


        # ── Velox Radar Teaser ─────────────────────────────────────────────────
        st.divider()
        st.markdown(
            '<div style="'
            'background:linear-gradient(135deg,'
            'rgba(245,158,11,0.09) 0%,rgba(245,158,11,0.03) 100%);'
            'border:1px solid rgba(245,158,11,0.28);'
            'border-radius:14px;padding:1.1rem 1.3rem;">'
            '<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.45rem;">'
            '<span style="font-size:1rem;">◎</span>'
            '<span style="font-size:0.65rem;font-weight:800;letter-spacing:0.18em;'
            'text-transform:uppercase;color:#f59e0b;">Velox Radar</span>'
            '<span style="font-size:0.62rem;padding:2px 8px;border-radius:20px;'
            'background:rgba(245,158,11,0.15);color:#f59e0b;'
            'border:1px solid rgba(245,158,11,0.3);">Premium</span>'
            '</div>'
            '<div style="font-size:0.85rem;color:var(--text-color);opacity:0.6;line-height:1.6;">'
            'Entdecke ähnliche Aktien zu deinen Watchlist-Positionen — '
            'mit vollständigen Velox-Scores vorberechnet.'
            '</div>'
            '<div style="font-size:0.72rem;color:var(--text-color);opacity:0.38;margin-top:0.4rem;">'
            '→ Analysiere eine Aktie und öffne den Velox Radar im Analyse-Tab.</div>'
            '</div>',
            unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Velox Radar
# ──────────────────────────────────────────────────────────────────────────────
with tab_radar:
    # ── Session State ─────────────────────────────────────────────────────────
    for _rk, _rv in [("radar_mode_sel", "Themen"),
                     ("radar_theme", None),
                     ("radar_search_tk", ""),
                     ("radar_from_ticker", None),
                     ("radar_kw_result", None),
                     ("radar_kw_offset", 0),
                     ("ki_radar_query", ""),
                     ("ki_radar_result", None),
                     ("ki_radar_bust", 0),
                     ("radar_theme_limit", 4),
                     ("radar_bookmarks", {})]:
        if _rk not in st.session_state: st.session_state[_rk] = _rv

    # Theme detection
    try:
        _r_dark = st.get_option("theme.base") == "dark"
    except Exception:
        _r_dark = False

    # ── Premium Hero ──────────────────────────────────────────────────────────
    st.markdown(
        '<div style="'
        'background:linear-gradient(135deg,'
        'rgba(245,158,11,0.10) 0%,rgba(251,191,36,0.05) 50%,rgba(245,158,11,0.08) 100%);'
        'border:1px solid rgba(245,158,11,0.22);'
        'border-radius:20px;padding:1.8rem 2rem 1.5rem 2rem;'
        'margin-bottom:1.5rem;position:relative;overflow:hidden;">'
        # Dekorative Kreise
        '<div style="position:absolute;top:-40%;right:-8%;width:320px;height:320px;'
        'background:radial-gradient(circle,rgba(245,158,11,0.12) 0%,transparent 65%);'
        'pointer-events:none;"></div>'
        '<div style="position:absolute;bottom:-30%;left:-5%;width:200px;height:200px;'
        'background:radial-gradient(circle,rgba(251,191,36,0.07) 0%,transparent 65%);'
        'pointer-events:none;"></div>'
        # Brand
        '<div style="display:flex;align-items:center;gap:0.55rem;margin-bottom:0.8rem;">'
        '<span style="font-size:1.4rem;line-height:1;">◎</span>'
        '<span style="font-size:0.62rem;font-weight:900;letter-spacing:0.22em;'
        'text-transform:uppercase;color:#f59e0b;">Velox Radar</span>'
        '<span style="font-size:0.55rem;padding:2px 9px;border-radius:20px;'
        'background:rgba(245,158,11,0.15);color:#f59e0b;'
        'border:1px solid rgba(245,158,11,0.3);font-weight:700;letter-spacing:0.1em;">'
        'DISCOVERY ENGINE</span>'
        '</div>'
        # Headline
        '<div style="font-size:1.65rem;font-weight:800;color:var(--text-color);'
        'letter-spacing:-0.025em;line-height:1.25;margin-bottom:0.55rem;">'
        'Entdecke dein<br>nächstes Investment.</div>'
        # Sub
        '<div style="font-size:0.88rem;color:var(--text-color);opacity:0.5;line-height:1.6;">'
        'Echte Velox-Scores · Kuratierte Ideen · Vergleichbare Aktien</div>'
        '</div>',
        unsafe_allow_html=True)

    # ── Kein Toggle — lineares Layout: KI-Suche → Themen & Branchen ────────
    _ki_available = OPENAI_AVAILABLE and bool(
        (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None) or
        __import__("os").environ.get("OPENAI_API_KEY"))

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 1: KI-RADAR VOLLTEXTSUCHE
    # ══════════════════════════════════════════════════════════════════════════
    if _ki_available:
        st.markdown(
            '<div style="font-size:0.62rem;font-weight:700;letter-spacing:0.2em;'
            'text-transform:uppercase;color:#f59e0b;margin-bottom:0.6rem;">'
            '✦ KI-Radar · Powered by OpenAI</div>',
            unsafe_allow_html=True)

        # Beispiel-Queries als klickbare Chips
        _examples = ["Hidden Champions Nischenwerte", "Erneuerbare Energien Cashflow",
                     "KI-Infrastruktur nicht NVIDIA", "Günstige Dividendenaktien",
                     "Rüstung Europa"]
        _ex_cols = st.columns(len(_examples))
        for _exi, (_exc, _etxt) in enumerate(zip(_ex_cols, _examples)):
            with _exc:
                if st.button(_etxt, key=f"ki_ex_{_exi}", use_container_width=True,
                             help=f"Suche: {_etxt}"):
                    st.session_state["ki_radar_query"]  = _etxt
                    st.session_state["ki_radar_result"] = None
                    st.session_state["ki_radar_bust"]   = (
                        st.session_state.get("ki_radar_bust", 0) + 1)
                    st.session_state.pop("ki_radar_input", None)
                    st.rerun()
        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

        # Suchfeld — prominent, full width
        _ki_prefill = st.session_state.pop("ki_radar_prefill", "")
        # Premium Suchfeld mit eingebautem Button-Look
        _ki_s1, _ki_s2 = st.columns([6, 1])
        with _ki_s1:
            _ki_q = st.text_input("KI-Suche",
                value=_ki_prefill,
                placeholder="Was interessiert dich? z.B. 'Hidden Champions Industrie' · 'KI-Infrastruktur' · 'Dividenden Europa'…",
                label_visibility="collapsed", key="ki_radar_input")
        with _ki_s2:
            _ki_go = st.button("✦ Analysieren", key="ki_go", use_container_width=True,
                               type="primary")

        if _ki_go and _ki_q:
            st.session_state["ki_radar_query"]  = _ki_q
            st.session_state["ki_radar_result"] = None
            st.session_state["ki_radar_bust"]   = (
                st.session_state.get("ki_radar_bust", 0) + 1)
            st.rerun()

        _ki_query  = st.session_state.get("ki_radar_query", "")
        _ki_result = st.session_state.get("ki_radar_result")

        if _ki_query and _ki_result is None:
            _bust = st.session_state.get("ki_radar_bust", 0)
            with st.spinner(f"✦ KI analysiert '{_ki_query}'…"):
                _ki_items = ai_radar_discovery(_ki_query, n=12, bust=_bust)
            st.session_state["ki_radar_result"] = _ki_items

        _ki_items = st.session_state.get("ki_radar_result", [])
        if _ki_items:
            _kw_off2  = st.session_state.get("radar_kw_offset", 0)
            _ki_page2 = _ki_items[_kw_off2:_kw_off2 + 6]
            st.markdown(
                f'<div style="font-size:0.62rem;letter-spacing:0.14em;'
                f'text-transform:uppercase;color:var(--text-color);'
                f'opacity:0.38;margin:0.8rem 0 0.5rem 0;">'
                f'KI-Ergebnisse für „{_ki_query}"</div>',
                unsafe_allow_html=True)
            _ki_bm_state = st.session_state.get("radar_bookmarks", {})
            # Bookmark bar
            if _ki_bm_state:
                _bm_tks2 = list(_ki_bm_state.keys())
                st.markdown(
                    f'<div style="background:var(--secondary-background-color);'
                    f'border:1px solid rgba(16,185,129,0.25);border-radius:10px;'
                    f'padding:0.55rem 0.8rem;margin-bottom:0.6rem;'
                    f'display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap;">'
                    f'<span style="font-size:0.6rem;font-weight:700;letter-spacing:0.12em;'
                    f'text-transform:uppercase;color:#10b981;">★ {len(_bm_tks2)} gemerkt</span>'
                    + " · ".join(f'<span style="font-size:0.78rem;">{t}</span>' for t in _bm_tks2[:4])
                    + '</div>',
                    unsafe_allow_html=True)
                _bba1, _bba2, _bba3 = st.columns([2, 2, 1])
                with _bba1:
                    if st.button("▶ Analyse starten", key="ki_bm_analyse",
                                 use_container_width=True, type="primary"):
                        _ftk2 = _bm_tks2[0]
                        _fnm2 = _ki_bm_state.get(_ftk2, _ftk2)
                        st.session_state["ace_selected_ticker"]     = _ftk2
                        st.session_state["ace_search_q"]            = _fnm2
                        st.session_state["_auto_switch_to_analyse"] = True
                        for _rk4 in ("fund_score","timing_score","story_score","story_info",
                                     "chart_df","ace_direct_ticker","ace_search_input"):
                            st.session_state.pop(_rk4, None)
                        st.session_state["auto_run_fund"] = True
                        st.rerun()
                with _bba3:
                    if st.button("✕", key="ki_bm_reset2", use_container_width=True):
                        st.session_state["radar_bookmarks"] = {}
                        st.rerun()

            _ki_col1, _ki_col2 = st.columns(2)
            for _kii2, (_ktk2, _knm2, _ktp2, _kth2, _kconf2) in enumerate(_ki_page2):
                _conf_c2 = {"hoch":"#10b981","mittel":"#f59e0b","explorativ":"#ef4444"}.get(_kconf2,"#888")
                _why2 = (f'<span style="font-size:0.58rem;font-weight:700;'
                         f'text-transform:uppercase;letter-spacing:0.12em;'
                         f'color:{_conf_c2};margin-right:4px;">[{_kconf2}]</span>' + _kth2)
                _kmode2 = "Core Asset" if _ktp2 in ("Core Asset","ETF","Bond ETF") else "Hidden Champion"
                _kdata2 = _card_full_data(_ktk2, _kmode2)
                _is_bm2 = _ktk2 in st.session_state.get("radar_bookmarks", {})
                with ([_ki_col1, _ki_col2][_kii2 % 2]):
                    render_radar_card(_ktk2, _knm2, _why2, _kdata2,
                                     idx=f"ki2_{_kw_off2+_kii2}", mode=_kmode2,
                                     is_dark=_r_dark, show_cta=False)
                    _va_c, _bm_c = st.columns([5, 1])
                    with _va_c:
                        if st.button(f"▶ Vollanalyse · {_ktk2}",
                                     key=f"rc2_{_ktk2}_ki2{_kii2}",
                                     use_container_width=True):
                            with st.spinner(f"Lade {_ktk2}…"):
                                _pm3 = fetch_yahoo_metrics(_ktk2)
                                _px3 = fetch_extended_metrics(_ktk2)
                            if _pm3:
                                st.session_state["ace_yf_metrics"]  = _pm3
                                st.session_state["ace_ext_metrics"] = _px3 or {}
                                st.session_state["ace_yf_ticker"]   = _ktk2
                            st.session_state["ace_selected_ticker"]     = _ktk2
                            st.session_state["ace_search_q"]            = _knm2
                            st.session_state["_auto_switch_to_analyse"] = True
                            for _rk5 in ("ace_search_results","story_info","fund_score",
                                         "timing_score","story_score","chart_df",
                                         "ace_direct_ticker","ace_search_input"):
                                st.session_state.pop(_rk5, None)
                            st.rerun()
                    with _bm_c:
                        _bm_i2 = "★" if _is_bm2 else "＋"
                        if st.button(_bm_i2, key=f"ki_bm2_{_ktk2}_{_kw_off2+_kii2}",
                                     use_container_width=True):
                            _bm3 = dict(st.session_state.get("radar_bookmarks", {}))
                            if _is_bm2: _bm3.pop(_ktk2, None)
                            else:
                                _bm3[_ktk2] = _knm2
                                save_snapshot_to_watchlist({
                                    "ticker": _ktk2, "name": _knm2, "mode": _kmode2,
                                    "saved_at": datetime.now().isoformat(),
                                    "fund_score": _kdata2.get("fund"),
                                    "timing_score": None,
                                    "story_score": _kdata2.get("story"),
                                    "total_score": None,
                                    "action": "Aus Velox KI-Radar",
                                    "triggers": [], "risks": [], "metrics": {}, "red_flags": [],
                                }, notes="Aus KI-Radar gemerkt")
                            st.session_state["radar_bookmarks"] = _bm3
                            st.rerun()

            # Weitere Vorschläge
            if _kw_off2 + 6 < len(_ki_items) or len(_ki_items) >= 6:
                _wm1, _wm2, _wm3 = st.columns([1, 2, 1])
                with _wm2:
                    if st.button("✦ Weitere KI-Vorschläge", key="ki_more2",
                                 use_container_width=True):
                        _nxt = _kw_off2 + 6
                        if _nxt >= len(_ki_items):
                            st.session_state["ki_radar_result"] = None
                            st.session_state["ki_radar_bust"] = (
                                st.session_state.get("ki_radar_bust", 0) + 1)
                            _nxt = 0
                        st.session_state["radar_kw_offset"] = _nxt
                        st.rerun()

        elif _ki_query and st.session_state.get("ki_radar_result") == []:
            st.caption(f"Keine Ergebnisse für '{_ki_query}'. Versuche andere Begriffe.")

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    else:
        st.markdown(
            '<div style="background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.2);'
            'border-radius:12px;padding:0.85rem 1.1rem;margin-bottom:1rem;">'
            '<div style="font-size:0.65rem;font-weight:700;letter-spacing:0.18em;'
            'text-transform:uppercase;color:#f59e0b;margin-bottom:0.25rem;">✦ KI-Radar</div>'
            '<div style="font-size:0.8rem;color:var(--text-color);opacity:0.55;">'
            'Für KI-gestützte Empfehlungen wird ein OpenAI API-Key benötigt. '
            'Setze OPENAI_API_KEY in den Streamlit Secrets.</div></div>',
            unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 2: THEMEN & BRANCHEN (kuratiert)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div style="font-size:0.62rem;font-weight:700;letter-spacing:0.2em;'
        'text-transform:uppercase;color:var(--text-color);opacity:0.45;'
        'margin:1.2rem 0 0.7rem 0;">Kuratierte Themen & Branchen</div>',
        unsafe_allow_html=True)

    _sel_theme = st.session_state.get("radar_theme")
    if not _sel_theme:
        _th_names = list(VELOX_RADAR_THEMES.keys())
        for _ti in range(0, len(_th_names), 3):
            _th_row = _th_names[_ti:_ti+3]
            _th_cols = st.columns(len(_th_row))
            for _th_name, _th_col in zip(_th_row, _th_cols):
                _th = VELOX_RADAR_THEMES[_th_name]
                with _th_col:
                    st.markdown(
                        f'<div style="background:var(--secondary-background-color);'
                        f'border:1px solid rgba(128,128,128,0.12);'
                        f'border-top:3px solid {_th["color"]};'
                        f'border-radius:14px 14px 0 0;border-bottom:none;'
                        f'padding:1.1rem 1.1rem 0.9rem 1.1rem;position:relative;'
                        f'overflow:hidden;">'
                        f'<div style="position:absolute;top:-30%;right:-5%;'
                        f'width:80px;height:80px;background:radial-gradient(circle,'
                        f'{_th["color"]}12 0%,transparent 65%);pointer-events:none;"></div>'
                        f'<div style="font-size:0.9rem;font-weight:700;'
                        f'color:var(--text-color);margin-bottom:0.35rem;line-height:1.3;">'
                        f'{_th_name}</div>'
                        f'<div style="font-size:0.72rem;color:var(--text-color);'
                        f'opacity:0.45;line-height:1.55;min-height:2.5rem;">{_th["desc"]}</div>'
                        f'<div style="margin-top:0.65rem;font-size:0.62rem;'
                        f'color:{_th["color"]};font-weight:600;letter-spacing:0.06em;">'
                        f'{len(_th["stocks"])} Investments →</div>'
                        f'</div>',
                        unsafe_allow_html=True)
                    if st.button(f"Öffnen", key=f"th_{_th_name}",
                                 use_container_width=True):
                        st.session_state["radar_theme"] = _th_name
                        st.session_state["radar_kw_result"] = None
                        st.rerun()

        # R6: "Eigene Branche suchen" entfernt — KI-Radar oben deckt Freitext ab
    else:
        # Theme-Ergebnisse
        _th = VELOX_RADAR_THEMES[_sel_theme]
        _th_mode = _th.get("mode", "Core Asset")
        _rb1, _rb2 = st.columns([1, 5])
        with _rb1:
            if st.button("← Zurück", key="th_back", use_container_width=True):
                st.session_state["radar_theme"] = None
                st.rerun()
        with _rb2:
            st.markdown(
                f'<div style="padding-top:0.4rem;">'
                f'<span style="font-size:0.85rem;font-weight:700;">{_sel_theme}</span>'
                f'<span style="font-size:0.72rem;color:var(--text-color);'
                f'opacity:0.45;margin-left:0.5rem;">{_th["desc"]}</span>'
                f'</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
        _bm_th = st.session_state.get("radar_bookmarks", {})
        if _bm_th:
            _bm_tks3 = list(_bm_th.keys())
            st.markdown(
                f'<div style="background:var(--secondary-background-color);'
                f'border:1px solid rgba(16,185,129,0.25);border-radius:10px;'
                f'padding:0.55rem 0.8rem;margin-bottom:0.6rem;">'
                f'<span style="font-size:0.6rem;font-weight:700;color:#10b981;">'
                f'★ {len(_bm_tks3)} gemerkt: {", ".join(_bm_tks3[:3])}'
                + ("…" if len(_bm_tks3) > 3 else "")
                + '</span></div>', unsafe_allow_html=True)
            if st.button("▶ Analyse starten", key="th_bm_analyse",
                         use_container_width=False, type="primary"):
                _ftk3 = _bm_tks3[0]
                _fnm3 = _bm_th.get(_ftk3, _ftk3)
                st.session_state["ace_selected_ticker"]     = _ftk3
                st.session_state["ace_search_q"]            = _fnm3
                st.session_state["_auto_switch_to_analyse"] = True
                for _rk6 in ("fund_score","timing_score","story_score","story_info",
                             "chart_df","ace_direct_ticker","ace_search_input"):
                    st.session_state.pop(_rk6, None)
                st.session_state["auto_run_fund"] = True
                st.rerun()

        _th_limit2 = st.session_state.get("radar_theme_limit", 4)
        with st.spinner("Scores werden berechnet…"):
            _th_stocks_all2 = get_theme_stocks_with_scores(_sel_theme)
        _th_stocks2 = _th_stocks_all2[:_th_limit2]
        if _th_stocks2:
            _th_col1, _th_col2 = st.columns(2)
            for _ti3, (_tk3, _tname3, _twhy3, _tdata3) in enumerate(_th_stocks2):
                _is_bm3 = _tk3 in st.session_state.get("radar_bookmarks", {})
                with ([_th_col1, _th_col2][_ti3 % 2]):
                    render_radar_card(_tk3, _tname3, _twhy3, _tdata3,
                                     idx=f"th2_{_ti3}", mode=_th_mode,
                                     is_dark=_r_dark, show_cta=False)
                    _va3, _bm3c = st.columns([5, 1])
                    with _va3:
                        if st.button(f"▶ Vollanalyse · {_tk3}",
                                     key=f"rc_th2_{_tk3}_{_ti3}",
                                     use_container_width=True):
                            with st.spinner(f"Lade {_tk3}…"):
                                _pm4 = fetch_yahoo_metrics(_tk3)
                                _px4 = fetch_extended_metrics(_tk3)
                            if _pm4:
                                st.session_state["ace_yf_metrics"]  = _pm4
                                st.session_state["ace_ext_metrics"] = _px4 or {}
                                st.session_state["ace_yf_ticker"]   = _tk3
                            st.session_state["ace_selected_ticker"]     = _tk3
                            st.session_state["ace_search_q"]            = _tname3
                            st.session_state["_auto_switch_to_analyse"] = True
                            for _rk7 in ("ace_search_results","story_info","fund_score",
                                         "timing_score","story_score","chart_df",
                                         "ace_direct_ticker","ace_search_input"):
                                st.session_state.pop(_rk7, None)
                            st.rerun()
                    with _bm3c:
                        _bm_i3 = "★" if _is_bm3 else "＋"
                        if st.button(_bm_i3, key=f"th_bm_{_tk3}_{_ti3}",
                                     use_container_width=True):
                            _bm4 = dict(st.session_state.get("radar_bookmarks", {}))
                            if _is_bm3: _bm4.pop(_tk3, None)
                            else:
                                _bm4[_tk3] = _tname3
                                save_snapshot_to_watchlist({
                                    "ticker": _tk3, "name": _tname3, "mode": _th_mode,
                                    "saved_at": datetime.now().isoformat(),
                                    "fund_score": _tdata3.get("fund"),
                                    "timing_score": None,
                                    "story_score": _tdata3.get("story"),
                                    "total_score": None,
                                    "action": f"Aus Radar · {_sel_theme}",
                                    "triggers": [], "risks": [], "metrics": {}, "red_flags": [],
                                }, notes=f"Aus Radar-Thema: {_sel_theme}")
                            st.session_state["radar_bookmarks"] = _bm4
                            st.rerun()

        if _th_limit2 < len(_th_stocks_all2):
            _wt1, _wt2, _wt3 = st.columns([1, 2, 1])
            with _wt2:
                if st.button(f"+ Weitere Investments ({len(_th_stocks_all2)-_th_limit2} verbleibend)",
                             key="th_more2", use_container_width=True):
                    st.session_state["radar_theme_limit"] = _th_limit2 + 4
                    st.rerun()



# TAB 4 — Portfolio
# ──────────────────────────────────────────────────────────────────────────────
with tab_portfolio:
    st.session_state["_active_tab"] = "Portfolio"  # für kontext-aware Header-Button
    port_data   = load_portfolio()
    _api_key_pf = (st.secrets.get("OPENAI_API_KEY") or
                   os.environ.get("OPENAI_API_KEY") or "") if OPENAI_AVAILABLE else ""

    _all_pos = [p for pn in PORTFOLIO_NAMES
                  for p in port_data.get(pn, {}).get("positions", [])]
    _has_pos = len(_all_pos) > 0

    # ── Top-Bar ───────────────────────────────────────────────────────────────
    # Setup-State (Button ist im globalen Header "＋ Portfolio einrichten")
    _n_with_ticker = sum(1 for p in _all_pos if p.get("ticker"))
    _setup_open    = st.session_state.get("pf_show_setup", not _has_pos)

    # Auto-aufklappen beim ersten Besuch ohne Portfolio
    if not _has_pos and "pf_show_setup" not in st.session_state:
        st.session_state["pf_show_setup"] = True

    # ── Setup-Bereich ─────────────────────────────────────────────────────────
    if st.session_state.get("pf_show_setup", not _has_pos):

        # ── Portfolio Onboarding Wizard ───────────────────────────────────────
        # Wizard nur zeigen wenn noch keine Ziele gesetzt
        _any_goals = any(
            port_data.get(pn, {}).get("goals") for pn in PORTFOLIO_NAMES)
        _wiz_step = st.session_state.get("pf_wiz_step", 0 if not _any_goals else 99)

        if _wiz_step < 99:
            # ── Progress Dots ─────────────────────────────────────────────────
            _total_steps = 6
            _dots = "".join([
                f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
                f'background:{"#10b981" if i <= _wiz_step else "rgba(128,128,128,0.2)"};'
                f'margin:0 3px;"></span>'
                for i in range(_total_steps)
            ])
            st.markdown(
                f'<div style="text-align:center;margin-bottom:1.2rem;">{_dots}</div>',
                unsafe_allow_html=True)

            # ── Dicts vorab definieren (für alle Steps verfügbar) ─────────────
            _ziel_opts = {
                "Vermögensaufbau": "Langfristig Kapital aufbauen — für Freiheit, große Anschaffungen oder die nächste Generation.",
                "Altersvorsorge":  "Für den Ruhestand vorsorgen — unabhängig von staatlicher Rente mit eigenem Puffer.",
                "Dividendeneinkommen": "Regelmäßige Ausschüttungen als passives Einkommen — Dividenden die monatlich fließen.",
                "Hidden Champions": "Nischenmarktführer entdecken — überdurchschnittliche Rendite durch unbekannte Qualitätswerte.",
                "Kapitalerhalt":   "Kaufkraft erhalten und leicht wachsen — sicher und stabil, ohne großes Risiko.",
            }
            _risk_opts = {
                "Konservativ": {
                    "desc": "Lieber weniger Rendite als schlechte Nächte. Schwankungen machen mir Sorgen.",
                    "ca": 70, "hc": 10, "etf": 20,
                    "hint": "Viel ETF-Basis, wenige Einzelaktien"
                },
                "Ausgewogen": {
                    "desc": "Rendite und Stabilität in Balance. Ich akzeptiere gelegentliche Rücksetzer.",
                    "ca": 50, "hc": 20, "etf": 30,
                    "hint": "Gute Mischung aus ETF + Qualitätsaktien"
                },
                "Wachstum": {
                    "desc": "Maximale Rendite ist das Ziel. Ich bleibe ruhig auch bei -30%.",
                    "ca": 40, "hc": 40, "etf": 20,
                    "hint": "Mehr Einzelaktien, mehr Chancen und Risiken"
                }
            }

            # ── Step 0: Willkommen ────────────────────────────────────────────
            if _wiz_step == 0:
                st.markdown(
                    '<div style="text-align:center;padding:1rem 0 1.5rem 0;">'
                    '<div style="font-size:1.5rem;font-weight:800;margin-bottom:0.5rem;">'
                    'Willkommen bei deinem Velox Depot</div>'
                    '<div style="font-size:0.9rem;color:var(--text-color);opacity:0.55;'
                    'line-height:1.7;max-width:480px;margin:0 auto;">'
                    'Bevor wir loslegen — sag uns kurz was du mit deinem Depot erreichen '
                    'willst. So können wir dich noch gezielter unterstützen.</div>'
                    '</div>',
                    unsafe_allow_html=True)
                _, _wc, _ = st.columns([1, 2, 1])
                with _wc:
                    if st.button("Los geht's →", key="wiz_start",
                                 use_container_width=True, type="primary"):
                        st.session_state["pf_wiz_step"] = 1
                        st.rerun()

            # ── Step 1: Ziel ──────────────────────────────────────────────────
            elif _wiz_step == 1:
                st.markdown(
                    '<div style="font-size:1.1rem;font-weight:700;'
                    'margin-bottom:1rem;">Was ist dein Ziel?</div>',
                    unsafe_allow_html=True)
                _sel_ziel = st.session_state.get("pf_wiz_ziel", "")
                for _zk, _zv in _ziel_opts.items():
                    _is_sel = _sel_ziel == _zk
                    _zbg = "rgba(16,185,129,0.08)" if _is_sel else "var(--secondary-background-color)"
                    _zbd = "1.5px solid #10b981" if _is_sel else "1px solid rgba(128,128,128,0.14)"
                    st.markdown(
                        f'<div style="background:{_zbg};border:{_zbd};border-radius:12px;'
                        f'padding:0.75rem 1rem;margin-bottom:0.5rem;cursor:pointer;">'
                        f'<div style="font-weight:600;margin-bottom:0.2rem;">{_zk}</div>'
                        f'<div style="font-size:0.78rem;color:var(--text-color);opacity:0.55;">{_zv}</div>'
                        f'</div>',
                        unsafe_allow_html=True)
                    if st.button(f"✓ {_zk}", key=f"wiz_ziel_{_zk}",
                                 use_container_width=True,
                                 type="primary" if _is_sel else "secondary"):
                        st.session_state["pf_wiz_ziel"] = _zk
                        st.rerun()
                if _sel_ziel:
                    _, _wnc, _ = st.columns([1, 2, 1])
                    with _wnc:
                        if st.button("Weiter →", key="wiz_1_next",
                                     use_container_width=True):
                            st.session_state["pf_wiz_step"] = 2
                            st.rerun()

            # ── Step 2: Laufzeit ──────────────────────────────────────────────
            elif _wiz_step == 2:
                st.markdown(
                    '<div style="font-size:1.1rem;font-weight:700;'
                    'margin-bottom:1rem;">Wie lange planst du zu investieren?</div>',
                    unsafe_allow_html=True)
                _lz_opts = {
                    "Unter 5 Jahre": "Kurzfristig — du brauchst das Geld möglicherweise bald.",
                    "5 bis 15 Jahre": "Mittelfristig — genug Zeit für Wachstum, aber nicht ewig.",
                    "Über 15 Jahre": "Langfristig — der Zinseszins-Effekt arbeitet voll für dich."
                }
                _sel_lz = st.session_state.get("pf_wiz_laufzeit", "")
                _lc1, _lc2, _lc3 = st.columns(3)
                for _col, (_lk, _lv) in zip([_lc1, _lc2, _lc3], _lz_opts.items()):
                    _is_sel = _sel_lz == _lk
                    _lbg = "rgba(16,185,129,0.09)" if _is_sel else "var(--secondary-background-color)"
                    _lbd = "1.5px solid #10b981" if _is_sel else "1px solid rgba(128,128,128,0.14)"
                    with _col:
                        st.markdown(
                            f'<div style="background:{_lbg};border:{_lbd};border-radius:12px;'
                            f'padding:0.9rem 1rem;text-align:center;min-height:100px;">'
                            f'<div style="font-weight:700;font-size:0.9rem;">{_lk}</div>'
                            f'<div style="font-size:0.7rem;color:var(--text-color);'
                            f'opacity:0.5;margin-top:0.3rem;line-height:1.5;">{_lv}</div>'
                            f'</div>',
                            unsafe_allow_html=True)
                        if st.button(_lk, key=f"wiz_lz_{_lk}",
                                     use_container_width=True,
                                     type="primary" if _is_sel else "secondary"):
                            st.session_state["pf_wiz_laufzeit"] = _lk
                            st.rerun()
                if _sel_lz:
                    _wb1, _wb2 = st.columns(2)
                    with _wb1:
                        if st.button("← Zurück", key="wiz_2_back",
                                     use_container_width=True):
                            st.session_state["pf_wiz_step"] = 1
                            st.rerun()
                    with _wb2:
                        if st.button("Weiter →", key="wiz_2_next",
                                     use_container_width=True):
                            st.session_state["pf_wiz_step"] = 3
                            st.rerun()

            # ── Step 3: Sparrate + Renten-Ziel ───────────────────────────────
            elif _wiz_step == 3:
                import math
                st.markdown(
                    '<div style="font-size:1.1rem;font-weight:700;margin-bottom:0.3rem;">'
                    'Deine Zahlen</div>'
                    '<div style="font-size:0.82rem;color:var(--text-color);opacity:0.5;'
                    'margin-bottom:1rem;">Optional — je mehr wir wissen, desto besser können wir rechnen.</div>',
                    unsafe_allow_html=True)

                # Renten-Ziel
                st.markdown(
                    '<div style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;'
                    'color:var(--text-color);opacity:0.4;margin-bottom:0.4rem;">'
                    'Dein Renten-Ziel</div>',
                    unsafe_allow_html=True)
                _ra1, _ra2, _ra3 = st.columns(3)
                with _ra1:
                    _alter_jetzt = st.number_input(
                        "Dein aktuelles Alter",
                        min_value=18, max_value=80, step=1,
                        value=st.session_state.get("pf_wiz_alter", 35),
                        key="pf_wiz_alter_inp")
                    st.session_state["pf_wiz_alter"] = _alter_jetzt
                with _ra2:
                    _renten_alter = st.number_input(
                        "Gewünschtes Rentenalter",
                        min_value=40, max_value=90, step=1,
                        value=st.session_state.get("pf_wiz_renten_alter", 67),
                        key="pf_wiz_renten_alter_inp")
                    st.session_state["pf_wiz_renten_alter"] = _renten_alter
                with _ra3:
                    _monats_rente = st.number_input(
                        "Monatliche Wunsch-Rente (€)",
                        min_value=0, max_value=50000, step=100,
                        value=st.session_state.get("pf_wiz_monats_rente", 2000),
                        key="pf_wiz_monats_rente_inp",
                        help="Wie viel möchtest du monatlich aus dem Depot entnehmen?")
                    st.session_state["pf_wiz_monats_rente"] = _monats_rente

                # Sparrate
                st.markdown(
                    '<div style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;'
                    'color:var(--text-color);opacity:0.4;margin:0.8rem 0 0.4rem 0;">'
                    'Deine Sparrate</div>',
                    unsafe_allow_html=True)
                _monatlich = st.number_input(
                    "Monatliche Sparrate (€)",
                    min_value=0, max_value=100000, step=50,
                    value=st.session_state.get("pf_wiz_monatlich", 300),
                    key="pf_wiz_monatlich_inp")
                st.session_state["pf_wiz_monatlich"] = _monatlich

                # Berechnung
                _jahre_bis_rente = max(_renten_alter - _alter_jetzt, 1)
                # 4%-Regel: Depot = Jahresausgaben / 0.04
                _zielwert_berechnet = int(_monats_rente * 12 / 0.04)
                st.session_state["pf_wiz_zielwert"] = _zielwert_berechnet

                if _monats_rente > 0:
                    _box_color = "rgba(16,185,129,0.07)"
                    _box_border = "rgba(16,185,129,0.2)"
                    _proj_lines = [
                        f'Für <strong>{_monats_rente:,} € mtl.</strong> Rente ab {_renten_alter} brauchst du ca. '
                        f'<strong>{_zielwert_berechnet:,} €</strong> im Depot (4%-Regel).',
                    ]
                    if _monatlich > 0:
                        _rate = 0.07 / 12
                        try:
                            _n = math.log(1 + (_zielwert_berechnet * _rate) / _monatlich) / math.log(1 + _rate)
                            _jahre_needed = _n / 12
                            if abs(_jahre_needed - _jahre_bis_rente) < 3:
                                _proj_lines.append(f'Mit {_monatlich:,} €/Monat erreichst du das in ca. <strong>{_jahre_needed:.0f} Jahren</strong> — perfekt! ✓')
                            elif _jahre_needed < _jahre_bis_rente:
                                _proj_lines.append(f'Mit {_monatlich:,} €/Monat erreichst du das schon früher — in ca. <strong>{_jahre_needed:.0f} Jahren</strong>. ✓')
                            else:
                                _diff = _jahre_needed - _jahre_bis_rente
                                _more = int((_zielwert_berechnet - _monatlich * _n) / _n * 12 / 12)
                                _proj_lines.append(
                                    f'Mit {_monatlich:,} €/Monat dauert es ca. <strong>{_jahre_needed:.0f} Jahre</strong> — '
                                    f'{_diff:.0f} Jahre länger als geplant. Erhöhe die Sparrate oder passe das Ziel an.')
                                _box_color = "rgba(245,158,11,0.07)"
                                _box_border = "rgba(245,158,11,0.25)"
                        except Exception:
                            pass
                    st.markdown(
                        f'<div style="background:{_box_color};border:1px solid {_box_border};'
                        f'border-radius:10px;padding:0.8rem 1rem;margin-top:0.5rem;">'
                        + "".join(f'<div style="font-size:0.82rem;line-height:1.65;margin-bottom:0.2rem;">{l}</div>'
                                  for l in _proj_lines)
                        + '</div>',
                        unsafe_allow_html=True)

                _wb1, _wb2 = st.columns(2)
                with _wb1:
                    if st.button("← Zurück", key="wiz_3_back", use_container_width=True):
                        st.session_state["pf_wiz_step"] = 2; st.rerun()
                with _wb2:
                    if st.button("Weiter →", key="wiz_3_next", use_container_width=True):
                        st.session_state["pf_wiz_step"] = 4; st.rerun()

            # ── Step 4: Risikoprofil ──────────────────────────────────────────
            elif _wiz_step == 4:
                st.markdown(
                    '<div style="font-size:1.1rem;font-weight:700;'
                    'margin-bottom:1rem;">Wie gehst du mit Schwankungen um?</div>',
                    unsafe_allow_html=True)
                _sel_risk = st.session_state.get("pf_wiz_risiko", "")
                for _rk2, _rv2 in _risk_opts.items():
                    _is_sel = _sel_risk == _rk2
                    _rbg = "rgba(16,185,129,0.08)" if _is_sel else "var(--secondary-background-color)"
                    _rbd = "1.5px solid #10b981" if _is_sel else "1px solid rgba(128,128,128,0.14)"
                    st.markdown(
                        f'<div style="background:{_rbg};border:{_rbd};border-radius:12px;'
                        f'padding:0.75rem 1rem;margin-bottom:0.5rem;">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'<span style="font-weight:700;">{_rk2}</span>'
                        f'<span style="font-size:0.65rem;color:#10b981;opacity:0.8;">{_rv2["hint"]}</span>'
                        f'</div>'
                        f'<div style="font-size:0.78rem;color:var(--text-color);opacity:0.55;'
                        f'margin-top:0.2rem;">{_rv2["desc"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True)
                    if st.button(_rk2, key=f"wiz_risk_{_rk2}",
                                 use_container_width=True,
                                 type="primary" if _is_sel else "secondary"):
                        st.session_state["pf_wiz_risiko"] = _rk2
                        st.rerun()
                if _sel_risk:
                    _wb1, _wb2 = st.columns(2)
                    with _wb1:
                        if st.button("← Zurück", key="wiz_4_back", use_container_width=True):
                            st.session_state["pf_wiz_step"] = 3; st.rerun()
                    with _wb2:
                        if st.button("Ergebnis anzeigen →", key="wiz_4_next",
                                     use_container_width=True, type="primary"):
                            st.session_state["pf_wiz_step"] = 5; st.rerun()

            # ── Step 5: Empfehlung + Abschluss ────────────────────────────────
            elif _wiz_step == 5:
                _ziel    = st.session_state.get("pf_wiz_ziel", "Vermögensaufbau")
                _lz      = st.session_state.get("pf_wiz_laufzeit", "Über 15 Jahre")
                _mon     = st.session_state.get("pf_wiz_monatlich", 0)
                _zv      = st.session_state.get("pf_wiz_zielwert", 0)
                _risk    = st.session_state.get("pf_wiz_risiko", "Ausgewogen")
                _risk_d  = _risk_opts.get(_risk, _risk_opts["Ausgewogen"])

                st.markdown(
                    f'<div style="font-size:1.1rem;font-weight:700;margin-bottom:0.8rem;">'
                    f'Dein persönliches Depot-Profil</div>',
                    unsafe_allow_html=True)

                # Empfehlung Card
                st.markdown(
                    f'<div style="background:rgba(16,185,129,0.07);'
                    f'border:1px solid rgba(16,185,129,0.22);border-radius:14px;'
                    f'padding:1.1rem 1.3rem;margin-bottom:1rem;">'
                    f'<div style="font-size:0.6rem;letter-spacing:0.15em;text-transform:uppercase;'
                    f'color:#10b981;margin-bottom:0.5rem;">Velox Übersicht</div>'
                    f'<div style="font-size:0.88rem;line-height:1.7;color:var(--text-color);">'
                    f'Für <strong>{_ziel}</strong> mit <strong>{_lz}</strong> Horizont '
                    f'und <strong>{_risk}</strong>em Profil ergibt folgendes Bild:'
                    f'</div>'
                    f'<div style="display:flex;gap:1rem;margin-top:0.8rem;flex-wrap:wrap;">'
                    f'<div style="text-align:center;flex:1;min-width:80px;">'
                    f'<div style="font-size:1.4rem;font-weight:800;color:#10b981;">{_risk_d["ca"]}%</div>'
                    f'<div style="font-size:0.65rem;opacity:0.6;">Core Assets</div></div>'
                    f'<div style="text-align:center;flex:1;min-width:80px;">'
                    f'<div style="font-size:1.4rem;font-weight:800;color:#8b5cf6;">{_risk_d["hc"]}%</div>'
                    f'<div style="font-size:0.65rem;opacity:0.6;">Hidden Champions</div></div>'
                    f'<div style="text-align:center;flex:1;min-width:80px;">'
                    f'<div style="font-size:1.4rem;font-weight:800;color:#f59e0b;">{_risk_d["etf"]}%</div>'
                    f'<div style="font-size:0.65rem;opacity:0.6;">ETFs</div></div>'
                    f'</div></div>',
                    unsafe_allow_html=True)

                # Zusammenfassung
                _sum_items = [
                    ("Ziel", _ziel), ("Laufzeit", _lz), ("Risikoprofil", _risk),
                ]
                if _mon > 0: _sum_items.append(("Monatlich", f"{_mon:,} €"))
                if _zv  > 0: _sum_items.append(("Zielwert",  f"{_zv:,} €"))
                _sum_html = "".join([
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:0.3rem 0;border-bottom:1px solid rgba(128,128,128,0.07);">'
                    f'<span style="font-size:0.75rem;opacity:0.5;">{k}</span>'
                    f'<span style="font-size:0.75rem;font-weight:600;">{v}</span></div>'
                    for k, v in _sum_items
                ])
                st.markdown(
                    f'<div style="background:var(--secondary-background-color);'
                    f'border-radius:10px;padding:0.75rem 1rem;margin-bottom:1rem;">'
                    f'{_sum_html}</div>',
                    unsafe_allow_html=True)

                _wb1, _wb2 = st.columns(2)
                with _wb1:
                    if st.button("← Nochmal anpassen", key="wiz_5_back",
                                 use_container_width=True):
                        st.session_state["pf_wiz_step"] = 4; st.rerun()
                with _wb2:
                    if st.button("Weiter →",
                                 key="wiz_5_done",
                                 use_container_width=True, type="primary"):
                        # Ziele im Portfolio JSON speichern
                        _goals = {
                            "ziel": _ziel, "laufzeit": _lz,
                            "risiko": _risk, "monatlich": _mon,
                            "zielwert": _zv,
                            "aufteilung": {
                                "core_pct": _risk_d["ca"],
                                "hc_pct": _risk_d["hc"],
                                "etf_pct": _risk_d["etf"],
                            }
                        }
                        for _pn in PORTFOLIO_NAMES:
                            port_data.setdefault(_pn, {"positions": [], "snapshots": []})
                            port_data[_pn]["goals"] = _goals
                        save_portfolio(port_data)
                        st.session_state["pf_wiz_step"] = 6
                        st.rerun()

            # ── Step 6: Hast du bereits ein Portfolio? ────────────────────────
            elif _wiz_step == 6:
                st.markdown(
                    '<div style="font-size:1.1rem;font-weight:700;margin-bottom:0.5rem;">'
                    'Hast du bereits ein Portfolio?</div>'
                    '<div style="font-size:0.82rem;color:var(--text-color);opacity:0.5;'
                    'margin-bottom:1.2rem;line-height:1.6;">'
                    'Wenn du Positionen bei einer Bank oder einem Broker hast, '
                    'können wir sie importieren. Oder du startest direkt mit der '
                    'Analyse und dem Radar.</div>',
                    unsafe_allow_html=True)

                _opt_a, _opt_b = st.columns(2)
                with _opt_a:
                    st.markdown(
                        '<div style="background:var(--secondary-background-color);'
                        'border:1px solid rgba(128,128,128,0.14);border-radius:14px;'
                        'padding:1.2rem 1.1rem;text-align:center;min-height:130px;">'
                        '<div style="font-weight:700;margin-bottom:0.3rem;font-size:0.95rem;">'
                        'Ja, ich habe Positionen</div>'
                        '<div style="font-size:0.75rem;color:var(--text-color);opacity:0.5;'
                        'line-height:1.5;">'
                        'Ich importiere mein bestehendes Depot via PDF oder gebe '
                        'Positionen manuell ein.</div>'
                        '</div>',
                        unsafe_allow_html=True)
                    if st.button("Portfolio importieren →", key="wiz_6_import",
                                 use_container_width=True, type="primary"):
                        st.session_state["pf_wiz_step"] = 99
                        st.rerun()

                with _opt_b:
                    st.markdown(
                        '<div style="background:var(--secondary-background-color);'
                        'border:1px solid rgba(128,128,128,0.14);border-radius:14px;'
                        'padding:1.2rem 1.1rem;text-align:center;min-height:130px;">'
                        '<div style="font-weight:700;margin-bottom:0.3rem;font-size:0.95rem;">'
                        'Ich starte neu</div>'
                        '<div style="font-size:0.75rem;color:var(--text-color);opacity:0.5;'
                        'line-height:1.5;">'
                        'Kein bestehendes Depot — ich möchte erst Aktien entdecken '
                        'und dann investieren.</div>'
                        '</div>',
                        unsafe_allow_html=True)
                    if st.button("Direkt loslegen →", key="wiz_6_fresh",
                                 use_container_width=True):
                        st.session_state["pf_wiz_step"] = 99
                        st.session_state["pf_show_setup"] = False
                        st.session_state["pf_wiz_fresh_start"] = True
                        st.rerun()

                # Nach "Direkt loslegen": Orientierungskarte
                if st.session_state.get("pf_wiz_fresh_start"):
                    st.markdown(
                        '<div style="background:rgba(16,185,129,0.06);'
                        'border:1px solid rgba(16,185,129,0.2);border-radius:12px;'
                        'padding:1rem 1.2rem;margin-top:0.8rem;">'
                        '<div style="font-weight:600;margin-bottom:0.5rem;">Was jetzt?</div>'
                        '<div style="font-size:0.82rem;color:var(--text-color);'
                        'opacity:0.65;line-height:1.7;">'
                        '→ <strong>Velox Radar</strong>: Entdecke Aktien die zu '
                        'deinem Profil passen<br>'
                        '→ <strong>Analyse</strong>: Gib eine Aktie ein und prüfe '
                        'ob sie zu dir passt<br>'
                        '→ <strong>Portfolio</strong>: Trage Positionen nach wenn du '
                        'dich entschieden hast'
                        '</div></div>',
                        unsafe_allow_html=True)
                    _fg1, _fg2 = st.columns(2)
                    with _fg1:
                        if st.button("◎ Zum Velox Radar", key="wiz_to_radar",
                                     use_container_width=True):
                            st.session_state["_auto_switch_to_radar"] = True
                            st.rerun()
                    with _fg2:
                        if st.button("▶ Aktie analysieren", key="wiz_to_analyse",
                                     use_container_width=True):
                            st.session_state["_auto_switch_to_analyse"] = True
                            st.rerun()

                st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
                if st.button("← Zurück", key="wiz_6_back", use_container_width=False):
                    st.session_state["pf_wiz_step"] = 5
                    st.rerun()

            st.divider()

        # ── Ziele-Zusammenfassung wenn schon gesetzt ──────────────────────────
        elif _any_goals:
            _saved_goals = next(
                (port_data.get(pn, {}).get("goals") for pn in PORTFOLIO_NAMES
                 if port_data.get(pn, {}).get("goals")), {})
            if _saved_goals:
                _rec = _saved_goals.get("empfehlung", {})
                st.markdown(
                    f'<div style="background:rgba(16,185,129,0.05);'
                    f'border:1px solid rgba(16,185,129,0.15);border-radius:10px;'
                    f'padding:0.65rem 1rem;margin-bottom:0.6rem;'
                    f'display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;">'
                    f'<div>'
                    f'<span style="font-size:0.62rem;letter-spacing:0.12em;text-transform:uppercase;'
                    f'color:#10b981;margin-right:0.5rem;">Dein Profil:</span>'
                    f'<span style="font-size:0.82rem;font-weight:600;">'
                    f'{_saved_goals.get("ziel","?")} · {_saved_goals.get("risiko","?")} · '
                    f'{_saved_goals.get("laufzeit","?")}</span>'
                    f'</div>'
                    f'<div style="font-size:0.72rem;opacity:0.5;cursor:pointer;"'
                    f'onclick="">Ziel: {_saved_goals.get("zielwert",0):,} €</div>'
                    f'</div>',
                    unsafe_allow_html=True)
                if st.button("Profil ändern", key="wiz_reset",
                             use_container_width=False):
                    st.session_state["pf_wiz_step"] = 0
                    st.rerun()
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        # ── Bestehende Setup-Optionen (PDF / Manuell) ─────────────────────────
        st.markdown('<div class="ace-section">Positionen hinzufügen</div>',
                    unsafe_allow_html=True)
        _method = st.radio(
            "Methode",
            ["Vermögensübersicht (PDF)", "Manuell"],
            horizontal=True, key="pf_method", label_visibility="collapsed")
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

        # ── A: Vermögensübersicht PDF ─────────────────────────────────────────
        if _method == "Vermögensübersicht (PDF)":
            st.markdown(
                '<div class="ace-hint">Lade die <b>Vermögensübersicht</b> hoch '
                '(TR App → Konto → Dokumente → Depotaufstellung). '
                'Enthält alle aktuellen Positionen mit exakten Anteilen und Kursen. '
                'Für den genauen Kaufkurs danach optional die Steuerübersicht-CSV '
                'hochladen.</div>', unsafe_allow_html=True)

            _pdf_up = st.file_uploader("Vermögensübersicht (.pdf)", type=["pdf"],
                                        key="pf_pdf_up")
            if _pdf_up and st.button("PDF analysieren", key="btn_pdf_parse"):
                with st.spinner("Lese Positionen aus PDF…"):
                    _pdf_positions = parse_vermoegensübersicht_pdf(_pdf_up.read())

                # Fehlercheck
                if _pdf_positions and "_error" in _pdf_positions[0]:
                    st.error(_pdf_positions[0]["_error"])
                elif not _pdf_positions:
                    st.error("Keine Positionen gefunden. Bitte Vermögensübersicht-PDF verwenden.")
                else:
                    # Ticker per ISIN suchen
                    _tmap2 = {}
                    _prog2 = st.progress(0, text=f"Suche Ticker für {len(_pdf_positions)} Positionen…")
                    for _idx5, _pp5 in enumerate(_pdf_positions):
                        _tmap2[_pp5["isin"]] = lookup_ticker_from_isin(_pp5["isin"])
                        _prog2.progress((_idx5 + 1) / len(_pdf_positions))
                    _prog2.empty()

                    # Aus CSV bereits vorhandene invested_csv-Daten holen
                    _existing_csv = {}
                    for _pn5, _pd5 in port_data.items():
                        for _ep5 in _pd5.get("positions", []):
                            _i5 = _ep5.get("isin", "")
                            if _i5 and _ep5.get("invested_csv"):
                                _existing_csv[_i5] = _ep5["invested_csv"]

                    st.session_state["pf_pdf_positions"] = _pdf_positions
                    st.session_state["pf_pdf_tickers"]   = _tmap2
                    st.session_state["pf_pdf_csv_inv"]   = _existing_csv
                    st.session_state["pf_pdf_assigns"]   = {
                        p["isin"]: PORTFOLIO_NAMES[0] for p in _pdf_positions}
                    st.rerun()

            # ── PDF Review-Tabelle ────────────────────────────────────────────
            if st.session_state.get("pf_pdf_positions"):
                _ppos  = st.session_state["pf_pdf_positions"]
                _ptk   = st.session_state.get("pf_pdf_tickers", {})
                _pcsv  = st.session_state.get("pf_pdf_csv_inv", {})

                _n_csv_match = sum(1 for p in _ppos if _pcsv.get(p["isin"]))
                _csv_badge = (
                    f'<span style="color:#00C864;"><b>{_n_csv_match} Kaufkurse</b>'
                    f' aus CSV berechnet</span>'
                    if _n_csv_match else
                    '<span style="color:#F5A623;">↓ Kaufkurs fehlt — '
                    'Steuerübersicht-CSV unten hochladen</span>'
                )
                st.markdown(
                    f'<div class="ace-card" style="margin-bottom:0.6rem;">'
                    f'<b>{len(_ppos)} Positionen</b> aus PDF erkannt · '
                    f'{_csv_badge}</div>', unsafe_allow_html=True)

                # Bulk-Zuweisung
                _pb1, _pb2, _ = st.columns([2.2, 2.2, 5.6])
                with _pb1:
                    if st.button(f"Alle → {PORTFOLIO_NAMES[0]}",
                                 key="btn_pdf_all_p0", use_container_width=True):
                        st.session_state["pf_pdf_assigns"] = {
                            p["isin"]: PORTFOLIO_NAMES[0] for p in _ppos}
                        st.rerun()
                with _pb2:
                    if st.button(f"Alle → {PORTFOLIO_NAMES[1]}",
                                 key="btn_pdf_all_p1", use_container_width=True):
                        st.session_state["pf_pdf_assigns"] = {
                            p["isin"]: PORTFOLIO_NAMES[1] for p in _ppos}
                        st.rerun()

                # Spaltenheader
                _phh = st.columns([2.5, 1.1, 1.1, 1.1, 1.6, 1.9])
                for _phc, _pht in zip(_phh, ["Position","Anteile","Kurs","Wert","Ticker","Portfolio"]):
                    _phc.markdown(
                        f'<div style="font-size:0.79rem;color:#888;">{_pht}</div>',
                        unsafe_allow_html=True)

                _pdf_upd_tickers = {}
                _pdf_upd_assigns = {}

                for _pp6 in _ppos:
                    _pi6  = _pp6["isin"]
                    _at6  = _ptk.get(_pi6, "")
                    _inv6 = _pcsv.get(_pi6)
                    _avg6 = round(_inv6 / _pp6["shares"], 2) if (
                        _inv6 and _pp6.get("shares")) else None

                    _pc1,_pc2,_pc3,_pc4,_pc5,_pc6 = st.columns([2.5,1.1,1.1,1.1,1.6,1.9])
                    with _pc1:
                        _avg_badge = (f' · <span style="color:#00C864;font-size:0.76rem;">'
                                      f'Ø {_avg6:.2f} €</span>' if _avg6 else
                                      '<span style="color:#888;font-size:0.76rem;"> · kein Kaufkurs</span>')
                        st.markdown(
                            f'<div style="font-size:0.87rem;padding:0.2rem 0;">'
                            f'<b>{_pp6["name"][:40]}</b>{_avg_badge}<br>'
                            f'<span style="color:#666;font-size:0.76rem;">{_pi6}</span>'
                            f'</div>', unsafe_allow_html=True)
                    with _pc2:
                        st.markdown(
                            f'<div style="font-size:0.88rem;padding:0.25rem 0;">'
                            f'{_pp6["shares"]}</div>', unsafe_allow_html=True)
                    with _pc3:
                        _prc6 = _pp6.get("current_price")
                        st.markdown(
                            f'<div style="font-size:0.88rem;padding:0.25rem 0;">'
                            f'{f"{_prc6:.2f}" if _prc6 else "—"}</div>',
                            unsafe_allow_html=True)
                    with _pc4:
                        _cv6 = _pp6.get("current_value")
                        st.markdown(
                            f'<div style="font-size:0.88rem;font-weight:600;padding:0.25rem 0;">'
                            f'{f"{_cv6:,.0f} €" if _cv6 else "—"}</div>',
                            unsafe_allow_html=True)
                    with _pc5:
                        _tk6 = st.text_input("tk", value=_at6,
                                             key=f"pdf_tk_{_pi6}",
                                             label_visibility="collapsed",
                                             placeholder="Ticker…")
                        _pdf_upd_tickers[_pi6] = _tk6.upper().strip()
                    with _pc6:
                        _cur6 = st.session_state.get("pf_pdf_assigns",{}).get(
                            _pi6, PORTFOLIO_NAMES[0])
                        _sel6 = st.selectbox("pf", PORTFOLIO_NAMES,
                                             index=PORTFOLIO_NAMES.index(_cur6),
                                             key=f"pdf_pf_{_pi6}",
                                             label_visibility="collapsed")
                        _pdf_upd_assigns[_pi6] = _sel6

                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

                # Optionaler CSV-Upload für Kaufkurs
                if _n_csv_match < len(_ppos):
                    with st.expander("Kaufkurse aus Steuerübersicht ergänzen (optional)",
                                     expanded=(_n_csv_match == 0)):
                        st.markdown(
                            '<div class="ace-hint">Lade zusätzlich die '
                            'Steuerübersicht-CSV hoch, um den Ø Kaufkurs aus '
                            'echten Transaktionsdaten zu berechnen.</div>',
                            unsafe_allow_html=True)
                        _csv_up3 = st.file_uploader("CSV hochladen", type=["csv"],
                                                     key="pf_csv_for_pdf")
                        if _csv_up3 and st.button("CSV einlesen",
                                                   key="btn_csv_for_pdf"):
                            with st.spinner("Lese CSV…"):
                                _csv3 = parse_tr_csv_for_costbasis(_csv_up3.read())
                            if _csv3:
                                st.session_state["pf_pdf_csv_inv"] = {
                                    isin: d["invested"] for isin, d in _csv3.items()}
                                st.rerun()

                _pv1, _pv2, _ = st.columns([2.5, 1.5, 6])
                with _pv1:
                    if st.button("Portfolio einrichten", key="btn_pdf_save",
                                 use_container_width=True, type="primary"):
                        _pf_fr2 = load_portfolio()
                        _padd = 0
                        for _pp7 in _ppos:
                            _pi7  = _pp7["isin"]
                            _tk7  = _pdf_upd_tickers.get(_pi7, "")
                            _pn7  = _pdf_upd_assigns.get(_pi7, PORTFOLIO_NAMES[0])
                            _inv7 = _pcsv.get(_pi7)
                            _avg7 = (round(_inv7 / _pp7["shares"], 4)
                                     if (_inv7 and _pp7.get("shares")) else None)
                            # Duplikat-Prüfung
                            _ex7 = any(
                                p.get("isin") == _pi7 or
                                (_tk7 and (p.get("ticker","") or "").upper() == _tk7.upper())
                                for p in _pf_fr2.get(_pn7, {}).get("positions", [])
                            )
                            if not _ex7:
                                _pf_fr2[_pn7]["positions"].append({
                                    "isin":              _pi7,
                                    "ticker":            _tk7,
                                    "name":              _pp7["name"],
                                    "shares":            _pp7.get("shares"),
                                    "avg_price":         _avg7,
                                    "invested_csv":      _inv7,
                                    "current_price":     _pp7.get("current_price"),
                                    "last_price_update": datetime.now().strftime("%Y-%m-%d"),
                                    "notes":             "",
                                })
                                _padd += 1
                        _pf_fr2 = add_portfolio_snapshot(_pf_fr2)
                        save_portfolio(_pf_fr2)
                        port_data = _pf_fr2
                        for _k7 in ["pf_pdf_positions","pf_pdf_tickers",
                                    "pf_pdf_csv_inv","pf_pdf_assigns"]:
                            st.session_state.pop(_k7, None)
                        st.session_state["pf_show_setup"] = False
                        st.success(
                            f"{_padd} Positionen eingerichtet"
                            + (f" · {_n_csv_match} Kaufkurse berechnet" if _n_csv_match else "")
                            + ".")
                        st.rerun()
                with _pv2:
                    if st.button("Abbrechen", key="btn_pdf_cancel",
                                 use_container_width=True):
                        for _k7 in ["pf_pdf_positions","pf_pdf_tickers",
                                    "pf_pdf_csv_inv","pf_pdf_assigns"]:
                            st.session_state.pop(_k7, None)
                        st.session_state["pf_show_setup"] = False
                        st.rerun()

        # ── B: Manuell ────────────────────────────────────────────────────────
        else:  # "Manuell"
            _ma1, _ma2, _ma3 = st.columns(3)
            with _ma1:
                _mn   = st.text_input("Name", key="madd_name2")
                _mtk  = st.text_input("Ticker (Yahoo)", key="madd_tk2",
                                       placeholder="z.B. ASML.AS, MSFT")
            with _ma2:
                _mpf  = st.selectbox("Portfolio", PORTFOLIO_NAMES, key="madd_pf2")
                _mshr = st.number_input("Anteile", min_value=0.0, step=0.0001,
                                         format="%.4f", key="madd_shr2")
            with _ma3:
                _mavg = st.number_input("Ø Kaufkurs (€)", min_value=0.0,
                                         step=0.01, key="madd_avg2")
                _mnt  = st.text_input("Notiz", key="madd_nt2", placeholder="optional")
            if st.button("Position hinzufügen", key="btn_madd2"):
                if _mn:
                    _np2 = {"isin": "", "ticker": _mtk.upper().strip(),
                             "name": _mn,
                             "shares":    _mshr if _mshr > 0 else None,
                             "avg_price": _mavg if _mavg > 0 else None,
                             "current_price": None, "last_price_update": None,
                             "notes": _mnt}
                    _pf3 = load_portfolio()
                    if _np2["ticker"]:
                        with st.spinner("Hole Kurs…"):
                            try:
                                _inf3 = yf.Ticker(_np2["ticker"]).info or {}
                                _cp3  = safe_float(_inf3.get("regularMarketPrice")
                                                   or _inf3.get("currentPrice"))
                                if _cp3:
                                    _np2["current_price"]     = _cp3
                                    _np2["last_price_update"] = (
                                        datetime.now().strftime("%Y-%m-%d"))
                            except Exception:
                                pass
                    _pf3[_mpf]["positions"].append(_np2)
                    save_portfolio(_pf3)
                    port_data = _pf3
                    st.success(f"{_mn} hinzugefügt.")
                    st.rerun()

        st.divider()

    # ── Gesamtdepot Top-Bar ───────────────────────────────────────────────────
    if _has_pos:
        # Metriken über alle Portfolios berechnen
        _gd_all = [p for pn in PORTFOLIO_NAMES
                   for p in port_data.get(pn, {}).get("positions", [])]
        _gd_tv = _gd_ti = 0
        for _gp in _gd_all:
            _gddv = calc_position_derived(_gp)
            _gd_tv += _gddv.get("current_value") or 0
            _gd_ti += _gddv.get("invested") or 0
        _gd_pl    = _gd_tv - _gd_ti
        _gd_pct   = ((_gd_tv / _gd_ti - 1) * 100) if _gd_ti > 0 else 0
        _gd_pc    = "#10b981" if _gd_pl >= 0 else "#ef4444"
        _gd_sign  = "+" if _gd_pl >= 0 else ""
        _gd_bench = fetch_benchmark_return("IWDA.AS", "1y") or 0
        _gd_vs_bm = _gd_pct - _gd_bench
        _gd_vbc   = "#10b981" if _gd_vs_bm >= 0 else "#ef4444"
        _gd_monthly = (((1 + _gd_bench / 100) ** (1/12)) - 1) * 100
        _gd_pot   = max(4.0, min(15.0, _gd_bench + _gd_vs_bm * 0.3))

        _gd_left, _gd_right = st.columns([2, 1], gap="large")

        with _gd_left:
            # Kennzahlen-Card
            _gd_items = [
                ("Depotwert",        f"€ {_gd_tv:,.0f}",                    "var(--text-color)", "1.4rem"),
                ("Investiert",       f"€ {_gd_ti:,.0f}",                    "rgba(128,128,128,0.65)", "1rem"),
                ("Gewinn / Verlust", f"{_gd_sign}€ {abs(_gd_pl):,.0f}",     _gd_pc, "1.1rem"),
                ("Gesamt-Rendite",   f"{_gd_sign}{_gd_pct:.1f}%",           _gd_pc, "1.3rem"),
                ("vs. MSCI World",   f"{_gd_vs_bm:+.1f}%",                  _gd_vbc, "1.1rem"),
                ("Mtl. Wachstum (∅)",f"+{_gd_monthly:.2f}%",                "#10b981", "1rem"),
                ("Jährl. Potenzial", f"~{_gd_pot:.1f}%",                    "#f59e0b", "1rem"),
                ("Positionen",       f"{len(_gd_all)}",                      "var(--text-color)", "1.1rem"),
            ]
            _gd_html = (
                '<div style="background:var(--secondary-background-color);'
                'border:1px solid rgba(16,185,129,0.15);border-radius:14px;'
                'padding:1rem 1.2rem;">'
                '<div style="font-size:0.55rem;letter-spacing:0.14em;'
                'text-transform:uppercase;color:#10b981;margin-bottom:0.7rem;'
                'font-weight:700;">Depot-Kennzahlen</div>'
                '<div style="display:grid;grid-template-columns:repeat(4,1fr);'
                'gap:0.8rem;margin-bottom:0.7rem;padding-bottom:0.7rem;'
                'border-bottom:1px solid rgba(128,128,128,0.08);">'
            )
            for _gl, _gv, _gc, _gf in _gd_items[:4]:
                _gd_html += (f'<div><div style="font-size:0.52rem;text-transform:uppercase;'
                             f'letter-spacing:0.1em;color:var(--text-color);opacity:0.35;'
                             f'margin-bottom:0.2rem;">{_gl}</div>'
                             f'<div style="font-size:{_gf};font-weight:800;color:{_gc};">'
                             f'{_gv}</div></div>')
            _gd_html += '</div><div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.8rem;">'
            for _gl, _gv, _gc, _gf in _gd_items[4:]:
                _gd_html += (f'<div><div style="font-size:0.52rem;text-transform:uppercase;'
                             f'letter-spacing:0.1em;color:var(--text-color);opacity:0.35;'
                             f'margin-bottom:0.2rem;">{_gl}</div>'
                             f'<div style="font-size:{_gf};font-weight:700;color:{_gc};">'
                             f'{_gv}</div></div>')
            _gd_html += '</div></div>'
            st.markdown(_gd_html, unsafe_allow_html=True)

        _da_btn_clicked = False
        _da_sel = [pn for pn in PORTFOLIO_NAMES if port_data.get(pn, {}).get("positions")]

        with _gd_right:
            st.markdown('<div style="height:0.15rem;"></div>', unsafe_allow_html=True)
            # Button 1: Depot-Analyse (nur wenn API Key vorhanden)
            if _api_key_pf:
                _da_btn_clicked = st.button("◎  Depot-Analyse starten",
                                            key="btn_depot_analyse",
                                            use_container_width=True, type="primary")
            else:
                st.markdown(
                    '<div style="font-size:0.72rem;color:var(--text-color);opacity:0.38;'
                    'text-align:center;padding:0.5rem 0 0.2rem 0;">'
                    'Depot-Analyse benötigt OpenAI API-Key</div>',
                    unsafe_allow_html=True)
            st.markdown('<div style="height:0.4rem;"></div>', unsafe_allow_html=True)
            # Button 2: Portfolio einrichten
            if st.button("＋  Portfolio einrichten", key="btn_pf_setup_top",
                         use_container_width=True):
                st.session_state["pf_show_setup"] = True
                st.rerun()

        # Depot-Analyse ausführen (wenn Button geklickt)
        if _api_key_pf and _da_btn_clicked:
            if _da_sel:
                if _da_sel:
                    # Gesamte Positionsdaten aggregieren
                    _da_positions = []
                    _da_total_val = 0
                    _da_total_inv = 0
                    for _dpn in _da_sel:
                        for _dpos in port_data.get(_dpn, {}).get("positions", []):
                            _ddv = calc_position_derived(_dpos)
                            _da_total_val += _ddv.get("current_value") or 0
                            _da_total_inv += _ddv.get("invested") or 0

                    for _dpn in _da_sel:
                        for _dpos in port_data.get(_dpn, {}).get("positions", []):
                            _ddv = calc_position_derived(_dpos)
                            _dcv = _ddv.get("current_value") or 0
                            _dpl = _ddv.get("pl_pct") or 0
                            _wgt = (_dcv / _da_total_val * 100) if _da_total_val > 0 else 0
                            _nm  = _dpos.get("name") or _dpos.get("ticker","")
                            # ETF oder Stock erkennen
                            _atype = "etf" if any(
                                k in _nm.upper() for k in ["ETF","ISHARES","VANGUARD",
                                                             "XTRACKERS","AMUNDI","SPDR"]
                            ) else "stock"
                            if _nm:
                                _da_positions.append({
                                    "name": _nm,
                                    "ticker": _dpos.get("ticker",""),
                                    "portfolio": _dpn,
                                    "value": round(_dcv, 2),
                                    "weight_pct": round(_wgt, 1),
                                    "performance_pct": round(_dpl, 1),
                                    "asset_type": _atype,
                                    "sector": _dpos.get("sector",""),
                                    "avg_price": _dpos.get("avg_price"),
                                    "current_price": _dpos.get("current_price"),
                                })

                    _da_perf = ((_da_total_val / _da_total_inv - 1) * 100
                                if _da_total_inv > 0 else 0)
                    _da_json = {
                        "depot_value": round(_da_total_val, 2),
                        "invested_total": round(_da_total_inv, 2),
                        "performance_total_pct": round(_da_perf, 1),
                        "portfolios_included": _da_sel,
                        "position_count": len(_da_positions),
                        "positions": _da_positions,
                    }

                    import json as _json2
                    with st.spinner("Ace analysiert dein Depot…"):
                        try:
                            _da_client = OpenAI(api_key=_api_key_pf)
                            _da_prompt = ACE_DEPOT_PROMPT.replace(
                                "{{portfolio_json}}",
                                _json2.dumps(_da_json, ensure_ascii=False, indent=2))
                            _da_resp = _da_client.responses.create(
                                model="gpt-4.1-mini",
                                input=[{"role": "user", "content": _da_prompt}]
                            )
                            _da_result = (getattr(_da_resp, "output_text", "") or "").strip()
                            st.session_state["depot_analyse_result"] = _da_result
                            st.session_state["depot_analyse_portfolios"] = _da_sel
                        except Exception as _da_e:
                            st.error(f"Analyse fehlgeschlagen: {_da_e}")

        st.markdown('<div style="height:0.6rem;"></div>', unsafe_allow_html=True)

        # Depot-Analyse Ergebnis (Ace-Text) — unterhalb der Top-Bar, volle Breite
        _da_result = st.session_state.get("depot_analyse_result", "")
        if _da_result:
            import re as _re3
            def _render_depot_md(text):
                lines = text.split('\n')
                html_parts = []
                for line in lines:
                    line = line.strip()
                    if line.startswith('### '):
                        title = line[4:]
                        html_parts.append(
                            f'<div style="font-size:0.65rem;font-weight:700;'
                            f'letter-spacing:0.14em;text-transform:uppercase;'
                            f'color:#10b981;margin:1rem 0 0.4rem 0;">{title}</div>')
                    elif line.startswith('- '):
                        content = _re3.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line[2:])
                        html_parts.append(
                            f'<div style="display:flex;gap:0.5rem;margin-bottom:0.3rem;">'
                            f'<span style="color:#10b981;flex-shrink:0;">›</span>'
                            f'<span>{content}</span></div>')
                    elif line:
                        content = _re3.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
                        html_parts.append(
                            f'<div style="margin-bottom:0.4rem;line-height:1.65;">'
                            f'{content}</div>')
                    else:
                        html_parts.append('<div style="height:0.3rem;"></div>')
                return "".join(html_parts)

            _pf_label = " + ".join(st.session_state.get("depot_analyse_portfolios", []))
            _da_collapsed = st.session_state.get("depot_ana_collapsed", False)
            _dah1, _dah2 = st.columns([8, 1])
            with _dah1:
                st.markdown(
                    f'<div style="font-size:0.6rem;letter-spacing:0.14em;'
                    f'text-transform:uppercase;color:#10b981;'
                    f'margin-top:0.8rem;padding-bottom:0.3rem;">'
                    f'Ace · Depot-Analyse · {_pf_label}</div>',
                    unsafe_allow_html=True)
            with _dah2:
                st.markdown('<div style="padding-top:0.55rem;"></div>',
                            unsafe_allow_html=True)
                if st.button("−" if not _da_collapsed else "＋",
                             key="depot_collapse_btn",
                             use_container_width=True,
                             help="Text ein-/ausklappen"):
                    st.session_state["depot_ana_collapsed"] = not _da_collapsed
                    st.rerun()

            if not _da_collapsed:
                st.markdown(
                    f'<div style="background:var(--secondary-background-color);'
                    f'border:1px solid rgba(16,185,129,0.18);border-radius:14px;'
                    f'padding:1.2rem 1.4rem;">'
                    f'<div style="font-size:0.88rem;color:var(--text-color);">'
                    f'{_render_depot_md(_da_result)}'
                    f'</div>'
                    f'<div style="margin-top:1rem;padding-top:0.6rem;'
                    f'border-top:1px solid rgba(128,128,128,0.1);'
                    f'font-size:0.65rem;color:var(--text-color);opacity:0.3;">'
                    f'Keine Anlageberatung. Nur zu Informationszwecken.</div>'
                    f'</div>',
                    unsafe_allow_html=True)

        st.divider()

    # ── Ziel-Fortschrittsbalken ───────────────────────────────────────────────
    _saved_goals = next(
        (port_data.get(pn, {}).get("goals") for pn in PORTFOLIO_NAMES
         if port_data.get(pn, {}).get("goals")), None)

    if _saved_goals and _has_pos:
        _gz         = _saved_goals.get("ziel", "Ziel")
        _gzv        = float(_saved_goals.get("zielwert", 0) or 0)
        _glz_raw    = str(_saved_goals.get("laufzeit", "10") or "10")
        import re as _re_lz
        _glz_m      = _re_lz.search(r'\d+', _glz_raw)
        _glz        = int(_glz_m.group()) if _glz_m else 10
        _grisk      = _saved_goals.get("risiko", "")
        _gmon       = float(_saved_goals.get("monatlich", 0) or 0)
        _gauf       = _saved_goals.get("aufteilung", {})

        # Fortschritt
        _prog_pct   = min(100.0, (_gd_tv / _gzv * 100) if _gzv > 0 else 0)
        _prog_color = ("#10b981" if _prog_pct >= 75 else
                       "#f59e0b" if _prog_pct >= 40 else "#3b82f6")

        # Hochrechnung: wie viele Jahre noch bis zum Ziel?
        _fehlend    = max(0, _gzv - _gd_tv)
        _proj_Jahre = None
        if _gmon > 0 and _fehlend > 0:
            # Vereinfacht: lineares Wachstum + 7% p.a. auf bestehendes Depot
            _r = 0.07 / 12
            _n = 0
            _sim = _gd_tv
            while _sim < _gzv and _n < 600:   # max 50 Jahre
                _sim = _sim * (1 + _r) + _gmon
                _n += 1
            _proj_Jahre = round(_n / 12, 1) if _n < 600 else None

        # Collapsed/Expanded toggle
        _goal_open = st.session_state.get("pf_goal_expanded", False)

        # ── Balken (immer sichtbar) ───────────────────────────────────────────
        _bar_right_text = (f"Noch ~{_proj_Jahre} J." if _proj_Jahre
                           else f"{_prog_pct:.0f}% erreicht")
        _goal_bar_html = (
            f'<div style="background:var(--secondary-background-color);'
            f'border:1px solid rgba(128,128,128,0.12);border-radius:14px;'
            f'padding:0.85rem 1.1rem 0.8rem 1.1rem;margin-bottom:0.6rem;'
            f'cursor:pointer;" '
            f'onclick="">'
            # Zeile 1: Label + Zielwert + Toggle-Hint
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;margin-bottom:0.55rem;">'
            f'<div style="display:flex;align-items:center;gap:0.6rem;">'
            f'<span style="font-size:0.52rem;font-weight:800;letter-spacing:0.16em;'
            f'text-transform:uppercase;color:{_prog_color};">◎ Ziel</span>'
            f'<span style="font-size:0.88rem;font-weight:700;color:var(--text-color);">'
            f'{_gz}</span>'
            f'<span style="font-size:0.72rem;color:var(--text-color);opacity:0.45;">'
            f'· {_gzv:,.0f} €</span>'
            + (f'<span style="font-size:0.68rem;color:var(--text-color);opacity:0.35;'
               f'padding:1px 7px;border-radius:20px;'
               f'background:rgba(128,128,128,0.08);">{_grisk}</span>' if _grisk else '')
            + f'</div>'
            # Rechts: Aktuell + Projektion
            f'<div style="display:flex;align-items:center;gap:0.7rem;">'
            f'<span style="font-size:0.72rem;color:var(--text-color);opacity:0.5;">'
            f'{_gd_tv:,.0f} € von {_gzv:,.0f} €</span>'
            + (f'<span style="font-size:0.7rem;font-weight:700;color:{_prog_color};">'
               f'Noch ~{_proj_Jahre} J.</span>' if _proj_Jahre else '')
            + f'</div>'
            f'</div>'
            # Fortschrittsbalken
            f'<div style="height:8px;background:rgba(128,128,128,0.12);'
            f'border-radius:6px;overflow:hidden;position:relative;">'
            f'<div style="width:{_prog_pct:.1f}%;height:100%;border-radius:6px;'
            f'background:linear-gradient(90deg,{_prog_color}99,{_prog_color});'
            f'transition:width 0.4s ease;position:relative;">'
            # Shine auf dem Balken
            f'<div style="position:absolute;top:0;right:0;bottom:0;width:30%;'
            f'background:linear-gradient(90deg,transparent,rgba(255,255,255,0.25));'
            f'border-radius:0 6px 6px 0;"></div>'
            f'</div></div>'
            # Kleine Prozentangabe unter dem Balken
            f'<div style="display:flex;justify-content:space-between;'
            f'margin-top:0.3rem;">'
            f'<span style="font-size:0.62rem;color:{_prog_color};font-weight:700;">'
            f'{_prog_pct:.1f}% erreicht</span>'
            + (f'<span style="font-size:0.62rem;color:var(--text-color);opacity:0.38;">'
               f'Sparrate {_gmon:,.0f} €/Monat</span>' if _gmon > 0 else '')
            + f'</div>'
            f'</div>'
        )
        st.markdown(_goal_bar_html, unsafe_allow_html=True)

        # Toggle-Button
        _gtb1, _gtb2, _gtb3 = st.columns([3, 1.2, 3])
        with _gtb2:
            if st.button("▼ Ziele" if not _goal_open else "▲ Schließen",
                         key="pf_goal_toggle", use_container_width=True):
                st.session_state["pf_goal_expanded"] = not _goal_open
                st.rerun()

        # ── Aufgeklappte Ziel-Details ─────────────────────────────────────────
        if _goal_open:
            _goal_editing = st.session_state.get("pf_goal_editing", False)

            if not _goal_editing:
                # ── Ansicht: Daten + Aufteilung ───────────────────────────────
                _ca_p  = _gauf.get("core_pct", 50)
                _hc_p  = _gauf.get("hc_pct", 25)
                _etf_p = _gauf.get("etf_pct", 25)
                _gv_c1, _gv_c2 = st.columns([1, 1], gap="medium")
                with _gv_c1:
                    st.markdown(
                        f'<div style="background:var(--secondary-background-color);'
                        f'border:1px solid rgba(128,128,128,0.12);border-radius:12px;'
                        f'padding:1rem 1.1rem;">'
                        f'<div style="font-size:0.52rem;font-weight:700;letter-spacing:0.14em;'
                        f'text-transform:uppercase;color:{_prog_color};margin-bottom:0.7rem;">Deine Ziele</div>'
                        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.7rem;">'
                        + "".join(
                            f'<div><div style="font-size:0.52rem;text-transform:uppercase;'
                            f'letter-spacing:0.1em;color:var(--text-color);opacity:0.35;'
                            f'margin-bottom:0.15rem;">{lbl}</div>'
                            f'<div style="font-size:0.92rem;font-weight:700;">{val}</div></div>'
                            for lbl, val in [
                                ("Ziel", _gz),
                                ("Zielwert", f"{_gzv:,.0f} €"),
                                ("Laufzeit", _glz_raw),
                                ("Sparrate", f"{_gmon:,.0f} €/Monat"),
                                ("Risikoprofil", _grisk),
                                ("Prognose", f"~{_proj_Jahre} J." if _proj_Jahre else "—"),
                            ])
                        + f'</div></div>',
                        unsafe_allow_html=True)
                with _gv_c2:
                    st.markdown(
                        f'<div style="background:var(--secondary-background-color);'
                        f'border:1px solid rgba(128,128,128,0.12);border-radius:12px;'
                        f'padding:1rem 1.1rem;">'
                        f'<div style="font-size:0.52rem;font-weight:700;letter-spacing:0.14em;'
                        f'text-transform:uppercase;color:#f59e0b;margin-bottom:0.7rem;">'
                        f'Empfohlene Aufteilung</div>'
                        + "".join(
                            f'<div style="margin-bottom:0.55rem;">'
                            f'<div style="display:flex;justify-content:space-between;margin-bottom:0.2rem;">'
                            f'<span style="font-size:0.72rem;color:var(--text-color);opacity:0.6;">{lbl}</span>'
                            f'<span style="font-size:0.72rem;font-weight:700;color:{col};">{pct}%</span></div>'
                            f'<div style="height:5px;background:rgba(128,128,128,0.1);border-radius:3px;">'
                            f'<div style="width:{pct}%;height:100%;background:{col};border-radius:3px;"></div>'
                            f'</div></div>'
                            for lbl, pct, col in [
                                ("Core Assets", _ca_p, "#3b82f6"),
                                ("Hidden Champions", _hc_p, "#8b5cf6"),
                                ("ETFs (Basis)", _etf_p, "#10b981"),
                            ])
                        + f'</div>',
                        unsafe_allow_html=True)
                st.markdown('<div style="height:0.4rem;"></div>', unsafe_allow_html=True)
                _gv_b1, _gv_b2, _gv_b3 = st.columns([3, 1.5, 3])
                with _gv_b2:
                    if st.button("✎ Ziele bearbeiten", key="pf_goal_edit_btn",
                                 use_container_width=True):
                        st.session_state["pf_goal_editing"] = True
                        st.rerun()

            else:
                # ── Bearbeitungsmodus — Inline, kein Wizard ───────────────────
                st.markdown(
                    '<div style="font-size:0.55rem;font-weight:700;letter-spacing:0.14em;'
                    'text-transform:uppercase;color:#f59e0b;margin-bottom:0.6rem;">'
                    'Ziele anpassen</div>',
                    unsafe_allow_html=True)

                _ziel_opts_list = ["Vermögensaufbau", "Altersvorsorge",
                                   "Dividendeneinkommen", "Hidden Champions", "Kapitalerhalt"]
                _risk_opts_list = ["Konservativ", "Ausgewogen", "Wachstum"]
                _lz_opts_list   = ["1–3 Jahre", "3–5 Jahre", "5–10 Jahre",
                                   "10–15 Jahre", "Über 15 Jahre"]

                _ed1, _ed2 = st.columns(2)
                with _ed1:
                    _new_ziel = st.selectbox("Ziel", _ziel_opts_list,
                        index=_ziel_opts_list.index(_gz) if _gz in _ziel_opts_list else 0,
                        key="pf_edit_ziel", label_visibility="visible")
                    _new_lz = st.selectbox("Laufzeit", _lz_opts_list,
                        index=_lz_opts_list.index(_glz_raw) if _glz_raw in _lz_opts_list else 2,
                        key="pf_edit_lz", label_visibility="visible")
                with _ed2:
                    _new_zv = st.number_input("Zielwert (€)", min_value=0,
                        value=int(_gzv) if _gzv > 0 else 50000,
                        step=5000, key="pf_edit_zv")
                    _new_mon = st.number_input("Monatliche Sparrate (€)", min_value=0,
                        value=int(_gmon) if _gmon > 0 else 200,
                        step=50, key="pf_edit_mon")
                _new_risk = st.select_slider("Risikoprofil",
                    options=_risk_opts_list,
                    value=_grisk if _grisk in _risk_opts_list else "Ausgewogen",
                    key="pf_edit_risk")

                _risk_map = {
                    "Konservativ": {"ca": 70, "hc": 10, "etf": 20},
                    "Ausgewogen":  {"ca": 50, "hc": 20, "etf": 30},
                    "Wachstum":    {"ca": 40, "hc": 40, "etf": 20},
                }

                st.markdown('<div style="height:0.3rem;"></div>', unsafe_allow_html=True)
                _esv1, _esv2, _esv3 = st.columns([1, 1, 2])
                with _esv1:
                    if st.button("✓ Speichern", key="pf_goal_save",
                                 use_container_width=True, type="primary"):
                        _new_goals = {
                            "ziel": _new_ziel,
                            "laufzeit": _new_lz,
                            "risiko": _new_risk,
                            "monatlich": float(_new_mon),
                            "zielwert": float(_new_zv),
                            "aufteilung": _risk_map.get(_new_risk, _risk_map["Ausgewogen"]),
                        }
                        for _pn_g in PORTFOLIO_NAMES:
                            port_data.setdefault(_pn_g, {"positions": []})
                            port_data[_pn_g]["goals"] = _new_goals
                        save_portfolio(port_data)
                        st.session_state["pf_goal_editing"] = False
                        st.rerun()
                with _esv2:
                    if st.button("Abbrechen", key="pf_goal_cancel",
                                 use_container_width=True):
                        st.session_state["pf_goal_editing"] = False
                        st.rerun()

        st.markdown('<div style="height:0.4rem;"></div>', unsafe_allow_html=True)

    # ── Portfolio-Übersicht ───────────────────────────────────────────────────
    if not _has_pos and not st.session_state.get("pf_show_setup"):
        st.markdown(
            '<div class="ace-placeholder">Noch keine Positionen — '
            'oben auf "Portfolio einrichten" klicken.</div>',
            unsafe_allow_html=True)

    for pname in PORTFOLIO_NAMES:
        pdata     = port_data.get(pname, {})
        positions = pdata.get("positions", [])
        if not positions and not _has_pos:
            continue   # Leere Portfolios nur verstecken wenn gar nichts da ist

        # Header
        _ph, _pr1, _pr2 = st.columns([4, 1.5, 1.5])
        with _ph:
            _n_pos       = len(positions)
            _disp_name   = pf_display_name(port_data, pname)
            _rename_key  = f"pf_rename_{pname}"
            _is_renaming = st.session_state.get(_rename_key, False)
            if _is_renaming:
                _rn_c1, _rn_c2, _rn_c3 = st.columns([3, 1.2, 1])
                with _rn_c1:
                    _new_name = st.text_input("Name", value=_disp_name,
                                             key=f"pf_rename_input_{pname}",
                                             label_visibility="collapsed",
                                             max_chars=40)
                with _rn_c2:
                    if st.button("Speichern", key=f"pf_rename_save_{pname}",
                                 use_container_width=True, type="primary"):
                        port_data = pf_set_display_name(port_data, pname, _new_name)
                        st.session_state[_rename_key] = False
                        st.rerun()
                with _rn_c3:
                    if st.button("✕", key=f"pf_rename_cancel_{pname}", use_container_width=True):
                        st.session_state[_rename_key] = False
                        st.rerun()
            else:
                _nm_c1, _nm_c2 = st.columns([5, 1])
                with _nm_c1:
                    st.markdown(
                        f'<div style="display:flex;align-items:baseline;gap:0.6rem;'
                        f'margin:0.8rem 0 0.5rem 0;">'
                        f'<span style="font-size:1.0rem;font-weight:700;'
                        f'color:var(--text-color);">{_disp_name}</span>'
                        f'<span style="font-size:0.68rem;color:var(--text-color);opacity:0.38;">'
                        f'{_n_pos} Position{"en" if _n_pos != 1 else ""}</span>'
                        f'</div>',
                        unsafe_allow_html=True)
                with _nm_c2:
                    st.markdown('<div style="padding-top:0.5rem;"></div>', unsafe_allow_html=True)
                    if st.button("✎", key=f"pf_rename_btn_{pname}",
                                 help="Portfolio umbenennen"):
                        st.session_state[_rename_key] = True
                        st.rerun()
        with _pr1:
            st.markdown('<div style="padding-top:0.65rem;"></div>', unsafe_allow_html=True)
            if positions and _n_with_ticker:
                if st.button("Kurse aktualisieren", key=f"refresh_{pname}",
                             use_container_width=True):
                    with st.spinner("Aktualisiere…"):
                        port_data = refresh_portfolio_prices(port_data)
                        port_data = add_portfolio_snapshot(port_data)
                        save_portfolio(port_data)
                    st.rerun()
        with _pr2:
            st.markdown('<div style="padding-top:0.65rem;"></div>', unsafe_allow_html=True)
            if positions and not st.session_state.get(f"confirm_reset_{pname}"):
                if st.button("Portfolio löschen", key=f"reset_{pname}",
                             help="Alle Positionen entfernen",
                             use_container_width=True):
                    st.session_state[f"confirm_reset_{pname}"] = True
                    st.rerun()

        if st.session_state.get(f"confirm_reset_{pname}"):
            st.markdown(
                f'<div style="background:rgba(239,68,68,0.07);'
                f'border:1px solid rgba(239,68,68,0.25);border-radius:10px;'
                f'padding:0.75rem 1rem;margin-bottom:0.5rem;">'
                f'<div style="font-size:0.85rem;font-weight:600;margin-bottom:0.25rem;">'
                f'Alle {len(positions)} Positionen aus "{pname}" löschen?</div>'
                f'<div style="font-size:0.75rem;color:var(--text-color);opacity:0.5;">'
                f'Diese Aktion kann nicht rückgängig gemacht werden.</div></div>',
                unsafe_allow_html=True)
            _rc1, _rc2, _ = st.columns([1, 1, 4])
            with _rc1:
                if st.button("Ja, löschen", key=f"yes_{pname}",
                              use_container_width=True):
                    port_data[pname]["positions"] = []
                    save_portfolio(port_data)
                    st.session_state.pop(f"confirm_reset_{pname}", None)
                    st.session_state.pop(f"pf_analysis_{pname}", None)
                    st.rerun()
            with _rc2:
                if st.button("Abbrechen", key=f"no_{pname}",
                              use_container_width=True):
                    st.session_state.pop(f"confirm_reset_{pname}", None)
                    st.rerun()

        if not positions:
            # P5: Premium "Portfolio anlegen" Card für leeren Zustand
            _pf_cat_label = "Core Asset" if "Compounder" in pname or "Core" in pname else "Hidden Champion"
            _pf_cat_color = "#3b82f6" if _pf_cat_label == "Core Asset" else "#8b5cf6"
            st.markdown(
                f'<div style="background:linear-gradient(135deg,'
                f'rgba({("59,130,246" if _pf_cat_label=="Core Asset" else "139,92,246")},0.07) 0%,'
                f'rgba({("59,130,246" if _pf_cat_label=="Core Asset" else "139,92,246")},0.03) 100%);'
                f'border:1px dashed {_pf_cat_color}44;border-radius:18px;'
                f'padding:2rem 1.5rem;margin-bottom:1.5rem;text-align:center;'
                f'position:relative;overflow:hidden;">'
                f'<div style="position:absolute;top:-20%;right:-5%;width:180px;height:180px;'
                f'background:radial-gradient(circle,{_pf_cat_color}10 0%,transparent 65%);'
                f'pointer-events:none;"></div>'
                f'<div style="font-size:0.55rem;font-weight:800;letter-spacing:0.22em;'
                f'text-transform:uppercase;color:{_pf_cat_color};margin-bottom:0.7rem;">'
                f'{_pf_cat_label} · Leer</div>'
                f'<div style="font-size:1.15rem;font-weight:700;margin-bottom:0.5rem;">'
                f'Noch keine Positionen</div>'
                f'<div style="font-size:0.82rem;color:var(--text-color);opacity:0.5;'
                f'line-height:1.65;max-width:400px;margin:0 auto 1.2rem auto;">'
                f'Füge deine ersten Aktien hinzu — manuell oder via Depotaufstellung-PDF.</div>'
                f'</div>',
                unsafe_allow_html=True)
            _pfc1, _pfc2, _pfc3 = st.columns([1, 2, 1])
            with _pfc2:
                # Pulsierender Premium-Button
                st.markdown(
                    '<style>@keyframes velox-pulse{'
                    '0%{box-shadow:0 0 0 0 rgba(16,185,129,0.4);}'
                    '70%{box-shadow:0 0 0 10px rgba(16,185,129,0);}'
                    '100%{box-shadow:0 0 0 0 rgba(16,185,129,0);}}</style>'
                    '<style>div[data-testid="stButton"]:has(button[key^="pf_empty_add_"]) button{'
                    'animation:velox-pulse 1.8s ease infinite;}</style>',
                    unsafe_allow_html=True)
                if st.button("＋ Portfolio anlegen", key=f"pf_empty_add_{pname}",
                             use_container_width=True, type="primary"):
                    st.session_state["pf_show_setup"] = True
                    st.rerun()
        else:
            # Gesamtwerte
            _tot_cv = _tot_inv = 0
            for _p in positions:
                _dv = calc_position_derived(_p)
                _tot_cv  += _dv["current_value"]
                _tot_inv += _dv["invested"]
            _tot_pl  = _tot_cv - _tot_inv
            _tot_pct = (_tot_pl / _tot_inv * 100) if _tot_inv > 0 else 0
            _pl_c    = "#00C864" if _tot_pl >= 0 else "#FF4444"
            _pl_s    = "+" if _tot_pl >= 0 else ""
            _n_miss  = sum(1 for p in positions if not p.get("shares"))

            # ── P2/P3: Zweigeteilter Top-Bereich — Metrics + Kurzcheck | Chart ──
            _snaps = pdata.get("snapshots", [])
            _p2_left, _p2_right = st.columns([1, 2], gap="medium")

            with _p2_left:
                # Kompakte Metrics-Karte
                _pl_text = f"{_pl_s}{_tot_pl:,.0f} €" if _tot_inv > 0 else "—"
                _pct_text = f"{_pl_s}{_tot_pct:.1f}%" if _tot_inv > 0 else "—"
                st.markdown(
                    f'<div style="background:var(--secondary-background-color);'
                    f'border:1px solid rgba(128,128,128,0.12);border-radius:14px;'
                    f'padding:1rem 1.1rem 0.8rem 1.1rem;margin-bottom:0.65rem;">'
                    # Portfoliowert
                    f'<div style="font-size:0.55rem;letter-spacing:0.14em;'
                    f'text-transform:uppercase;color:var(--text-color);opacity:0.38;'
                    f'margin-bottom:0.15rem;">Portfoliowert</div>'
                    f'<div style="font-size:1.35rem;font-weight:800;color:var(--text-color);'
                    f'letter-spacing:-0.02em;margin-bottom:0.7rem;">'
                    f'{_tot_cv:,.0f} €</div>'
                    # P&L
                    f'<div style="display:flex;gap:1.2rem;">'
                    f'<div>'
                    f'<div style="font-size:0.55rem;letter-spacing:0.12em;'
                    f'text-transform:uppercase;color:var(--text-color);opacity:0.38;">G&V</div>'
                    f'<div style="font-size:0.9rem;font-weight:700;color:{_pl_c};">'
                    f'{_pl_text}</div></div>'
                    f'<div>'
                    f'<div style="font-size:0.55rem;letter-spacing:0.12em;'
                    f'text-transform:uppercase;color:var(--text-color);opacity:0.38;">Rendite</div>'
                    f'<div style="font-size:0.9rem;font-weight:700;color:{_pl_c};">'
                    f'{_pct_text}</div></div>'
                    f'<div>'
                    f'<div style="font-size:0.55rem;letter-spacing:0.12em;'
                    f'text-transform:uppercase;color:var(--text-color);opacity:0.38;">Positionen</div>'
                    f'<div style="font-size:0.9rem;font-weight:700;">{len(positions)}</div>'
                    f'</div></div>'
                    f'</div>',
                    unsafe_allow_html=True)
                # P3: Kurzcheck Button
                if st.button("↻  Kurzcheck", key=f"kurzcheck_{pname}",
                             use_container_width=True, type="primary",
                             help="Kurse aktualisieren + Portfolio-Snapshot"):
                    with st.spinner("Aktualisiere Kurse…"):
                        port_data = refresh_portfolio_prices(port_data)
                        port_data = add_portfolio_snapshot(port_data)
                        save_portfolio(port_data)
                    st.success("Kurse aktualisiert!")
                    st.rerun()

            with _p2_right:
                # ── Portfolioentwicklung Chart ────────────────────────────────
                if len(_snaps) >= 2:
                    _sdf = pd.DataFrame(_snaps).set_index("date")
                    _sdf.index = pd.to_datetime(_sdf.index)
                    _sdf = _sdf.sort_index()
                    _sbase = _sdf["total_value"].iloc[0]
                    _sdf["Portfolio (indexiert)"] = _sdf["total_value"] / _sbase * 100
                    try:
                        _sdays  = max((_sdf.index[-1] - _sdf.index[0]).days + 5, 30)
                        _sper   = "1y" if _sdays <= 365 else "2y"
                        _sbm    = yf.Ticker("IWDA.AS").history(period=_sper)["Close"]
                        if not _sbm.empty:
                            _sbm.index = _sbm.index.tz_localize(None)
                            _sbm = _sbm[_sbm.index >= _sdf.index[0]]
                            if not _sbm.empty:
                                _sbmb = _sbm.iloc[0]
                                _sdf["MSCI World (indexiert)"] = None
                                for _sdt in _sdf.index:
                                    _sni = _sbm.index.get_indexer([_sdt], method="nearest")[0]
                                    _sdf.at[_sdt, "MSCI World (indexiert)"] = (
                                        _sbm.iloc[_sni] / _sbmb * 100)
                    except Exception:
                        pass
                    _sndays = (_sdf.index[-1] - _sdf.index[0]).days
                    _snret  = (_sdf["total_value"].iloc[-1] /
                               _sdf["total_value"].iloc[0] - 1) * 100
                    _snrc   = "#10b981" if _snret >= 0 else "#ef4444"
                    # Chart rendern — Plotly oder Fallback
                    if PLOTLY_AVAILABLE:
                        try:
                            _ch_dark2 = st.get_option("theme.base") == "dark"
                            _pfont   = "#aaaaaa" if _ch_dark2 else "#666666"
                            _pgrid   = "rgba(255,255,255,0.05)" if _ch_dark2 else "rgba(0,0,0,0.05)"
                            _pfig = go.Figure()
                            if "MSCI World (indexiert)" in _sdf.columns:
                                _bm_s = _sdf["MSCI World (indexiert)"].dropna()
                                _pfig.add_trace(go.Scatter(
                                    x=_bm_s.index, y=_bm_s.values,
                                    name="MSCI World",
                                    line=dict(color="rgba(128,128,128,0.45)",
                                             width=1.5, dash="dot"),
                                    hovertemplate="%{y:.1f}<extra>MSCI World</extra>"
                                ))
                            _port_s = _sdf["Portfolio (indexiert)"].dropna()
                            _fill_col = "rgba(16,185,129,0.12)" if _snret >= 0 else "rgba(239,68,68,0.10)"
                            _pfig.add_trace(go.Scatter(
                                x=_port_s.index, y=_port_s.values,
                                name="Dein Portfolio",
                                line=dict(color=_snrc, width=2.5),
                                fill="tozeroy", fillcolor=_fill_col,
                                hovertemplate="%{y:.1f}<extra>Portfolio</extra>"
                            ))
                            _pfig.add_hline(y=100,
                                line=dict(color="rgba(128,128,128,0.25)", width=1, dash="dot"))
                            _pfig.update_layout(
                                height=220,
                                margin=dict(l=0, r=0, t=8, b=0),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color=_pfont, size=11),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                           xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                                           font=dict(size=10)),
                                xaxis=dict(showgrid=False, zeroline=False,
                                          tickfont=dict(size=9, color=_pfont)),
                                yaxis=dict(gridcolor=_pgrid, zeroline=False,
                                          tickfont=dict(size=9, color=_pfont)),
                                hovermode="x unified",
                                modebar=dict(bgcolor="rgba(0,0,0,0)",
                                            color=_pfont, activecolor=_snrc),
                            )
                            st.plotly_chart(_pfig, use_container_width=True,
                                            config={"displayModeBar": False})
                        except Exception:
                            st.line_chart(_sdf[[c for c in ["Portfolio (indexiert)",
                                                             "MSCI World (indexiert)"]
                                                if c in _sdf.columns]], height=200)
                    else:
                        st.line_chart(_sdf[[c for c in ["Portfolio (indexiert)",
                                                         "MSCI World (indexiert)"]
                                            if c in _sdf.columns]], height=200)
                elif len(_snaps) == 1:
                    st.caption("Portfolioentwicklung wird nach dem nächsten "
                               "Kurs-Update sichtbar.")
                else:
                    st.caption("Noch kein Snapshot — Kurzcheck starten.")

            # Fehlende Anteile-Warnung (Summary Card entfernt — Kennzahlen sind oben)
            if _n_miss:
                st.markdown(
                    f'<div style="font-size:0.82rem;color:#F5A623;padding:0.3rem 0 0.5rem 0;">'
                    f'Hinweis: {_n_miss} Position(en) ohne Anteile — unter "✎ Position bearbeiten" eintragen.</div>',
                    unsafe_allow_html=True)

            # ── Sortierung ───────────────────────────────────────────────────
            _sort_key = st.selectbox(
                "Sortierung", ["Größe ↓", "Größe ↑", "Performance ↓", "Performance ↑", "Name"],
                key=f"sort_{pname}", label_visibility="collapsed",
                index=0)
            def _sort_fn(p):
                d = calc_position_derived(p)
                if "Größe"       in _sort_key: return d["current_value"] or 0
                if "Performance" in _sort_key: return d["pl_pct"] or 0
                return (p.get("name") or "").lower()
            _sorted_positions = sorted(
                enumerate(positions),
                key=lambda x: _sort_fn(x[1]),
                reverse="↓" in _sort_key)

            # ── Positionsliste — Kachel-Design mit Inline-Detail ───────────
            _pf_open_key = f"pf_open_{pname}"
            if _pf_open_key not in st.session_state:
                st.session_state[_pf_open_key] = None

            # Scores im Hintergrund vorladen
            for _, _p in _sorted_positions:
                _tk_pre = _p.get("ticker", "")
                if _tk_pre and f"pf_score_{_tk_pre}" not in st.session_state:
                    _mode_pre = "Core Asset" if "Compounder" in pname or "Core" in pname else "Hidden Champion"
                    st.session_state[f"pf_score_{_tk_pre}"] = watchlist_quick_check(_tk_pre, _mode_pre)

            # CSS: "Position bearbeiten" Button andocken an Card
            st.markdown("""<style>
div[data-testid="stButton"]:has(button[key^="pf_tog_"]) {
    margin-top:-0.5rem!important;
}
div[data-testid="stButton"]:has(button[key^="pf_tog_"]) button {
    border-radius:0 0 14px 14px!important;border-top:none!important;
    background:transparent!important;font-size:0.72rem!important;
    opacity:0.55!important;height:2.3rem!important;min-height:unset!important;
}
div[data-testid="stButton"]:has(button[key^="pf_tog_"]) button:hover {
    opacity:1!important;background:rgba(16,185,129,0.06)!important;color:#10b981!important;
}
</style>""", unsafe_allow_html=True)

            _pos_cols = st.columns(2)
            for _pci, (_pi, _pos) in enumerate(_sorted_positions):
                _pdv   = calc_position_derived(_pos)
                _pcv   = _pdv["current_value"]
                _pinv  = _pdv["invested"]
                _ppl   = _pdv["pl_abs"]
                _ppct  = _pdv["pl_pct"]
                _pcol  = "#10b981" if (_ppl or 0) >= 0 else "#ef4444"
                _psign = "+" if (_ppl or 0) >= 0 else ""
                _pwgt  = (_pcv / _tot_cv * 100) if _tot_cv > 0 else 0
                _ptkr  = _pos.get("ticker") or "—"
                _pshr  = _pos.get("shares")
                _pavg  = _pos.get("avg_price")
                _pcp   = _pos.get("current_price")
                _pname_disp = _pos.get("name") or _ptkr
                _is_open_pos = st.session_state[_pf_open_key] == f"{pname}_{_pi}"

                # Datenqualitäts-Warnungen
                _warn_items = []
                if not _pos.get("ticker"):
                    _warn_items.append(("Ticker fehlt", "#ef4444"))
                elif _ptkr == "—":
                    _warn_items.append(("Ticker fehlt", "#ef4444"))
                if not _pos.get("avg_price") or (_pos.get("avg_price") or 0) <= 0:
                    _warn_items.append(("Kaufkurs fehlt", "#f59e0b"))
                if not _pos.get("shares") or (_pos.get("shares") or 0) <= 0:
                    _warn_items.append(("Anteile fehlen", "#f59e0b"))
                if not _pos.get("current_price") or (_pos.get("current_price") or 0) <= 0:
                    _warn_items.append(("Kurs nicht aktuell", "rgba(128,128,128,0.6)"))
                _mode_short  = "Core" if "Compounder" in pname or "Core" in pname else "HC"
                _mode_c = "#3b82f6" if _mode_short == "Core" else "#8b5cf6"
                _mode_bg= "rgba(59,130,246,0.09)" if _mode_short == "Core" else "rgba(139,92,246,0.09)"
                _qk_score = st.session_state.get(f"pf_score_{_ptkr}", {})
                _sc_val = _qk_score.get("fund")
                _sc_c   = ("#10b981" if (_sc_val or 0) >= 6.5
                            else "#f59e0b" if (_sc_val or 0) >= 5.0
                            else "#ef4444" if _sc_val else "rgba(128,128,128,0.3)")
                _sc_pct = int((_sc_val or 0) * 10)
                _border = f"2px solid {_sc_c}" if _is_open_pos else f"1px solid rgba(128,128,128,0.14)"

                if _pwgt >= 15:    _rolle = "Schwergewicht"
                elif _pwgt >= 8:   _rolle = "Kernposition"
                elif _pwgt >= 4:   _rolle = "Beimischung"
                else:              _rolle = "Kleine Position"

                with _pos_cols[_pci % 2]:
                    # Card mit abgeflachten unteren Ecken (Button dockt an)
                    st.markdown(
                        f'<div style="background:var(--secondary-background-color);'
                        f'border:{_border};border-radius:14px 14px 0 0;border-bottom:none;'
                        f'padding:0.9rem 1rem 0.75rem 1rem;position:relative;overflow:hidden;">'
                        f'<div style="position:absolute;top:0;left:0;right:0;height:2.5px;'
                        f'background:{_pcol};border-radius:14px 14px 0 0;"></div>'
                        f'<div style="display:flex;justify-content:space-between;'
                        f'align-items:center;margin-top:0.1rem;margin-bottom:0.35rem;">'
                        f'<div style="display:flex;align-items:center;gap:0.4rem;">'
                        f'<span style="font-size:0.45rem;font-weight:800;letter-spacing:0.14em;'
                        f'text-transform:uppercase;color:{_mode_c};background:{_mode_bg};'
                        f'border-radius:20px;padding:2px 6px;">{_mode_short}</span>'
                        f'<span style="font-size:1.05rem;font-weight:800;'
                        f'color:var(--text-color);letter-spacing:0.04em;">{_ptkr}</span>'
                        f'</div>'
                        f'<div style="text-align:right;">'
                        + (f'<div style="font-size:0.95rem;font-weight:700;color:var(--text-color);">'
                           f'€{_pcv:,.0f}</div>'
                           f'<div style="font-size:0.65rem;color:{_pcol};font-weight:600;">'
                           f'{_psign}{_ppct:.1f}% P&L</div>' if _pcv else
                           f'<div style="font-size:0.72rem;color:var(--text-color);opacity:0.4;">kein Kurs</div>')
                        + f'</div></div>'
                        f'<div style="font-size:0.72rem;color:var(--text-color);opacity:0.45;'
                        f'margin-bottom:{"0.3rem" if _warn_items else "0.45rem"};'
                        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
                        f'{_pname_disp[:30] + "…" if len(_pname_disp) > 30 else _pname_disp}'
                        f'<span style="margin-left:0.4rem;font-size:0.6rem;padding:1px 6px;'
                        f'border-radius:10px;background:rgba(128,128,128,0.08);'
                        f'color:var(--text-color);opacity:0.5;">{_rolle} · {_pwgt:.1f}%</span>'
                        f'</div>'
                        # Warn-Badges
                        + ("".join(
                            f'<span style="display:inline-block;font-size:0.6rem;'
                            f'font-weight:600;padding:2px 7px;border-radius:20px;'
                            f'background:{wc}18;color:{wc};margin-right:4px;margin-bottom:0.3rem;">'
                            f'⚠ {wt}</span>'
                            for wt, wc in _warn_items
                        ) if _warn_items else "")
                        + (f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.3rem;">'
                           f'<span style="font-size:1.2rem;font-weight:800;color:{_sc_c};min-width:2.5rem;">{_sc_val:.1f}</span>'
                           f'<div style="flex:1;height:3px;background:rgba(128,128,128,0.12);border-radius:2px;overflow:hidden;">'
                           f'<div style="width:{_sc_pct}%;height:100%;background:{_sc_c};border-radius:2px;"></div></div></div>'
                           if _sc_val else f'<div style="height:0.3rem;"></div>')
                        + f'<div style="display:flex;justify-content:space-between;'
                          f'font-size:0.68rem;color:var(--text-color);opacity:0.5;'
                          f'border-top:1px solid rgba(128,128,128,0.08);padding-top:0.4rem;">'
                        + (f'<span>Ø {_pavg:.2f} €</span>' if _pavg else '<span>kein Kaufkurs</span>')
                        + (f'<span>{_pshr:.2f} Anteile</span>' if _pshr else '<span>—</span>')
                        + (f'<span style="color:{_pcol};font-weight:600;">{_psign}{_ppl:,.0f} €</span>' if _ppl is not None else '<span>—</span>')
                        + f'</div>'
                        + f'</div>',
                        unsafe_allow_html=True)

                    # "Position bearbeiten" Button — dockt an Card an
                    _tog_lbl = "▲ Schließen" if _is_open_pos else "✎ Position bearbeiten"
                    if st.button(_tog_lbl, key=f"pf_tog_{pname}_{_pi}",
                                 use_container_width=True):
                        st.session_state[_pf_open_key] = (
                            None if _is_open_pos else f"{pname}_{_pi}")
                        st.rerun()

                    # ── Inline Detail-Panel ───────────────────────────────────
                    if _is_open_pos:
                        st.markdown(
                            f'<div style="background:var(--secondary-background-color);'
                            f'border:1px solid rgba(128,128,128,0.14);border-radius:0 0 12px 12px;'
                            f'padding:0.9rem 1rem;margin-top:-0.1rem;">'
                            f'<div style="font-size:0.58rem;letter-spacing:0.14em;text-transform:uppercase;'
                            f'color:var(--text-color);opacity:0.35;margin-bottom:0.6rem;">Daten anpassen</div>',
                            unsafe_allow_html=True)

                        # ── Ticker — prominent wenn fehlend ──────────────────
                        _ticker_missing = not _pos.get("ticker") or _ptkr == "—"
                        if _ticker_missing:
                            st.markdown(
                                '<div style="background:rgba(239,68,68,0.07);'
                                'border:1px solid rgba(239,68,68,0.25);'
                                'border-radius:8px;padding:0.5rem 0.75rem;'
                                'margin-bottom:0.55rem;font-size:0.78rem;">'
                                '⚠ Kein Ticker zugeordnet — bitte eintragen damit Kurse '
                                'geladen werden können.</div>',
                                unsafe_allow_html=True)
                        _tk_a, _tk_b = st.columns([2, 3])
                        with _tk_a:
                            _e_ticker = st.text_input(
                                "Ticker (z.B. REL.L, MSFT)",
                                value=_pos.get("ticker", ""),
                                key=f"etk_{pname}_{_pi}",
                                placeholder="Yahoo Finance Ticker",
                                help="Ticker-Symbol von Yahoo Finance. .DE = Xetra, .L = London").upper().strip()
                        with _tk_b:
                            if _e_ticker and _e_ticker != (_pos.get("ticker") or ""):
                                st.markdown(
                                    f'<div style="padding-top:1.65rem;font-size:0.72rem;'
                                    f'color:var(--text-color);opacity:0.5;">'
                                    f'Wird beim Speichern als Ticker gesetzt</div>',
                                    unsafe_allow_html=True)

                        # Kaufkurs + Anteile
                        _ia, _ib = st.columns(2)
                        with _ia:
                            _e_avg = st.number_input(
                                "Ø Kaufkurs (€)", value=float(_pavg or 0),
                                min_value=0.0, step=0.01,
                                key=f"eavg_{pname}_{_pi}",
                                help="Durchschnittlicher Einkaufskurs")
                        with _ib:
                            _e_shr = st.number_input(
                                "Anteile", value=float(_pshr or 0),
                                min_value=0.0, step=0.001, format="%.4f",
                                key=f"esh_{pname}_{_pi}")

                        # Notiz
                        _e_nt = st.text_input(
                            "Notiz", value=_pos.get("notes",""),
                            key=f"ent_{pname}_{_pi}",
                            placeholder="z.B. Nachkauf Q2, Dividende reinvestiert…",
                            label_visibility="visible")

                        # ── Transaktion: nur Toggle + Kurs ─────────────────────
                        st.markdown(
                            '<div style="font-size:0.58rem;letter-spacing:0.1em;'
                            'text-transform:uppercase;color:var(--text-color);'
                            'opacity:0.3;margin:0.6rem 0 0.3rem 0;">'
                            'Kauf / Verkauf buchen</div>',
                            unsafe_allow_html=True)
                        _tx_a, _tx_b = st.columns([1, 2])
                        with _tx_a:
                            _tx_typ = st.radio("Typ", ["Kauf", "Verkauf"],
                                               horizontal=True,
                                               key=f"tx_typ_{pname}_{_pi}",
                                               label_visibility="collapsed")
                        with _tx_b:
                            _tx_prc = st.number_input(
                                "Kurs (€)", min_value=0.01, step=0.01,
                                key=f"tx_prc_{pname}_{_pi}",
                                label_visibility="collapsed",
                                placeholder="Transaktionskurs in €")

                        st.caption(
                            "Beim Buchen wird der Ø Kaufkurs neu berechnet. "
                            "Anteile bitte oben anpassen.")

                        # Buttons: Buchen (= speichern) · Analyse · Entfernen
                        _bs1, _bs2, _bs3 = st.columns(3)
                        with _bs1:
                            if st.button("Buchen & Speichern",
                                         key=f"esave_{pname}_{_pi}",
                                         use_container_width=True, type="primary"):
                                # Daten speichern
                                _pos["shares"]    = _e_shr if _e_shr > 0 else None
                                _pos["avg_price"] = _e_avg if _e_avg > 0 else None
                                _pos["notes"]     = _e_nt
                                # Ticker speichern und Kurs laden wenn neu gesetzt
                                if _e_ticker and _e_ticker != (_pos.get("ticker") or ""):
                                    _pos["ticker"] = _e_ticker
                                    try:
                                        _ti_new = yf.Ticker(_e_ticker).info or {}
                                        _cp_new = safe_float(
                                            _ti_new.get("regularMarketPrice") or
                                            _ti_new.get("currentPrice"))
                                        if _cp_new:
                                            _pos["current_price"]     = _cp_new
                                            _pos["last_price_update"] = (
                                                datetime.now().strftime("%Y-%m-%d"))
                                        if not _pos.get("name") and (
                                                _ti_new.get("longName") or
                                                _ti_new.get("shortName")):
                                            _pos["name"] = (
                                                _ti_new.get("longName") or
                                                _ti_new.get("shortName", ""))
                                    except Exception:
                                        pass
                                # Transaktion buchen (Kurs-Änderung → Mittelwert neu)
                                if _tx_prc and _tx_prc > 0.01:
                                    _cur_shr = float(_e_shr or 0)
                                    _old_shr = float(_pshr or 0)
                                    if _tx_typ == "Kauf" and _cur_shr > _old_shr:
                                        apply_buy(_pos, _cur_shr - _old_shr, _tx_prc)
                                    elif _tx_typ == "Verkauf" and _old_shr > _cur_shr:
                                        apply_sell(_pos, _old_shr - _cur_shr)
                                    elif _tx_typ == "Kauf":
                                        apply_buy(_pos, 1.0, _tx_prc)
                                # Kurs aktualisieren
                                if _pos.get("ticker"):
                                    try:
                                        _ei = yf.Ticker(_pos["ticker"]).info or {}
                                        _ecp = safe_float(
                                            _ei.get("regularMarketPrice") or
                                            _ei.get("currentPrice"))
                                        if _ecp:
                                            _pos["current_price"]     = _ecp
                                            _pos["last_price_update"] = (
                                                datetime.now().strftime("%Y-%m-%d"))
                                    except Exception:
                                        pass
                                positions[_pi] = _pos
                                save_portfolio(port_data)
                                st.session_state[_pf_open_key] = None
                                st.rerun()
                        with _bs2:
                            if _ptkr != "—":
                                if st.button("▶ Analyse", key=f"pf_anal_{pname}_{_pi}",
                                             use_container_width=True):
                                    st.session_state["ace_selected_ticker"]     = _ptkr
                                    st.session_state["ace_search_q"]            = _pname_disp
                                    st.session_state["_auto_switch_to_analyse"] = True
                                    for _rk3 in ("ace_direct_ticker","ace_search_input",
                                                 "fund_score","timing_score","story_score",
                                                 "story_info","chart_df"):
                                        st.session_state.pop(_rk3, None)
                                    st.rerun()
                        with _bs3:
                            if st.button("Entfernen", key=f"del_{pname}_{_pi}",
                                         use_container_width=True):
                                positions.pop(_pi)
                                save_portfolio(port_data)
                                st.session_state[_pf_open_key] = None
                                st.rerun()

                        st.markdown('</div>', unsafe_allow_html=True)


            # ── Portfolio-Analyse ─────────────────────────────────────────────
            _ana_c, _ = st.columns([2, 6])
            with _ana_c:
                if st.button("Portfolio analysieren",
                              key=f"btn_ana_{pname}",
                              use_container_width=True):
                    st.session_state.pop(f"pf_analysis_{pname}", None)
                    with st.spinner(
                            f"Lade Marktdaten für {len(positions)} Positionen…"):
                        _ametas = {}
                        for _ap in positions:
                            _atk = (_ap.get("ticker") or "").upper().strip()
                            if _atk:
                                _ametas[_atk] = fetch_position_meta(_atk)
                    # Positionen mit berechneten Werten für Scoring
                    _apos = [{**p, **calc_position_derived(p)} for p in positions]
                    _ascores = score_portfolio(_apos, pname, _ametas)
                    _atotcv  = sum(calc_position_derived(p)["current_value"]
                                   for p in positions)
                    _atotinv = sum(calc_position_derived(p)["invested"]
                                   for p in positions)
                    _apfret  = ((_atotcv - _atotinv) / _atotinv * 100
                                if _atotinv > 0 else 0)
                    with st.spinner("Lade Benchmark…"):
                        _abench = fetch_benchmark_return("IWDA.AS", "1y")
                    _anar = ""
                    if _api_key_pf and OPENAI_AVAILABLE:
                        with st.spinner("Ace analysiert…"):
                            # Prompt je nach Portfolio-Typ
                            _is_core = "Compounder" in pname or "Core" in pname
                            _is_hc   = "Champion" in pname or "HC" in pname
                            _use_ace = (_is_core or _is_hc)
                            if _use_ace:
                                import json as _json3
                                _pjson = {
                                    "portfolio_value": round(_atotcv, 2),
                                    "invested_total": round(_atotinv, 2),
                                    "performance_pct": round(_apfret, 1),
                                    "positions": [
                                        {
                                            "name": p.get("name") or p.get("ticker",""),
                                            "ticker": p.get("ticker",""),
                                            "value": round(calc_position_derived(p)["current_value"] or 0, 2),
                                            "weight_pct": round(calc_position_derived(p)["current_value"] / _atotcv * 100 if _atotcv > 0 else 0, 1),
                                            "performance_pct": round(calc_position_derived(p)["pl_pct"] or 0, 1),
                                            "asset_type": "etf" if any(k in (p.get("name","")).upper() for k in ["ETF","ISHARES","VANGUARD","XTRACKERS","AMUNDI","SPDR"]) else "stock",
                                            "sector": _ametas.get(p.get("ticker",""), {}).get("sector",""),
                                        }
                                        for p in positions if p.get("name") or p.get("ticker")
                                    ]
                                }
                                # Prompt je nach Portfolio-Typ
                                _sel_prompt = (ACE_HC_PROMPT if _is_hc else ACE_CORE_PROMPT)
                                try:
                                    _ac2 = OpenAI(api_key=_api_key_pf)
                                    _filled = _sel_prompt.replace(
                                        "{{portfolio_json}}",
                                        _json3.dumps(_pjson, ensure_ascii=False, indent=2))
                                    _cr = _ac2.responses.create(
                                        model="gpt-4.1-mini",
                                        input=[{"role": "user", "content": _filled}])
                                    _anar = (getattr(_cr, "output_text", "") or "").strip()
                                except Exception:
                                    _anar = generate_portfolio_narrative(
                                        pname, _ascores, _apos, _ametas,
                                        _abench, _apfret, _api_key_pf)
                            else:
                                _anar = generate_portfolio_narrative(
                                    pname, _ascores, _apos, _ametas,
                                    _abench, _apfret, _api_key_pf)
                    st.session_state[f"pf_analysis_{pname}"] = {
                        "scores": _ascores, "metas": _ametas,
                        "bench": _abench, "pf_ret": _apfret,
                        "narrative": _anar,
                    }

            _ana = st.session_state.get(f"pf_analysis_{pname}")
            if _ana:
                _asc = _ana["scores"]
                _aben = _ana["bench"]
                _aret = _ana["pf_ret"]
                _anar2 = _ana["narrative"]

                # ── Premium Kennzahlen-Block ──────────────────────────────────
                def _sc_col(s):
                    return ("#10b981" if s >= 7 else "#f59e0b" if s >= 5 else "#ef4444")
                def _sc_bar(s, color):
                    pct = int(s * 10)
                    return (f'<div style="height:3px;background:rgba(128,128,128,0.12);'
                            f'border-radius:2px;margin-top:0.3rem;overflow:hidden;">'
                            f'<div style="width:{pct}%;height:100%;background:{color};'
                            f'border-radius:2px;"></div></div>')

                _abstr  = f"{_aben:+.1f}%" if _aben is not None else "—"
                _apstr  = f"{_aret:+.1f}%"
                _apc    = "#10b981" if _aret >= 0 else "#ef4444"
                _abc    = "#10b981" if (_aben or 0) >= 0 else "#ef4444"
                _vs_bm  = ((_aret - (_aben or 0)) if _aben is not None else None)
                _vs_str = (f"{_vs_bm:+.1f}%" if _vs_bm is not None else "—")
                _vs_col = "#10b981" if (_vs_bm or 0) >= 0 else "#ef4444"

                # Zeile 1: Scores
                _score_metrics = [
                    ("Gesamt-Score",      _asc.get("total_score",  0), True),
                    ("Ausgewogenheit",     _asc.get("balance_score",0), True),
                    ("Stabilität",        _asc.get("stab_score",   0), True),
                    ("Wachstumspotenzial", _asc.get("growth_score", 0), True),
                ]
                # Zeile 2: Markt-Kennzahlen
                _market_metrics = [
                    ("Rendite Portfolio", _apstr,   _apc,   False),
                    ("MSCI World (1J)",   _abstr,   _abc,   False),
                    ("vs. Benchmark",     _vs_str,  _vs_col,False),
                    ("Ø Beta",            f"{_asc.get('avg_beta',0):.2f}", "#888", False),
                    ("Defensiv-Anteil",   f"{_asc.get('def_weight',0)*100:.0f}%", "#888", False),
                    ("ETF-Anteil",        f"{_asc.get('etf_weight',0)*100:.0f}%", "#888", False),
                ]

                _m_html = (
                    '<div style="background:var(--secondary-background-color);'
                    'border:1px solid rgba(128,128,128,0.12);border-radius:14px;'
                    'padding:1rem 1.1rem;margin:0.8rem 0;">'
                    # Score-Zeile
                    '<div style="display:grid;grid-template-columns:repeat(4,1fr);'
                    'gap:0.8rem;margin-bottom:0.8rem;padding-bottom:0.8rem;'
                    'border-bottom:1px solid rgba(128,128,128,0.08);">'
                )
                for _ml, _mv, _is_score in _score_metrics:
                    _mc = _sc_col(_mv) if _is_score else "#888"
                    _m_html += (
                        f'<div>'
                        f'<div style="font-size:0.58rem;text-transform:uppercase;'
                        f'letter-spacing:0.1em;color:var(--text-color);opacity:0.38;'
                        f'margin-bottom:0.2rem;">{_ml}</div>'
                        f'<div style="font-size:1.35rem;font-weight:800;color:{_mc};'
                        f'line-height:1;">{_mv if not _is_score else f"{_mv:.1f}"}</div>'
                        + _sc_bar(_mv, _mc)
                        + f'</div>'
                    )
                _m_html += '</div>'

                # Markt-Kennzahlen-Zeile
                _m_html += (
                    '<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:0.6rem;">'
                )
                for _ml, _mv, _mc, _ in _market_metrics:
                    _m_html += (
                        f'<div style="text-align:center;">'
                        f'<div style="font-size:0.56rem;text-transform:uppercase;'
                        f'letter-spacing:0.08em;color:var(--text-color);opacity:0.35;'
                        f'margin-bottom:0.2rem;">{_ml}</div>'
                        f'<div style="font-size:0.92rem;font-weight:700;color:{_mc};">'
                        f'{_mv}</div></div>'
                    )
                _m_html += '</div></div>'
                st.markdown(_m_html, unsafe_allow_html=True)

                if _asc.get("sectors"):
                    st.markdown("<div style='height:0.8rem'></div>",
                                unsafe_allow_html=True)
                    _slist = sorted(_asc["sectors"].items(), key=lambda x: -x[1])
                    _shtml = ('<div class="ace-card"><div class="ace-score-lbl" '
                              'style="margin-bottom:0.5rem;">Sektor-Verteilung</div>'
                              '<div style="display:flex;flex-wrap:wrap;gap:0.4rem;">')
                    for _sn, _sw in _slist:
                        _sp = _sw * 100
                        # Einheitliches Design — Größe durch Schriftgröße + Opacity
                        _s_opacity = "1" if _sp >= 15 else "0.65" if _sp >= 5 else "0.45"
                        _s_fw = "700" if _sp >= 15 else "500"
                        _shtml += (
                            f'<span style="background:var(--secondary-background-color);'
                            f'border:1px solid rgba(128,128,128,0.18);'
                            f'border-radius:20px;padding:0.25rem 0.75rem;'
                            f'font-size:0.78rem;font-weight:{_s_fw};'
                            f'color:var(--text-color);opacity:{_s_opacity};">'
                            f'{_sn} {_sp:.0f}%</span>')
                    _shtml += '</div></div>'
                    st.markdown(_shtml, unsafe_allow_html=True)

                if _asc.get("fit_notes"):
                    _fhtml = ('<div class="ace-card" style="border-left:'
                               '3px solid #F5A623;"><div class="ace-score-lbl" '
                               f'style="margin-bottom:0.4rem;">Profil-Hinweise '
                               f'— {pname}</div>')
                    for _fn in _asc["fit_notes"]:
                        _fhtml += (f'<div class="trig-wait" '
                                   f'style="margin-bottom:0.3rem;">{_fn}</div>')
                    _fhtml += '</div>'
                    st.markdown(_fhtml, unsafe_allow_html=True)

                if _anar2:
                    import re as _re2
                    def _render_pf_md(text):
                        parts = []
                        for line in text.split('\n'):
                            line = line.strip()
                            if line.startswith('### '):
                                title = line[4:]
                                parts.append(
                                    f'<div style="font-size:0.62rem;font-weight:700;'
                                    f'letter-spacing:0.14em;text-transform:uppercase;'
                                    f'color:#10b981;margin:1rem 0 0.4rem 0;">{title}</div>')
                            elif line.startswith('- '):
                                c = _re2.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line[2:])
                                parts.append(
                                    f'<div style="display:flex;gap:0.5rem;margin-bottom:0.3rem;">'
                                    f'<span style="color:#10b981;flex-shrink:0;">›</span>'
                                    f'<span>{c}</span></div>')
                            elif line:
                                c = _re2.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
                                parts.append(f'<div style="margin-bottom:0.4rem;line-height:1.65;">{c}</div>')
                            else:
                                parts.append('<div style="height:0.3rem;"></div>')
                        return "".join(parts)

                    _is_core_pf = "Compounder" in pname or "Core" in pname
                    _is_hc_pf   = "Champion" in pname or "HC" in pname
                    _ana_label  = ("Ace · Core-Analyse" if _is_core_pf else
                                   "Ace · Hidden Champions-Analyse" if _is_hc_pf else
                                   "Ace · Portfolio-Analyse")
                    _collapse_key = f"pf_ana_collapsed_{pname}"
                    if _collapse_key not in st.session_state:
                        st.session_state[_collapse_key] = False
                    _is_collapsed = st.session_state[_collapse_key]

                    # Header mit Toggle-Button
                    _ach1, _ach2 = st.columns([8, 1])
                    with _ach1:
                        st.markdown(
                            f'<div style="font-size:0.6rem;letter-spacing:0.14em;'
                            f'text-transform:uppercase;color:#10b981;'
                            f'margin-top:0.8rem;padding-bottom:0.4rem;">'
                            f'{_ana_label}</div>',
                            unsafe_allow_html=True)
                    with _ach2:
                        st.markdown('<div style="padding-top:0.6rem;"></div>',
                                    unsafe_allow_html=True)
                        if st.button("−" if not _is_collapsed else "＋",
                                     key=f"pf_collapse_{pname}",
                                     use_container_width=True,
                                     help="Text ein-/ausklappen"):
                            st.session_state[_collapse_key] = not _is_collapsed
                            st.rerun()

                    if not _is_collapsed:
                        st.markdown(
                            f'<div style="background:var(--secondary-background-color);'
                            f'border:1px solid rgba(16,185,129,0.18);border-radius:14px;'
                            f'padding:1.2rem 1.4rem;">'
                            f'<div style="font-size:0.88rem;color:var(--text-color);">'
                            f'{_render_pf_md(_anar2)}'
                            f'</div>'
                            f'<div style="margin-top:1rem;padding-top:0.6rem;'
                            f'border-top:1px solid rgba(128,128,128,0.1);'
                            f'font-size:0.65rem;color:var(--text-color);opacity:0.3;">'
                            f'Keine Anlageberatung. Nur zu Informationszwecken.</div>'
                            f'</div>',
                            unsafe_allow_html=True)

                    # Next Steps (nur Core + HC)
                    if (_is_core_pf or _is_hc_pf) and _api_key_pf:
                        _ns_key = f"pf_next_steps_{pname}"
                        if st.button("Ideen für dein Portfolio",
                                     key=f"btn_ns_{pname}",
                                     use_container_width=False):
                            with st.spinner("Ace erarbeitet Next Steps…"):
                                _steps = generate_next_steps(_anar2, _api_key_pf)
                                st.session_state[_ns_key] = _steps
                        _steps = st.session_state.get(_ns_key, [])
                        if _steps:
                            render_next_steps(_steps, is_hc=_is_hc_pf,
                                             api_key=_api_key_pf, pname=pname)
                elif not _api_key_pf:
                    st.caption(
                        "OpenAI API-Key fehlt — AI-Analyse nicht verfügbar.")

                st.markdown("---")

        st.divider()


# ── Rechtlicher Hinweis ───────────────────────────────────────────────────────
st.markdown(
    '<div style="margin-top:2rem;padding:0.8rem 1.2rem;'
    'background:rgba(128,128,128,0.05);border-radius:10px;'
    'border:1px solid rgba(128,128,128,0.1);">'
    '<div style="font-size:0.65rem;color:var(--text-color);opacity:0.4;line-height:1.7;">'
    '<strong style="opacity:0.7;">Rechtlicher Hinweis:</strong> '
    'Velox ist ein reines Informations- und Portfolioverwaltungstool. '
    'Alle Inhalte, Analysen, Scores und Darstellungen dienen ausschließlich '
    'zu Informationszwecken und stellen <strong>keine Anlageberatung, '
    'keine Anlageempfehlung und keine Aufforderung zum Kauf oder Verkauf '
    'von Wertpapieren</strong> dar. '
    'Investitionen in Wertpapiere sind mit Risiken verbunden — '
    'bis hin zum Totalverlust. Vergangene Wertentwicklungen sind kein '
    'verlässlicher Indikator für zukünftige Ergebnisse. '
    'Bitte konsultiere bei Investitionsentscheidungen einen '
    'zugelassenen Finanzberater.'
    '</div></div>',
    unsafe_allow_html=True)
