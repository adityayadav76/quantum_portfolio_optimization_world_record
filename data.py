"""Data fetching and preprocessing for portfolio optimization."""

from __future__ import annotations

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import yfinance as yf

EQUITIES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "equities.txt")

# ─── Built-in fallback universe (100 large-cap US stocks) ────────────────────
FALLBACK_SP500: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "LLY",
    "JPM", "V",    "AVGO", "XOM",  "PG",   "MA",   "HD",   "COST",  "MRK",  "ABBV",
    "CVX",  "PEP", "ORCL", "BAC",  "KO",   "ADBE", "CRM",  "AMD",   "TMO",  "ACN",
    "MCD",  "NFLX","ABT",  "WMT",  "LIN",  "DIS",  "TXN",  "PM",    "QCOM", "DHR",
    "VZ",   "INTU","AMGN", "IBM",  "CAT",  "GS",   "MS",   "BLK",   "SPGI", "GILD",
    "AXP",  "T",   "RTX",  "BA",   "GE",   "MMM",  "HON",  "UPS",   "DE",   "LOW",
    "ELV",  "CI",  "AMAT", "NOW",  "LRCX", "ADI",  "PLD",  "MO",    "REGN", "SYK",
    "BSX",  "ZTS", "ISRG", "VRTX", "BIIB", "IDXX", "NKE",  "SBUX",  "TGT",  "F",
    "GM",   "UBER","ABNB", "CRWD", "NET",  "DDOG", "SNOW", "SHOP",  "SQ",   "PYPL",
    "INTC", "CSCO","MU",   "KLAC", "MCHP", "SNPS", "CDNS", "ANSS",  "PANW", "ZS",
]

# In-memory price cache
_price_cache: dict = {}


# ─── Ticker universe ──────────────────────────────────────────────────────────

def get_sp500_tickers(n: int = 50) -> list[str]:
    """Alias kept for backward compatibility — delegates to get_large_ticker_universe."""
    return get_large_ticker_universe(n)


def load_equity_universe_from_file(filepath: str = EQUITIES_FILE) -> list[str]:
    """
    Load the pre-downloaded equity universe from equities.txt.

    Returns an empty list if the file does not exist.
    The file is written by  download_equities.py  (run once to populate it).
    """
    if not os.path.exists(filepath):
        return []
    with open(filepath, encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    # Explicit dedup preserving order (file should already be clean, but be safe)
    return list(dict.fromkeys(tickers))


def get_large_ticker_universe(n: int = 1000) -> list[str]:
    """
    Return up to n US equity tickers.

    Checks equities.txt first (populated by download_equities.py).
    Falls back to live Wikipedia + NASDAQ Trader fetching if the file is absent.
    Finally falls back to the built-in 100-ticker list.
    """
    # ── Fast path: pre-downloaded file ───────────────────────────────────────
    file_tickers = load_equity_universe_from_file()
    if file_tickers:
        return file_tickers[:n]

    # ── Slow path: live fetching (only if file missing) ───────────────────────
    tickers: list[str] = []

    # ── 1. S&P 500 ────────────────────────────────────────────────────────────
    try:
        sp500 = _fetch_wikipedia_index(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            ["Symbol"],
        )
        tickers = _merge_unique(tickers, sp500)
    except Exception:
        pass

    # ── 2. S&P MidCap 400 ────────────────────────────────────────────────────
    if len(tickers) < n:
        try:
            sp400 = _fetch_wikipedia_index(
                "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
                ["Symbol", "Ticker symbol", "Ticker"],   # try multiple column names
            )
            tickers = _merge_unique(tickers, sp400)
        except Exception:
            pass

    # ── 3. NASDAQ Trader listings ─────────────────────────────────────────────
    if len(tickers) < n:
        try:
            nasdaq = _fetch_nasdaq_trader(n * 2)
            tickers = _merge_unique(tickers, nasdaq)
        except Exception:
            pass

    # ── 4. Fallback ───────────────────────────────────────────────────────────
    if not tickers:
        return FALLBACK_SP500[:n]

    return tickers[:n]


# ─── Price fetching ───────────────────────────────────────────────────────────

def fetch_prices(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
) -> pd.DataFrame:
    """
    Download adjusted close prices via yfinance with in-memory caching.

    Args:
        tickers: list of ticker symbols
        start:   'YYYY-MM-DD' start date (used when period is None)
        end:     'YYYY-MM-DD' end   date (used when period is None)
        period:  one of '1y', '2y', '3y', '4y', '5y'

    Returns:
        DataFrame indexed by date, one column per valid ticker.
    """
    if not tickers:
        raise ValueError("No tickers provided.")

    cache_key = (tuple(sorted(tickers)), start, end, period)
    if cache_key in _price_cache:
        return _price_cache[cache_key].copy()

    kwargs: dict = dict(auto_adjust=True, progress=False)
    if period:
        kwargs["period"] = period
    else:
        if not start or not end:
            raise ValueError("Provide either 'period' or both 'start' and 'end'.")
        kwargs["start"] = start
        kwargs["end"] = end

    raw = _download_with_retry(tickers, kwargs)

    if raw.empty:
        raise ValueError(
            "No price data returned. Check ticker symbols and date range."
        )

    prices = _extract_close(raw, tickers)

    if prices.empty:
        raise ValueError("Could not extract price data from the downloaded result.")

    _price_cache[cache_key] = prices
    return prices.copy()


def preprocess_prices(prices: pd.DataFrame, min_data_points: int = 60) -> pd.DataFrame:
    """
    Remove invalid tickers and fill small calendar gaps.

    Drops all-NaN columns (dead/invalid tickers), forward-fills up to 5 gaps,
    then drops rows with any remaining NaN.
    """
    prices = prices.dropna(axis=1, how="all")
    if prices.empty:
        raise ValueError("All tickers returned empty data. Verify your ticker symbols.")

    prices = prices.ffill(limit=5).dropna()

    if len(prices) < min_data_points:
        raise ValueError(
            f"Only {len(prices)} clean trading days after preprocessing "
            f"(minimum required: {min_data_points}). "
            "Use a longer date range or replace tickers with shorter histories."
        )
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_statistics(
    returns: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Annualise mean returns and covariance matrix (252 trading days).

    A small Tikhonov regulariser (1e-6 · I) is added to sigma to guarantee
    positive-definiteness even for nearly collinear return series.
    """
    TRADING_DAYS = 252
    mu = returns.mean().values * TRADING_DAYS
    sigma = returns.cov().values * TRADING_DAYS
    sigma += 1e-6 * np.eye(len(mu))
    return mu, sigma, list(returns.columns)


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _fetch_wikipedia_index(url: str, symbol_cols: list[str]) -> list[str]:
    """Scrape a Wikipedia index table and return the ticker symbols."""
    tables = pd.read_html(url, header=0)
    for table in tables:
        for col in symbol_cols:
            if col in table.columns:
                raw = table[col].dropna().astype(str).tolist()
                return [t.replace(".", "-").upper().strip() for t in raw
                        if t and t.lower() != "nan"]
    raise ValueError(f"No recognised symbol column found at {url}. "
                     f"Tried: {symbol_cols}")


def _fetch_nasdaq_trader(max_tickers: int = 3000) -> list[str]:
    """
    Fetch US-listed equity tickers from NASDAQ's public symbol directory.

    Files:
    - nasdaqlisted.txt : NASDAQ-listed securities
    - otherlisted.txt  : NYSE / NYSE American / other exchange securities
    """
    tickers: list[str] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    base = "https://ftp.nasdaqtrader.com/dynamic/SymDir"

    # ── NASDAQ-listed: Symbol | ... | TestIssue(3) | FinStatus(4) | ... | ETF(6) ──
    try:
        r = session.get(f"{base}/nasdaqlisted.txt", timeout=15)
        r.raise_for_status()
        for line in r.text.strip().splitlines()[1:]:   # skip header
            parts = line.split("|")
            if len(parts) < 7:
                continue
            sym, _, _, test, fin, _, etf = parts[:7]
            sym = sym.strip()
            if test == "N" and fin == "N" and etf == "N" and _valid_equity_ticker(sym):
                tickers.append(sym)
    except Exception:
        pass

    # ── Other-listed: ACTSymbol(0) | ... | ETF(4) | ... | TestIssue(6) ──────────
    try:
        r = session.get(f"{base}/otherlisted.txt", timeout=15)
        r.raise_for_status()
        for line in r.text.strip().splitlines()[1:]:
            parts = line.split("|")
            if len(parts) < 7:
                continue
            sym   = parts[0].strip()
            etf   = parts[4].strip()
            test  = parts[6].strip()
            if test == "N" and etf == "N" and _valid_equity_ticker(sym):
                tickers.append(sym)
    except Exception:
        pass

    return tickers[:max_tickers]


def _valid_equity_ticker(t: str) -> bool:
    """Keep common equity tickers; reject warrants, units, and test symbols."""
    if not t or len(t) > 5:
        return False
    # Allow letters and the hyphen used by NYSE (e.g. BRK-B)
    clean = t.replace("-", "")
    if not clean.isalpha():
        return False
    # Skip symbols that look like last-line metadata
    if "File" in t or "Creation" in t:
        return False
    return True


def _merge_unique(base: list[str], new: list[str]) -> list[str]:
    """Append new tickers to base, skipping duplicates (preserves order)."""
    seen = set(base)
    result = list(base)
    for t in new:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extract the closing price column from a yfinance download result."""
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0).unique().tolist()
        col = "Close" if "Close" in level0 else "Adj Close"
        prices = raw[col]
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0] if tickers else "price")
    else:
        col = "Close" if "Close" in raw.columns else "Adj Close"
        if col not in raw.columns:
            raise ValueError(
                f"Cannot find a price column. Available: {raw.columns.tolist()}"
            )
        prices = raw[[col]].copy()
        prices.columns = [tickers[0]] if len(tickers) == 1 else prices.columns

    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.droplevel(0)

    return prices


def _download_with_retry(tickers: list[str], kwargs: dict, max_retries: int = 3) -> pd.DataFrame:
    """yf.download with exponential back-off on Yahoo Finance rate-limit errors."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    })

    for attempt in range(max_retries):
        try:
            raw = yf.download(tickers, session=session, **kwargs)
            if not raw.empty:
                return raw
        except Exception as exc:
            msg = str(exc).lower()
            if any(kw in msg for kw in ("rate", "too many", "429")):
                time.sleep(5 * (2 ** attempt))
                continue
            raise

        if attempt < max_retries - 1:
            time.sleep(5 * (attempt + 1))

    return yf.download(tickers, **kwargs)
