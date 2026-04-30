"""Data fetching and preprocessing for portfolio optimization."""

from __future__ import annotations

import os
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import yfinance as yf

EQUITIES_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "equities.txt")
PRICE_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "price_cache")
CACHE_TTL_SECONDS = 86400   # 24 hours per-ticker cache lifetime

DOWNLOAD_CHUNK_SIZE  = 100  # tickers per yf.download call
DOWNLOAD_CHUNK_DELAY = 2.5  # seconds between chunks

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
    return get_large_ticker_universe(n)


def load_equity_universe_from_file(filepath: str = EQUITIES_FILE) -> list[str]:
    if not os.path.exists(filepath):
        return []
    with open(filepath, encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    return list(dict.fromkeys(tickers))


def get_large_ticker_universe(n: int = 1000) -> list[str]:
    file_tickers = load_equity_universe_from_file()
    if file_tickers:
        return file_tickers[:n]

    tickers: list[str] = []
    try:
        tickers = _merge_unique(tickers, _fetch_wikipedia_index(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ["Symbol"]))
    except Exception:
        pass

    if len(tickers) < n:
        try:
            tickers = _merge_unique(tickers, _fetch_wikipedia_index(
                "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
                ["Symbol", "Ticker symbol", "Ticker"]))
        except Exception:
            pass

    if len(tickers) < n:
        try:
            tickers = _merge_unique(tickers, _fetch_nasdaq_trader(n * 2))
        except Exception:
            pass

    return tickers[:n] if tickers else FALLBACK_SP500[:n]


# ─── Price fetching ───────────────────────────────────────────────────────────

def fetch_prices(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
) -> pd.DataFrame:
    """
    Download adjusted close prices with two-level caching.

    Cache strategy — per-ticker disk files (price_cache/<TICKER>_<dates>.pkl):
      - Each ticker is cached individually so a timeout on one ticker never
        invalidates cached data for the others.
      - On every call, tickers already on disk are loaded instantly; only the
        missing ones are downloaded (in chunks with rate-limit delays).
      - Files older than CACHE_TTL_SECONDS (24 h) are refreshed.

    Errors on individual tickers (timeouts, delisted, etc.) are silently
    skipped; the portfolio is built from whatever data is available.
    """
    if not tickers:
        raise ValueError("No tickers provided.")

    cache_key = (tuple(sorted(tickers)), start, end, period)

    # ── Level 1: in-memory ───────────────────────────────────────────────────
    if cache_key in _price_cache:
        return _price_cache[cache_key].copy()

    # ── Build yfinance kwargs & date tag for file names ───────────────────────
    kwargs: dict = dict(auto_adjust=True, progress=False)
    if period:
        kwargs["period"] = period
        date_tag = period
    else:
        if not start or not end:
            raise ValueError("Provide either 'period' or both 'start' and 'end'.")
        kwargs["start"] = start
        kwargs["end"]   = end
        date_tag = f"{start}_{end}"

    os.makedirs(PRICE_CACHE_DIR, exist_ok=True)

    # ── Level 2: per-ticker disk cache ────────────────────────────────────────
    cached_frames:      dict[str, pd.DataFrame] = {}
    tickers_to_download: list[str]              = []

    for ticker in tickers:
        df = _load_ticker_cache(ticker, date_tag)
        if df is not None:
            cached_frames[ticker] = df
        else:
            tickers_to_download.append(ticker)

    # ── Download missing tickers in chunks ────────────────────────────────────
    if tickers_to_download:
        newly = _download_chunked(tickers_to_download, kwargs)
        for col in newly.columns:
            series = newly[[col]].dropna(how="all")
            if not series.empty:
                _save_ticker_cache(col, date_tag, series)
                cached_frames[col] = series

    if not cached_frames:
        raise ValueError(
            "No price data returned. Check ticker symbols and date range."
        )

    prices = pd.concat(cached_frames.values(), axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]

    _price_cache[cache_key] = prices
    return prices.copy()


def preprocess_prices(prices: pd.DataFrame, min_data_points: int = 60) -> pd.DataFrame:
    """Remove invalid tickers and fill small calendar gaps."""
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
    return np.log(prices / prices.shift(1)).dropna()


def compute_statistics(
    returns: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    TRADING_DAYS = 252
    mu    = returns.mean().values * TRADING_DAYS
    sigma = returns.cov().values  * TRADING_DAYS
    sigma += 1e-6 * np.eye(len(mu))
    return mu, sigma, list(returns.columns)


# ─── Per-ticker disk cache ────────────────────────────────────────────────────

def _ticker_cache_path(ticker: str, date_tag: str) -> str:
    safe_ticker  = ticker.replace("-", "_").replace("/", "_")
    safe_date    = date_tag.replace("/", "-").replace(" ", "_")
    return os.path.join(PRICE_CACHE_DIR, f"{safe_ticker}_{safe_date}.pkl")


def _load_ticker_cache(ticker: str, date_tag: str) -> pd.DataFrame | None:
    path = _ticker_cache_path(ticker, date_tag)
    if not os.path.exists(path):
        return None
    if time.time() - os.path.getmtime(path) > CACHE_TTL_SECONDS:
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_ticker_cache(ticker: str, date_tag: str, df: pd.DataFrame) -> None:
    try:
        with open(_ticker_cache_path(ticker, date_tag), "wb") as f:
            pickle.dump(df, f)
    except Exception:
        pass  # best-effort


# ─── Chunked downloader ───────────────────────────────────────────────────────

def _download_chunked(tickers: list[str], kwargs: dict) -> pd.DataFrame:
    """Download tickers in chunks, sleeping between each to avoid rate limits."""
    chunks = [tickers[i: i + DOWNLOAD_CHUNK_SIZE]
              for i in range(0, len(tickers), DOWNLOAD_CHUNK_SIZE)]

    all_frames: list[pd.DataFrame] = []

    for idx, chunk in enumerate(chunks):
        if idx > 0:
            time.sleep(DOWNLOAD_CHUNK_DELAY)

        raw = _download_with_retry(chunk, kwargs)
        if raw.empty:
            continue

        prices = _extract_close(raw, chunk)
        if not prices.empty:
            all_frames.append(prices)

    if not all_frames:
        return pd.DataFrame()
    if len(all_frames) == 1:
        return all_frames[0]

    combined = pd.concat(all_frames, axis=1)
    return combined.loc[:, ~combined.columns.duplicated()]


def _download_with_retry(tickers: list[str], kwargs: dict, max_retries: int = 3) -> pd.DataFrame:
    """Download one chunk; retry on rate-limit errors; swallow all other errors."""
    for attempt in range(max_retries):
        try:
            raw = yf.download(tickers, **kwargs)
            if not raw.empty:
                return raw
        except Exception as exc:
            msg = str(exc).lower()
            if any(kw in msg for kw in ("rate", "too many", "429")):
                time.sleep(10 * (2 ** attempt))
                continue
            # Any other exception (network error, bad ticker, etc.) → skip chunk
            return pd.DataFrame()

        if attempt < max_retries - 1:
            time.sleep(DOWNLOAD_CHUNK_DELAY * (attempt + 1))

    return pd.DataFrame()


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _fetch_wikipedia_index(url: str, symbol_cols: list[str]) -> list[str]:
    tables = pd.read_html(url, header=0)
    for table in tables:
        for col in symbol_cols:
            if col in table.columns:
                raw = table[col].dropna().astype(str).tolist()
                return [t.replace(".", "-").upper().strip() for t in raw
                        if t and t.lower() != "nan"]
    raise ValueError(f"No recognised symbol column found at {url}.")


def _fetch_nasdaq_trader(max_tickers: int = 3000) -> list[str]:
    tickers: list[str] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    base = "https://ftp.nasdaqtrader.com/dynamic/SymDir"

    try:
        r = session.get(f"{base}/nasdaqlisted.txt", timeout=15)
        r.raise_for_status()
        for line in r.text.strip().splitlines()[1:]:
            parts = line.split("|")
            if len(parts) < 7:
                continue
            sym, _, _, test, fin, _, etf = parts[:7]
            sym = sym.strip()
            if test == "N" and fin == "N" and etf == "N" and _valid_equity_ticker(sym):
                tickers.append(sym)
    except Exception:
        pass

    try:
        r = session.get(f"{base}/otherlisted.txt", timeout=15)
        r.raise_for_status()
        for line in r.text.strip().splitlines()[1:]:
            parts = line.split("|")
            if len(parts) < 7:
                continue
            sym  = parts[0].strip()
            etf  = parts[4].strip()
            test = parts[6].strip()
            if test == "N" and etf == "N" and _valid_equity_ticker(sym):
                tickers.append(sym)
    except Exception:
        pass

    return tickers[:max_tickers]


def _valid_equity_ticker(t: str) -> bool:
    if not t or len(t) > 5:
        return False
    clean = t.replace("-", "")
    if not clean.isalpha():
        return False
    if "File" in t or "Creation" in t:
        return False
    return True


def _merge_unique(base: list[str], new: list[str]) -> list[str]:
    seen   = set(base)
    result = list(base)
    for t in new:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0).unique().tolist()
        col    = "Close" if "Close" in level0 else "Adj Close"
        prices = raw[col]
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0] if tickers else "price")
    else:
        col = "Close" if "Close" in raw.columns else "Adj Close"
        if col not in raw.columns:
            raise ValueError(f"Cannot find a price column. Available: {raw.columns.tolist()}")
        prices = raw[[col]].copy()
        prices.columns = [tickers[0]] if len(tickers) == 1 else prices.columns

    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.droplevel(0)

    return prices
