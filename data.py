"""
価格データ層 — 取得とローカルキャッシュ
------------------------------------------------------------
yfinance から JPY建ての日足OHLCVを取り、cache/ にCSVで保存する。
2回目以降はキャッシュを読むので、同じデータで何度でも再現性のある検証ができる。

注意: Yahoo の JPY建て価格は実質「USD建て × 為替」の合成に近く、国内取引所
（bitFlyer / bitbank 等）の実際の板とは乖離する。将来 ccxt 等に差し替えられる
よう、外部が触るのは fetch() / load_panel() の2つだけにしてある。
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")

# 監視ユニバース（JPY建て直接 → 為替換算が不要）
SYMBOLS = {
    "BTC-JPY": "ビットコイン",
    "ETH-JPY": "イーサリアム",
    "XRP-JPY": "リップル",
    "SOL-JPY": "ソラナ",
}

CACHE_MAX_AGE_HOURS = 12  # これより古いキャッシュは取り直す


def _cache_path(symbol: str) -> str:
    return os.path.join(CACHE_DIR, f"{symbol}_1d.csv")


def _cache_is_fresh(path: str) -> bool:
    if not os.path.exists(path):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    return age < timedelta(hours=CACHE_MAX_AGE_HOURS)


def fetch(symbol: str, force: bool = False) -> pd.DataFrame:
    """
    日足OHLCVを返す。index は tz なしの日付、列は Open/High/Low/Close/Volume。
    キャッシュが新しければ通信しない。
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(symbol)

    if not force and _cache_is_fresh(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df

    hist = yf.Ticker(symbol).history(period="max", interval="1d", auto_adjust=False)
    if hist.empty:
        if os.path.exists(path):  # 通信に失敗しても古いキャッシュがあれば使う
            print(f"  [警告] {symbol} の取得に失敗。既存キャッシュを使用します。")
            return pd.read_csv(path, index_col=0, parse_dates=True)
        raise RuntimeError(f"{symbol} のデータを取得できませんでした")

    df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_csv(path)
    return df


def load_panel(symbols: list[str] | None = None,
               start: str | None = None,
               end: str | None = None,
               force: bool = False) -> dict[str, pd.DataFrame]:
    """
    全銘柄を共通の日付インデックスに揃えた {"open","high","low","close"} を返す。
    各 DataFrame は index=日付 / columns=銘柄。
    上場前などデータが無い箇所は NaN のまま（＝その日は売買不可として扱う）。
    """
    symbols = symbols or list(SYMBOLS)
    frames = {s: fetch(s, force=force) for s in symbols}

    index = None
    for df in frames.values():
        index = df.index if index is None else index.union(df.index)
    if start:
        index = index[index >= pd.Timestamp(start)]
    if end:
        index = index[index <= pd.Timestamp(end)]

    panel = {}
    for field in ("Open", "High", "Low", "Close"):
        panel[field.lower()] = pd.DataFrame(
            {s: frames[s][field].reindex(index) for s in symbols}, index=index
        )

    # 直近に抜けがあるなら、配信側が後から埋めている可能性がある。
    # キャッシュの寿命（12時間）を待たずに1度だけ取り直す。
    if not force and find_gaps(panel, days=7) and _gap_retry_due():
        _mark_gap_retry()
        return load_panel(symbols, start, end, force=True)
    return panel


_GAP_RETRY_FILE = os.path.join(CACHE_DIR, ".gap_retry")
GAP_RETRY_INTERVAL = timedelta(hours=1)   # 取り直しはこの間隔より頻繁にしない


def _gap_retry_due() -> bool:
    if not os.path.exists(_GAP_RETRY_FILE):
        return True
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(_GAP_RETRY_FILE))
    return age > GAP_RETRY_INTERVAL


def _mark_gap_retry() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_GAP_RETRY_FILE, "w", encoding="utf-8") as f:
        f.write(datetime.now().isoformat())


def find_gaps(panel: dict, days: int = 30) -> list[str]:
    """
    直近 days 日のうち、日足が欠けている日付を返す。
    暗号資産は年中無休なので、抜けは配信側の不具合を意味する。
    指標の窓は「本数」で数えているため、抜けがあると集計期間が静かにずれる。
    黙って計算を続けるのが一番まずいので、呼び出し側で表に出すこと。
    """
    idx = panel["close"].index
    if len(idx) < 2:
        return []
    start = max(idx[0], idx[-1] - pd.Timedelta(days=days))
    expected = pd.date_range(start, idx[-1], freq="D")
    return [str(d.date()) for d in expected.difference(idx)]


def fetch_live(symbols: list[str] | None = None) -> dict[str, float]:
    """
    現在値（直近5分足の終値）を一括取得する。1リクエストで全銘柄ぶん。
    取れなかった銘柄はキーごと落とす（呼び出し側で「売買不可」として扱う）。
    """
    symbols = symbols or list(SYMBOLS)
    df = yf.download(" ".join(symbols), period="1d", interval="5m",
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        return {}
    close = df["Close"]
    out = {}
    for s in symbols:
        try:
            ser = close[s] if isinstance(close, pd.DataFrame) else close
            ser = ser.dropna()
            if len(ser):
                out[s] = float(ser.iloc[-1])
        except (KeyError, IndexError):
            continue
    return out


def main() -> None:
    """python data.py で最新データを取り直してサマリを表示する。"""
    for s in SYMBOLS:
        df = fetch(s, force=True)
        print(f"{s:<9} {len(df):>5}本  "
              f"{df.index[0].date()} 〜 {df.index[-1].date()}  "
              f"最新終値 {df['Close'].iloc[-1]:,.0f}円")
    print(f"\nキャッシュ先: {CACHE_DIR}")


if __name__ == "__main__":
    main()
