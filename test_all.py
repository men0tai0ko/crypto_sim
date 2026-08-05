"""
自動テスト — 壊れたら困る不変条件だけを検査する
------------------------------------------------------------
実行: python test_all.py

外部ライブラリ不要。ネットワークにも本番の state/ にも触らない。
「動くこと」ではなく「静かに間違わないこと」を守るのが目的。
"""

import json
import os
import sys
import tempfile
import traceback
from datetime import datetime, timedelta, timezone

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import backtest as bt
import broker as broker_mod
import regime as regime_mod
from broker import Broker
from strategies.base import Context, atr
from strategies.trend import DonchianTrend

_results = []


def test(name):
    def deco(fn):
        _results.append((name, fn))
        return fn
    return deco


# ---- テスト用の合成パネル ----

def make_panel(closes: dict, start="2020-01-01") -> dict:
    """{銘柄: [終値...]} から OHLC パネルを作る。高値・安値は終値から機械的に作る。"""
    n = len(next(iter(closes.values())))
    idx = pd.date_range(start, periods=n, freq="D")
    close = pd.DataFrame(closes, index=idx)
    return {
        "open": close.copy(),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
    }


# ============================================================
# 執行モデル
# ============================================================

@test("買いで手数料とスリッページが引かれる")
def _():
    b = Broker(cash=1_000_000)
    b._buy("d", "X", 100_000, 1000, "テスト")
    pos = b.positions["X"]
    assert b.cash == 900_000, b.cash
    # 約定単価は不利側（+0.10%）、数量は手数料0.15%を引いた額で買える
    expect_qty = 100_000 * (1 - 0.0015) / (1000 * 1.0010)
    assert abs(pos.qty - expect_qty) < 1e-9, (pos.qty, expect_qty)
    assert pos.cost_jpy == 100_000


@test("売りの実現損益が手数料込みで正しい")
def _():
    b = Broker(cash=1_000_000)
    b._buy("d", "X", 100_000, 1000, "買い")
    qty = b.positions["X"].qty
    b._sell("d", "X", qty, 1000, "同値で売却")
    # 同値でも往復コスト（手数料0.3%＋スリッページ0.2%）ぶん必ず負ける
    realized = b.trades[-1].realized_jpy
    assert realized < 0, realized
    assert abs(realized) < 100_000 * 0.006, realized
    assert "X" not in b.positions


@test("目標ウェイトの合計が1を超えたら正規化される")
def _():
    b = Broker(cash=1_000_000)
    b.rebalance("d", {"A": 100, "B": 100}, {"A": 0.8, "B": 0.8})
    eq = b.equity({"A": 100, "B": 100})
    assert b.cash >= -1e-6, f"現金がマイナス: {b.cash}"
    w = b.weights({"A": 100, "B": 100})
    assert sum(w.values()) <= 1.001, w


@test("現金を超えて買わない")
def _():
    b = Broker(cash=10_000)
    b._buy("d", "X", 1_000_000, 100, "過大な注文")
    assert b.cash >= 0, b.cash


@test("価格が欠損している銘柄は売買対象から外れる")
def _():
    b = Broker(cash=1_000_000)
    b.rebalance("d", {"A": float("nan"), "B": 100}, {"A": 0.5, "B": 0.5})
    assert "A" not in b.positions
    assert "B" in b.positions


# ============================================================
# バックテストの健全性
# ============================================================

@test("戦略は未来のバーを見られない")
def _():
    seen = []

    class Peeker:
        warmup = 0
        name = "覗き見テスト"

        def targets(self, ctx):
            seen.append(len(ctx.hist("close")))
            return {}

    panel = make_panel({"A": [100] * 10})
    bt.run(Peeker(), panel, 1_000_000)
    # i日目に見えるのは i+1 本ぶんだけ
    assert seen == list(range(1, 11)), seen


@test("執行は翌日の始値で行われる（当日終値ではない）")
def _():
    class BuyOnDay0:
        warmup = 0
        name = "初日に買う"

        def targets(self, ctx):
            return {"A": 1.0} if ctx.i == 0 else {"A": 1.0}

    # 終値は初日100、翌日から200。始値を終値と同じにしてある
    panel = make_panel({"A": [100, 200, 200, 200]})
    eq, exp, b = bt.run(BuyOnDay0(), panel, 1_000_000)
    # 初日終値で判断 → 2日目の始値200で約定。100で買えていたら look-ahead
    fill = b.trades[0].price
    assert fill > 150, f"当日終値で約定している疑い: {fill}"


@test("買い持ちの成績が価格変化と一致する")
def _():
    from strategies.baselines import BuyHoldBTC
    panel = make_panel({"BTC-JPY": [100, 100, 200, 200]})
    eq, exp, b = bt.run(BuyHoldBTC(), panel, 1_000_000)
    # 2日目始値100で買い、最終200 → 約2倍（往復コストは片道ぶんのみ）
    assert 1_950_000 < eq.iloc[-1] < 2_000_000, eq.iloc[-1]


# ============================================================
# ストップとクールダウン
# ============================================================

@test("ATRトレーリングストップが発動する")
def _():
    # 上げてから急落させる
    prices = [100 + i for i in range(40)] + [139 - i * 5 for i in range(10)]
    panel = make_panel({"A": prices})
    s = DonchianTrend(entry=20, exit=10)
    b = Broker(cash=1_000_000)
    b._buy("d", "A", 300_000, 100, "事前保有")
    s.state["A"] = {"peak": 139}
    ctx = Context(panel, len(prices) - 1, b, b.equity({"A": prices[-1]}))
    out = s.targets(ctx)
    assert out.get("A", 0.0) == 0.0, f"急落後も保有し続けている: {out}"


@test("クールダウン中は買いシグナルが出ても建てない")
def _():
    from strategies.regime_switch import RegimeSwitching
    s = RegimeSwitching(cooldown=10)
    s.cool_until["A"] = 100
    panel = make_panel({"A": [100] * 5})
    b = Broker(cash=1_000_000)

    class AlwaysBuy:
        def targets(self, ctx):
            return {"A": 0.5}
    s.subs["ドンチャン55/20"] = AlwaysBuy()
    s.warmup = 0
    orig = regime_mod.classify
    regime_mod.classify = lambda closes: {"レジーム": "弱気", "戦略": "ドンチャン55/20", "上限": 0.4}
    try:
        ctx = Context(panel, 4, b, 1_000_000)      # i=4 < cool_until=100
        assert s.targets(ctx) == {}, "クールダウンを無視して建てている"
        s.cool_until["A"] = 2                      # 期限切れ
        assert "A" in s.targets(Context(panel, 4, b, 1_000_000))
    finally:
        regime_mod.classify = orig


@test("現在値の反映で日足を1本余計に増やさない")
def _():
    """
    yfinanceの日足はUTC区切り。ローカル日付で「今日の足」を判定すると
    日本時間の00:00〜09:00に1日ずれ、実データの上に合成行が積まれる。
    そうなると前日比が「数分前との比較」になり、指標の集計窓も1本ずれる。
    """
    import live_trade as lt
    utc_today = datetime.now(timezone.utc).date()
    idx = pd.to_datetime([utc_today - timedelta(days=2),
                          utc_today - timedelta(days=1),
                          utc_today])
    close = pd.DataFrame({"BTC-JPY": [100.0, 110.0, 120.0]}, index=idx)
    panel = {"open": close.copy(), "high": close * 1.01,
             "low": close * 0.99, "close": close.copy()}

    trader = lt.LiveTrader.__new__(lt.LiveTrader)     # __init__ は状態を読むので通さない
    trader.panel = panel
    out = trader._panel_now({"BTC-JPY": 125.0})

    assert len(out["close"]) == 3, f"行が増えている: {len(out['close'])}本"
    assert float(out["close"].iloc[-1, 0]) == 125.0, "現在値が反映されていない"
    # 前日比が「前日の終値」に対して計算されること
    prev = float(out["close"].iloc[-2, 0])
    assert prev == 110.0, f"直前の行が前日でない: {prev}"


@test("UTCの日付が変わったら新しい足を足す")
def _():
    import live_trade as lt
    utc_today = datetime.now(timezone.utc).date()
    idx = pd.to_datetime([utc_today - timedelta(days=2), utc_today - timedelta(days=1)])
    close = pd.DataFrame({"BTC-JPY": [100.0, 110.0]}, index=idx)
    panel = {"open": close.copy(), "high": close * 1.01,
             "low": close * 0.99, "close": close.copy()}
    trader = lt.LiveTrader.__new__(lt.LiveTrader)
    trader.panel = panel
    out = trader._panel_now({"BTC-JPY": 125.0})
    assert len(out["close"]) == 3, "新しい日の足が追加されていない"
    assert float(out["close"].iloc[-2, 0]) == 110.0


# ============================================================
# 設定の一貫性（実運用とバックテストのズレ防止）
# ============================================================

@test("実運用とバックテストが同じ設定値を使っている")
def _():
    import live_trade as lt
    import strategies.regime_switch as rs
    assert lt.MAX_WEIGHT == regime_mod.MAX_WEIGHT == rs.MAX_WEIGHT, "MAX_WEIGHTがズレている"
    assert lt.COOLDOWN_DAYS == regime_mod.COOLDOWN_DAYS, "COOLDOWN_DAYSがズレている"
    assert rs.RegimeSwitching().cooldown == regime_mod.COOLDOWN_DAYS
    assert lt.ATR_N == regime_mod.ATR_N == rs.ATR_N, "ATR_Nがズレている"
    assert lt.ATR_MULT == regime_mod.ATR_MULT == rs.ATR_MULT, "ATR_MULTがズレている"
    assert rs.RegimeSwitching().atr_mult == regime_mod.ATR_MULT


@test("実運用とバックテストが同じプレイブックを持っている")
def _():
    import live_trade as lt
    from strategies.regime_switch import RegimeSwitching
    live_keys = set(lt.PLAYBOOK)
    bt_keys = set(RegimeSwitching().subs)
    playbook = {v[0] for v in regime_mod.PLAYBOOK.values()}
    assert live_keys == bt_keys == playbook, (live_keys, bt_keys, playbook)


# ============================================================
# レジーム判定
# ============================================================

@test("200日線の下なら必ず弱気")
def _():
    closes = pd.DataFrame({"BTC-JPY": [200] * 200 + [100]})
    closes.index = pd.date_range("2020-01-01", periods=201, freq="D")
    assert regime_mod.classify(closes)["レジーム"] == "弱気"


@test("200日線の上・低ボラ・全銘柄上昇なら強気")
def _():
    n = 260
    rise = [100 + i for i in range(n)]
    closes = pd.DataFrame({"BTC-JPY": rise, "ETH-JPY": rise})
    closes.index = pd.date_range("2020-01-01", periods=n, freq="D")
    assert regime_mod.classify(closes)["レジーム"] == "強気"


# ============================================================
# 二重起動の防止
# ============================================================

@test("生きているプロセスのロックは弾き、死んだPIDのロックは無視する")
def _():
    import json
    import live_trade as lt
    tmp = tempfile.mkdtemp()
    saved = lt.LOCK_FILE
    lt.LOCK_FILE = os.path.join(tmp, "l.lock")
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        json.dump({"pid": os.getpid(), "interval": 300, "heartbeat": now},
                  open(lt.LOCK_FILE, "w"))
        assert lt.read_lock() is not None, "稼働中のロックを見落としている"
        json.dump({"pid": 999_999, "interval": 300, "heartbeat": now},
                  open(lt.LOCK_FILE, "w"))
        assert lt.read_lock() is None, "死んだPIDのロックで起動を止めている"
        old = (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
        json.dump({"pid": os.getpid(), "interval": 300, "heartbeat": old},
                  open(lt.LOCK_FILE, "w"))
        assert lt.read_lock() is None, "心拍が古いロックで起動を止めている"
    finally:
        lt.LOCK_FILE = saved


@test("Windowsでも生存確認がプロセスを殺さない")
def _():
    import live_trade as lt
    assert lt.pid_alive(os.getpid()) is True
    assert lt.pid_alive(999_999) is False
    assert lt.pid_alive(os.getpid()) is True      # 1回目で死んでいないこと


# ============================================================
# 状態ファイルの壊れ耐性
# ============================================================

@test("壊れた状態ファイルでも例外を投げず、新規状態にフォールバックする")
def _():
    """
    _load_state() は main() のリトライ機構(errors/log_error)の外側、
    __init__ から直接呼ばれる。ここで例外を投げると無言でプロセスごと落ち、
    しかもCIでは同じ壊れたファイルを毎回 .prev から復元するので、
    手動で介入するまで永久に失敗し続ける。
    """
    import live_trade as lt
    tmp = tempfile.mkdtemp()
    saved = (lt.STATE_DIR, lt.STATE_FILE)
    lt.STATE_DIR = tmp
    lt.STATE_FILE = os.path.join(tmp, "broken.json")
    try:
        with open(lt.STATE_FILE, "w", encoding="utf-8") as f:
            f.write("{ this is not valid json ")
        trader = lt.LiveTrader()          # 例外を投げないこと
        assert trader.broker.cash == lt.CAPITAL
        assert trader.broker.positions == {}
    finally:
        lt.STATE_DIR, lt.STATE_FILE = saved


@test("brokerキーが欠けた状態ファイルでも例外を投げない")
def _():
    import live_trade as lt
    tmp = tempfile.mkdtemp()
    saved = (lt.STATE_DIR, lt.STATE_FILE)
    lt.STATE_DIR = tmp
    lt.STATE_FILE = os.path.join(tmp, "s.json")
    try:
        with open(lt.STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"peaks": {}}, f)   # "broker" キーが無い
        trader = lt.LiveTrader()
        assert trader.broker.cash == lt.CAPITAL
    finally:
        lt.STATE_DIR, lt.STATE_FILE = saved


# ============================================================
# 指標
# ============================================================

@test("最大ドローダウンの計算が正しい")
def _():
    import metrics
    eq = pd.Series([100, 200, 100, 150],
                   index=pd.date_range("2020-01-01", periods=4, freq="D"))
    dd, _ = metrics.max_drawdown(eq)
    assert abs(dd - (-0.5)) < 1e-9, dd


@test("データ欠損を検知する")
def _():
    import data
    idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-04"])
    panel = {"close": pd.DataFrame({"A": [1, 2, 3]}, index=idx)}
    assert data.find_gaps(panel, days=30) == ["2026-01-03"]


def main() -> int:
    print(f"テスト {len(_results)}件\n")
    failed = []
    for name, fn in _results:
        try:
            fn()
            print(f"  OK   {name}")
        except Exception as exc:
            failed.append((name, exc))
            print(f"  NG   {name}")
            print("       " + str(exc).replace("\n", "\n       "))
    print()
    if failed:
        print(f"失敗 {len(failed)}件 / 全{len(_results)}件")
        for name, exc in failed:
            print(f"\n--- {name} ---")
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        return 1
    print(f"すべて成功 ({len(_results)}件)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
