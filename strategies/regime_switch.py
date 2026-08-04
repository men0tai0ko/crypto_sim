"""
レジーム切替戦略 — 実運用（live_trade.py）と同じ判断を過去データで再現する
------------------------------------------------------------
live_trade.py は「相場付きを判定してから戦略を選ぶ」という作りになっているが、
その組み合わせ自体は過去検証を通していなかった。ここで同じ手順を再現し、
バックテストで検証できるようにする。

実運用と揃えているもの:
  - regime.classify() による強気/中立/弱気の判定（同じ関数を呼ぶ）
  - レジームごとの戦略とエクスポージャー上限（regime.PLAYBOOK）
  - 1銘柄あたりの上限 50%
  - ATRトレーリングストップ（建玉後の最高値から ATR×3）

実運用と違うところ（構造上どうしても揃わない点）:
  - 実運用のストップは5分ごとに現在値で判定する。ここは日足の終値でしか
    判定できないため、実際より逃げ遅れる方向にずれる（＝保守的な見積もり）。
"""

import regime as regime_mod

from .base import Strategy, Context, atr
from .trend import DonchianTrend, FilteredEqualWeight

MAX_WEIGHT = regime_mod.MAX_WEIGHT   # 実運用と同じ値を同じ場所から読む
ATR_N = 14
ATR_MULT = 3.0


class RegimeSwitching(Strategy):
    name = "レジーム切替"

    def __init__(self, max_weight: float = MAX_WEIGHT,
                 atr_mult: float = ATR_MULT, use_stop: bool = True,
                 cooldown: int = regime_mod.COOLDOWN_DAYS):
        self.max_weight = max_weight
        self.atr_mult = atr_mult
        self.use_stop = use_stop
        self.cooldown = cooldown      # ストップ後、この本数ぶん再エントリーを禁じる
        self.cool_until = {}
        self.subs = {
            "分散保有": FilteredEqualWeight(50),
            "ドンチャン20/10": DonchianTrend(entry=20, exit=10),
            "ドンチャン55/20": DonchianTrend(entry=55, exit=20),
        }
        self.warmup = regime_mod.TREND_MA + 2      # 200日線が必要
        self.peaks = {}          # 銘柄 -> 建玉後の最高終値
        self.last_regime = None

    def targets(self, ctx: Context) -> dict:
        if ctx.i < self.warmup:
            return {}
        closes = ctx.hist("close")
        reg = regime_mod.classify(closes)
        self.last_regime = reg["レジーム"]

        sub = self.subs[reg["戦略"]]
        # ドンチャン系は自分でピークを持つので、こちらの記録と同期させる
        if isinstance(sub, DonchianTrend):
            sub.state = {s: {"peak": p} for s, p in self.peaks.items()}
        raw = sub.targets(ctx)
        if isinstance(sub, DonchianTrend):
            self.peaks.update({s: v["peak"] for s, v in sub.state.items()})

        raw = {s: min(w, self.max_weight) for s, w in raw.items()}
        total = sum(raw.values())
        cap = reg["上限"]
        if total > cap and total > 0:
            raw = {s: w * cap / total for s, w in raw.items()}

        # ストップ直後の買い直しを禁じる期間（cooldown=0 なら無効）
        if self.cooldown:
            raw = {s: w for s, w in raw.items()
                   if ctx.i >= self.cool_until.get(s, -1)}

        if self.use_stop:
            raw = self._apply_stops(ctx, raw)

        # 手仕舞い済みの銘柄のピークは捨てる（残すと再エントリー直後に即ストップ）
        held = set(ctx.weights) | set(raw)
        self.peaks = {s: v for s, v in self.peaks.items() if s in held}
        return raw

    def _apply_stops(self, ctx: Context, raw: dict) -> dict:
        """実運用の _guard と同じトレーリングストップを終値ベースで適用する。"""
        closes, highs, lows = ctx.hist("close"), ctx.hist("high"), ctx.hist("low")
        weights = ctx.weights
        out = dict(raw)
        for sym in list(weights):
            if weights.get(sym, 0.0) <= 0.001:
                continue
            c = closes[sym].dropna()
            if c.empty:
                continue
            price = float(c.iloc[-1])
            self.peaks[sym] = max(self.peaks.get(sym, price), price)
            a = atr(highs[sym].dropna(), lows[sym].dropna(), c, ATR_N)
            if a != a or a <= 0:
                continue
            if price <= self.peaks[sym] - self.atr_mult * a:
                out[sym] = 0.0
                self.peaks.pop(sym, None)
                if self.cooldown:
                    self.cool_until[sym] = ctx.i + self.cooldown
        return out
