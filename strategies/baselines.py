"""
ベンチマーク戦略
------------------------------------------------------------
「凝った戦略」は必ずこれらと比べる。ここに勝てないなら作る意味がない。
"""

from .base import Strategy, Context, is_month_start


class BuyHoldBTC(Strategy):
    """BTCを初日に全力で買って、あとは何もしない。最重要のベンチマーク。"""
    name = "BTC買い持ち"

    def targets(self, ctx: Context) -> dict:
        return {"BTC-JPY": 1.0} if "BTC-JPY" in ctx.available else {}


class EqualWeight(Strategy):
    """上場済み全銘柄を均等保有し、月初にリバランス。"""
    name = "均等分散(月次)"

    def __init__(self):
        self._prev = None
        self._held = {}

    def targets(self, ctx: Context) -> dict:
        if is_month_start(self._prev, ctx.date):
            avail = ctx.available
            self._held = {s: 1.0 / len(avail) for s in avail} if avail else {}
        self._prev = ctx.date
        return self._held


class DCA(Strategy):
    """
    元手100万円をBTCへ12ヶ月かけて分割投入し、以後は一切売らない（積立の代用）。
    追加入金はしない前提なので「時間分散して入る」効果だけを見る。
    月初以外は現在ウェイトをそのまま返す＝売買を発生させない。
    """
    name = "分割投入(12ヶ月)"
    MONTHS = 12

    def __init__(self, capital: float):
        self.tranche = capital / self.MONTHS
        self._prev = None
        self._steps = 0

    def targets(self, ctx: Context) -> dict:
        if "BTC-JPY" not in ctx.available:
            return {}
        held = ctx.weights.get("BTC-JPY", 0.0)
        if is_month_start(self._prev, ctx.date) and self._steps < self.MONTHS:
            self._steps += 1
            self._prev = ctx.date
            add = self.tranche / ctx.equity if ctx.equity > 0 else 0.0
            return {"BTC-JPY": min(1.0, held + add)}
        self._prev = ctx.date
        return {"BTC-JPY": held}  # 売らない
