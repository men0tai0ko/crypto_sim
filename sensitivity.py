"""
パラメータ感度の検査
------------------------------------------------------------
「20日が良くて19日/21日が崩れるならそれは偶然」— README に書いた自分のルールを
実際に確かめるための道具。レジーム切替戦略の主要パラメータを1つずつ動かし、
結果が滑らかに変わるか、それとも特定の値でだけ跳ねるかを見る。

実行: python sensitivity.py
"""

import backtest as bt
import data as data_mod
import metrics as metrics_mod
import regime as regime_mod
from strategies.regime_switch import RegimeSwitching

PERIODS = [
    ("2018-2022", bt.IS_PERIOD[0], bt.IS_PERIOD[1]),
    ("2023-2026", bt.OOS_PERIOD[0], bt.OOS_PERIOD[1]),
]


def evaluate(factory, panel) -> dict:
    eq, exp, broker = bt.run(factory(), panel, bt.CAPITAL)
    return metrics_mod.summarize(eq, broker.trades, exp, bt.CAPITAL)


def sweep(title: str, setter, values, panels: dict) -> None:
    print(f"\n{title}")
    print(f"  {'値':<10}" + "".join(f"{lbl+' CAGR':>14}{'最大DD':>10}{'Calmar':>9}"
                                    for lbl, _, _ in PERIODS))
    base = setter(None)                       # 現在値を控えておく
    for v in values:
        setter(v)
        row = f"  {str(v):<10}"
        for lbl, _, _ in PERIODS:
            m = evaluate(RegimeSwitching, panels[lbl])
            mark = "*" if v == base else " "
            row += (f"{m['年率(CAGR)']*100:>13.1f}%{m['最大DD']*100:>9.1f}%"
                    f"{m['Calmar']:>8.2f}{mark}")
        print(row)
    setter(base)                              # 必ず元に戻す


def main() -> None:
    panels = {lbl: data_mod.load_panel(start=s, end=e) for lbl, s, e in PERIODS}
    print("=" * 96)
    print("レジーム切替戦略のパラメータ感度   * は現在の設定値")
    print("結果が値の変化に対して滑らかなら頑健、特定の値でだけ跳ねるなら偶然を拾っている")
    print("=" * 96)

    def set_ma(v):
        old = regime_mod.TREND_MA
        if v is not None:
            regime_mod.TREND_MA = v
        return old
    sweep("■ 大局トレンドの移動平均（日）", set_ma, [100, 150, 200, 250, 300], panels)

    def set_vol(v):
        old = regime_mod.HIGH_VOL
        if v is not None:
            regime_mod.HIGH_VOL = v
        return old
    sweep("■ 「荒れている」と見なす年率ボラ", set_vol, [0.6, 0.75, 0.9, 1.05, 1.2], panels)

    import strategies.regime_switch as rs

    def set_atr(v):
        old = rs.ATR_MULT
        if v is not None:
            rs.ATR_MULT = v
            RegimeSwitching.__init__.__defaults__ = (rs.MAX_WEIGHT, v, True)
        return old
    sweep("■ ATRトレーリングストップの幅（ATRの何倍）", set_atr,
          [2.0, 2.5, 3.0, 3.5, 4.0], panels)

    def set_mw(v):
        old = rs.MAX_WEIGHT
        if v is not None:
            rs.MAX_WEIGHT = v
            RegimeSwitching.__init__.__defaults__ = (v, rs.ATR_MULT, True)
        return old
    sweep("■ 1銘柄あたりの上限ウェイト", set_mw, [0.3, 0.4, 0.5, 0.6, 0.8], panels)


if __name__ == "__main__":
    main()
