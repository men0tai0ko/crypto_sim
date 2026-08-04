"""
結果の作図
------------------------------------------------------------
実行:
  python plot.py            # results/ にPNGを3枚出力（ライト配色）
  python plot.py --dark     # ダーク配色で出力

出力:
  fig1_equity.png      資産曲線（作り込み用 / 検証用の2面・対数軸）
  fig2_drawdown.png    ドローダウン（3戦略に絞る。6本重ねると読めない）
  fig3_summary.png     期間をまたいだ比較。左=買い持ちに対する年率の差 / 右=最大DD
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

import backtest as bt

RESULTS_DIR = bt.RESULTS_DIR
CAPITAL = bt.CAPITAL
BENCH = "BTC買い持ち"

# 資産曲線に載せる6戦略（隣接ペアの配色検証済み）
LINE_SERIES = [BENCH, "均等分散(月次)", "ドンチャン55/20",
               "移動平均20/60", "モメンタム上位2", "RSI逆張り"]
# ドローダウンは重ね書きが効かないので3本まで（全ペアの配色検証済み）
DD_SERIES = [BENCH, "均等分散(月次)", "ドンチャン55/20"]

THEME = {
    "light": {
        "surface": "#fcfcfb", "page": "#f9f9f7",
        "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
    },
    "dark": {
        "surface": "#1a1a19", "page": "#0d0d0d",
        "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
    },
}

PERIODS = [
    ("作り込み用 2018-2022（in-sample）", bt.IS_PERIOD[0], bt.IS_PERIOD[1]),
    ("検証用 2023-2026（out-of-sample）", bt.OOS_PERIOD[0], bt.OOS_PERIOD[1]),
]


def setup(theme: str) -> dict:
    t = THEME[theme]
    plt.rcParams.update({
        "font.family": "Yu Gothic",
        "axes.unicode_minus": False,
        "figure.facecolor": t["page"],
        "axes.facecolor": t["surface"],
        "savefig.facecolor": t["page"],
        "text.color": t["ink"],
        "axes.labelcolor": t["ink2"],
        "xtick.color": t["muted"],
        "ytick.color": t["muted"],
        "font.size": 10,
    })
    return t


def style_axes(ax, t: dict, grid_axis: str = "y") -> None:
    """罫線と枠は控えめに。データより目立たせない。"""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def year_ticks(ax, last) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xticks([d for d in ax.get_xticks() if d <= mdates.date2num(last)])


def spread_labels(ax, points: list, min_gap_px: float = 14.0) -> dict:
    """
    右端ラベルが重ならないよう、表示座標で最小間隔を確保したオフセット(pt)を返す。
    points: [(key, y_value), ...]
    """
    ax.figure.canvas.draw()
    px = {k: ax.transData.transform((0, y))[1] for k, y in points}
    order = sorted(px, key=lambda k: -px[k])
    moved = dict(px)
    for prev, cur in zip(order, order[1:]):
        if moved[prev] - moved[cur] < min_gap_px:
            moved[cur] = moved[prev] - min_gap_px
    scale = 72.0 / ax.figure.dpi
    return {k: (moved[k] - px[k]) * scale for k in px}


def fig_equity(all_curves: dict, t: dict, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))
    colors = dict(zip(LINE_SERIES, t["series"]))

    for ax, (label, _, _) in zip(axes, PERIODS):
        curves = {k: all_curves[label][k] for k in LINE_SERIES}
        for name, s in curves.items():
            ax.plot(s.index, s.values, color=colors[name], linewidth=2.0,
                    solid_capstyle="round")
        ax.axhline(CAPITAL, color=t["muted"], linewidth=1.0, linestyle=(0, (4, 4)))
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/10_000:,.0f}万"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.set_title(label, color=t["ink"], fontsize=11.5, loc="left", pad=10)
        style_axes(ax, t)

        # 右側にラベル用の余白を確保してから、線の右端に戦略名を直接置く。
        # 薄い系列は背景とのコントラストが3:1未満なので凡例任せにしない。
        first = next(iter(curves.values()))
        span = first.index[-1] - first.index[0]
        ax.set_xlim(first.index[0], first.index[-1] + span * 0.34)
        year_ticks(ax, first.index[-1])

        ends = [(n, float(s.dropna().iloc[-1])) for n, s in curves.items()]
        offsets = spread_labels(ax, ends)
        for name, val in ends:
            ax.annotate(f" {name} {val/10_000:,.0f}万",
                        xy=(curves[name].index[-1], val),
                        xytext=(7, offsets[name]), textcoords="offset points",
                        color=colors[name], fontsize=8.5, va="center",
                        fontweight="bold" if name == BENCH else "normal")
        ax.annotate("元手100万円", xy=(1.0, CAPITAL), xycoords=("axes fraction", "data"),
                    xytext=(-2, 4), textcoords="offset points",
                    color=t["muted"], fontsize=8.5, ha="right")

    fig.suptitle("仮想100万円の資産推移（対数軸・手数料0.15%＋スリッページ0.10%込み）",
                 color=t["ink"], fontsize=14, x=0.012, ha="left", y=0.975)
    handles = [Line2D([], [], color=colors[n], lw=2.4, label=n) for n in LINE_SERIES]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.005), labelcolor=t["ink2"])
    fig.tight_layout(rect=(0, 0.045, 1, 0.945))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_drawdown(all_curves: dict, t: dict, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4))
    colors = dict(zip(DD_SERIES, t["series"]))

    for ax, (label, _, _) in zip(axes, PERIODS):
        dds = {}
        for name in DD_SERIES:
            s = all_curves[label][name]
            dd = (s / s.cummax() - 1.0) * 100
            dds[name] = dd
            ax.fill_between(dd.index, dd.values, 0, color=colors[name], alpha=0.13,
                            linewidth=0)
            ax.plot(dd.index, dd.values, color=colors[name], linewidth=1.8,
                    solid_capstyle="round")
        ax.axhline(0, color=t["axis"], linewidth=0.8)
        ax.set_title(label, color=t["ink"], fontsize=11.5, loc="left", pad=10)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        style_axes(ax, t)
        first = next(iter(dds.values()))
        ax.set_xlim(first.index[0], first.index[-1])
        year_ticks(ax, first.index[-1])
        ax.set_ylim(-95, 7)

        # 最悪値だけ直接ラベル（全点に数字は置かない）
        pts = [(n, float(dd.min())) for n, dd in dds.items()]
        offsets = spread_labels(ax, pts, min_gap_px=15)
        for name, worst in pts:
            ax.annotate(f"{name} 最大{worst:.0f}%", xy=(dds[name].idxmin(), worst),
                        xytext=(6, offsets[name] - 3), textcoords="offset points",
                        color=colors[name], fontsize=9, va="top",
                        bbox=dict(boxstyle="round,pad=0.18", facecolor=t["surface"],
                                  edgecolor="none", alpha=0.85))

    fig.suptitle("ドローダウン（そのときの最高値から何%下げた状態か）",
                 color=t["ink"], fontsize=14, x=0.012, ha="left", y=0.975)
    fig.text(0.012, 0.905, "トレンドフォローだけが、相場付きが変わっても落ち込みの深さを"
             "同じ水準に抑えている", color=t["ink2"], fontsize=10, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_summary(all_results: dict, t: dict, path: str) -> None:
    """
    期間をまたいだ比較。同じ戦略の2期間を線で結ぶ（ダンベル図）。
    色は「期間」の2値にしか使わず、戦略の識別は軸ラベルで行う。
    """
    labels = [p[0] for p in PERIODS]
    c_is, c_oos = t["series"][0], t["series"][1]
    res_is, res_oos = all_results[labels[0]], all_results[labels[1]]

    names = [n for n in res_is if res_is[n] and res_oos[n]]
    bench_is = res_is[BENCH]["年率(CAGR)"] * 100
    bench_oos = res_oos[BENCH]["年率(CAGR)"] * 100
    diff_is = {n: res_is[n]["年率(CAGR)"] * 100 - bench_is for n in names}
    diff_oos = {n: res_oos[n]["年率(CAGR)"] * 100 - bench_oos for n in names}
    names.sort(key=lambda n: diff_oos[n])          # 下から悪い順

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.6), sharey=True)
    y = range(len(names))

    def dumbbell(ax, va, vb, fmt):
        for i, n in enumerate(names):
            ax.plot([va[n], vb[n]], [i, i], color=t["muted"], linewidth=1.6,
                    alpha=0.45, zorder=1, solid_capstyle="round")
            ax.scatter(va[n], i, s=88, color=c_is, zorder=3,
                       edgecolor=t["surface"], linewidth=2)
            ax.scatter(vb[n], i, s=88, color=c_oos, zorder=3,
                       edgecolor=t["surface"], linewidth=2)
            # 検証用期間の値だけ直接ラベルする
            right = vb[n] >= va[n]
            ax.annotate(fmt(vb[n]), xy=(vb[n], i),
                        xytext=(9 if right else -9, 0), textcoords="offset points",
                        color=t["ink"], fontsize=9, va="center",
                        ha="left" if right else "right")

    dumbbell(ax1, diff_is, diff_oos, lambda v: f"{v:+.0f}pt")
    ax1.axvline(0, color=t["axis"], linewidth=1.0)
    ax1.set_yticks(list(y))
    ax1.set_yticklabels(names, color=t["ink"], fontsize=9.5)
    ax1.set_title("年率リターンの「BTC買い持ちとの差」\n"
                  "0より右＝買い持ちに勝った期間",
                  color=t["ink"], fontsize=11, loc="left", pad=10)
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}pt"))
    style_axes(ax1, t, grid_axis="x")
    ax1.set_xlim(-62, 62)

    dd_is = {n: res_is[n]["最大DD"] * 100 for n in names}
    dd_oos = {n: res_oos[n]["最大DD"] * 100 for n in names}
    dumbbell(ax2, dd_is, dd_oos, lambda v: f"{v:.0f}%")
    ax2.set_title("最大ドローダウン\n右ほど浅い＝安全",
                  color=t["ink"], fontsize=11, loc="left", pad=10)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    style_axes(ax2, t, grid_axis="x")
    ax2.set_xlim(-97, 2)

    for ax in (ax1, ax2):
        ax.set_ylim(-0.7, len(names) - 0.3)
        ax.spines["left"].set_visible(False)

    fig.suptitle("作り込んだ期間と、見ていない期間の比較",
                 color=t["ink"], fontsize=14, x=0.012, ha="left", y=0.975)
    fig.text(0.012, 0.935,
             "左: 作り込み期間で買い持ちに勝っていた戦略が、検証期間ではことごとく負けに転じた（点が0の左へ移動）。"
             "プラスを保ったのは、売買タイミングを一切判断しない均等分散だけ。\n"
             "右: 一方で最大ドローダウンの水準は、期間をまたいでも順位がほとんど入れ替わらない。",
             color=t["ink2"], fontsize=9.5, ha="left", va="top", linespacing=1.6)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=9, color=c, label=l)
               for c, l in ((c_is, "2018-2022（作り込み用）"), (c_oos, "2023-2026（検証用）"))]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.004), labelcolor=t["ink2"])
    fig.tight_layout(rect=(0, 0.04, 1, 0.865))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dark", action="store_true", help="ダーク配色で出力")
    args = ap.parse_args()

    t = setup("dark" if args.dark else "light")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results, all_curves = {}, {}
    for label, start, end in PERIODS:
        print(f"集計中: {label}")
        res, curves = bt.run_period(label, start, end, quiet=True)
        all_results[label], all_curves[label] = res, curves

    sfx = "_dark" if args.dark else ""
    jobs = [
        (f"fig1_equity{sfx}.png", lambda p: fig_equity(all_curves, t, p)),
        (f"fig2_drawdown{sfx}.png", lambda p: fig_drawdown(all_curves, t, p)),
        (f"fig3_summary{sfx}.png", lambda p: fig_summary(all_results, t, p)),
    ]
    for fname, fn in jobs:
        path = os.path.join(RESULTS_DIR, fname)
        fn(path)
        print(f"出力: {path}")


if __name__ == "__main__":
    main()
