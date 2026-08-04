"""
相場付き（レジーム）の判定
------------------------------------------------------------
バックテストで分かったのは「リターンの優劣は相場付きでほぼ決まり、戦略の腕では
決まらない」ということだった。ならば固定の戦略を回すのではなく、
まず相場付きを判定し、それに合った戦略とエクスポージャー上限を選ぶ。

判定に使うのは3つだけ（増やすほど当てにいく＝過剰最適化になる）:
  1. BTCが200日移動平均より上か下か  … 大局のトレンド
  2. 直近30日の年率ボラティリティ      … 荒れているか
  3. 上昇銘柄の比率（各銘柄の50日線超え）… 相場全体が伸びているか

  強気 : BTCが200日線の上 / 荒れていない / 半分以上の銘柄が上昇     → 積極（上限100%）
  中立 : BTCは200日線の上だが、荒れている or 銘柄がついてきていない → 慎重（上限60%）
  弱気 : BTCが200日線の下                                          → 防御（上限40%）
"""

import math

import pandas as pd

TREND_MA = 200          # 大局トレンドの判定に使う移動平均（日）
BREADTH_MA = 50         # 各銘柄の上昇判定に使う移動平均（日）
VOL_WINDOW = 30         # ボラティリティの計測窓（日）
HIGH_VOL = 0.90         # 年率ボラがこれを超えたら「荒れている」
ANCHOR = "BTC-JPY"      # 大局判定の基準銘柄

# レジーム名 -> (戦略キー, エクスポージャー上限)
PLAYBOOK = {
    "強気": ("分散保有", 1.00),
    "中立": ("ドンチャン20/10", 0.60),
    "弱気": ("ドンチャン55/20", 0.40),
}

# 1銘柄あたりの上限ウェイト。1銘柄への集中を避けるための歯止め。
# 実運用(live_trade.py)とバックテスト(strategies/regime_switch.py)の両方がここを読む。
# 2箇所に別々に書くと必ずズレて、「検証したものと違うものが動いている」状態になる。
#
# 0.30 を選んだ経緯（python sensitivity.py で確認できる）:
#   0.5 … リターン最大だが最大DDが相場次第で -34%〜-39% とぶれる
#   0.4 … 効くのが全体の8%の日だけで、守勢のレジームでは一度も効かない（中途半端）
#   0.3 … リターンは削れるが、最大DDが両期間とも -31.4% に揃う
#          ＝落ち込みの深さを相場任せにしない
MAX_WEIGHT = 0.30

# ストップで手仕舞った銘柄を、この日数だけ買い直さない。
# トレーリングストップに引っかかった直後の銘柄を即座に買い戻すのは、
# 崩れ始めた動きに乗り直す行為で、往復売買（ホイップソー）になりやすい。
# 実測では買い直しの22.9%が売却の翌日以内に起きていた。
#
# 7〜20日が広い平坦域を作り、両期間ともリターンと最大DDが同時に改善する
# （Calmar: 検証用期間 1.26 → 1.58）。6日以下・25日以上では崩れる。
# 平坦域の中央として10日を採用。python sensitivity.py で再確認できる。
COOLDOWN_DAYS = 10


def classify(closes: pd.DataFrame) -> dict:
    """
    closes: index=日付 / columns=銘柄 の終値（**今日まで**に切ってあること）。
    判定結果と、その根拠になった数値を返す。
    """
    btc = closes[ANCHOR].dropna()
    price = float(btc.iloc[-1])

    ma = float(btc.iloc[-TREND_MA:].mean()) if len(btc) >= TREND_MA else float("nan")
    above = ma == ma and price > ma

    rets = btc.pct_change().dropna().iloc[-VOL_WINDOW:]
    vol = float(rets.std() * math.sqrt(365)) if len(rets) > 2 else float("nan")
    calm = vol == vol and vol < HIGH_VOL

    up, total = 0, 0
    for sym in closes.columns:
        c = closes[sym].dropna()
        if len(c) < BREADTH_MA:
            continue
        total += 1
        if float(c.iloc[-1]) > float(c.iloc[-BREADTH_MA:].mean()):
            up += 1
    breadth = up / total if total else 0.0

    if not above:
        name = "弱気"
    elif calm and breadth >= 0.5:
        name = "強気"
    else:
        name = "中立"

    strategy_key, cap = PLAYBOOK[name]
    return {
        "レジーム": name,
        "戦略": strategy_key,
        "上限": cap,
        "BTC価格": price,
        "200日線": ma,
        "200日線比": price / ma - 1.0 if ma == ma else float("nan"),
        "年率ボラ": vol,
        "上昇銘柄比率": breadth,
        "理由": _reason(above, calm, breadth, vol),
    }


def _reason(above: bool, calm: bool, breadth: float, vol: float) -> str:
    parts = ["BTCが200日線の上" if above else "BTCが200日線の下"]
    parts.append(f"年率ボラ{vol*100:.0f}%" + ("（落ち着き）" if calm else "（荒れ）")
                 if vol == vol else "ボラ不明")
    parts.append(f"上昇銘柄{breadth*100:.0f}%")
    return " / ".join(parts)
