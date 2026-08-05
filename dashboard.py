"""
運用状況をブラウザで見るためのローカルサーバ
------------------------------------------------------------
実行:
  python dashboard.py            → http://127.0.0.1:8787 をブラウザで開く
  python dashboard.py --port 9000
  python dashboard.py --open     → 既定のブラウザで自動的に開く

live_trade.py とは別プロセスで動く。トレーダーが止まっていても画面は開けて、
「停止中」と分かるようにするため（組み込みにすると、止まった瞬間に確認手段も消える）。
表示するのはトレーダーが書いたファイルだけなので、この画面が価格を取りにいくことはない。

  state/snapshot.json   … 現況（毎ループ更新）
  state/live.lock       … 稼働判定に使う心拍
  results/live_equity.csv … 資産推移
  results/live_trades.csv … 売買履歴

127.0.0.1 のみで待ち受ける（LANや外部からは見えない）。
"""

import argparse
import csv
import json
import os
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import live_trade as lt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAGE_FILE = os.path.join(BASE_DIR, "dashboard.html")
MAX_CURVE_POINTS = 1500     # これを超えたら間引いて返す
MAX_TRADES = 40


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


_csv_cache: dict[str, tuple] = {}


def _read_csv(path: str) -> list[dict]:
    """
    更新時刻とサイズが変わっていなければ前回の結果を使い回す。
    資産ログは5分ごとに1行増えるだけなのに、画面は15秒ごとに問い合わせてくる。
    毎回ファイル全体を読み直すと、運用が長引くほど無駄が増えていく。
    """
    try:
        st = os.stat(path)
    except OSError:
        return []
    key = (st.st_mtime_ns, st.st_size)
    hit = _csv_cache.get(path)
    if hit and hit[0] == key:
        return hit[1]
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return []
    _csv_cache[path] = (key, rows)
    return rows


def _equity_curve() -> list[dict]:
    rows = _read_csv(lt.EQUITY_LOG)
    if len(rows) > MAX_CURVE_POINTS:      # 古いほど粗くてよい
        step = len(rows) // MAX_CURVE_POINTS + 1
        thinned = rows[::step]
        # (len-1) が step で割り切れると最終行がすでに含まれており、
        # 単純に + rows[-1:] すると同じ時刻・同じ値の点が2つ並んでしまう
        if thinned[-1] is not rows[-1]:
            thinned = thinned + rows[-1:]
        rows = thinned
    out = []
    for r in rows:
        try:
            out.append({"t": r["日時"], "equity": float(r["総資産(円)"]),
                        "regime": r.get("レジーム", "")})
        except (KeyError, ValueError):
            continue
    return out


def _trades() -> list[dict]:
    rows = _read_csv(lt.TRADES_LOG)[-MAX_TRADES:]
    out = []
    for r in reversed(rows):              # 新しいものを先に
        try:
            out.append({
                "t": r["日時"], "symbol": r["銘柄"], "side": r["売買"],
                "price": float(r["約定単価"]), "amount": float(r["金額(円)"]),
                "realized": float(r["実現損益(円)"]) if r.get("実現損益(円)") else None,
                "reason": r.get("理由", ""),
            })
        except (KeyError, ValueError):
            continue
    return out


def _trade_stats() -> dict:
    """
    決済済みの売買だけを集計する。含み損益は snapshot 側にあるので、
    ここでは「確定した結果」だけを見る。運用が長引くほど、
    今の評価額より「これまで勝てているか」が判断材料になる。
    """
    realized = []
    by_symbol: dict[str, dict] = {}
    for r in _read_csv(lt.TRADES_LOG):
        if r.get("売買") != "売" or not r.get("実現損益(円)"):
            continue
        try:
            v = float(str(r["実現損益(円)"]).replace(",", ""))
        except ValueError:
            continue
        realized.append(v)
        # 銘柄ごとの実績。どの通貨で勝てているかは全体の合計では分からない
        s = by_symbol.setdefault(r.get("銘柄", "?"),
                                 {"symbol": r.get("銘柄", "?"), "realized": 0.0,
                                  "count": 0, "wins": 0})
        s["realized"] += v
        s["count"] += 1
        s["wins"] += 1 if v > 0 else 0
    if not realized:
        return {"count": 0, "by_symbol": []}
    wins = [v for v in realized if v > 0]
    losses = [v for v in realized if v <= 0]
    gross_win, gross_loss = sum(wins), -sum(losses)
    return {
        "count": len(realized),
        "wins": len(wins),
        "win_rate": len(wins) / len(realized) * 100,
        "total": sum(realized),
        "avg_win": gross_win / len(wins) if wins else 0.0,
        "avg_loss": -gross_loss / len(losses) if losses else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else None,
        "best": max(realized),
        "worst": min(realized),
        "by_symbol": sorted(by_symbol.values(), key=lambda s: -s["realized"]),
    }


def _recent_errors(limit: int = 5) -> list[str]:
    """
    直近のエラーを画面に出すために読む。
    「ログを確認してください」と書きながらブラウザからは見られない、では
    診断の役に立たない。行頭が [日時] の行だけを拾う（本文は要約のみ）。
    """
    try:
        with open(lt.ERROR_LOG, encoding="utf-8") as f:
            lines = [l.rstrip() for l in f if l.startswith("[")]
    except OSError:
        return []
    return lines[-limit:][::-1]


def build_state() -> dict:
    snap = _read_json(lt.SNAPSHOT_FILE)
    lock = lt.read_lock()

    running = lock is not None
    stale_reason = ""
    if not running:
        raw = _read_json(lt.LOCK_FILE)
        if raw:
            stale_reason = f"最後の心拍 {raw.get('heartbeat', '不明')}"

    return {
        "running": running,
        "pid": lock.get("pid") if lock else None,
        "interval": lock.get("interval") if lock else None,
        "heartbeat": lock.get("heartbeat") if lock else None,
        "stale_reason": stale_reason,
        "snapshot": snap,
        "curve": _equity_curve(),
        "trades": _trades(),
        "stats": _trade_stats(),
        "errors": _recent_errors(),
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


class Server(ThreadingHTTPServer):
    # Windows では SO_REUSEADDR が有効だと「使用中のポート」にも bind が通ってしまい、
    # 二重起動が検出できないまま後から起動した方がポートを奪う。明示的に切る。
    allow_reuse_address = False


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            try:
                with open(PAGE_FILE, "rb") as f:
                    self._send(f.read(), "text/html; charset=utf-8")
            except OSError:
                self.send_error(500, "dashboard.html が見つかりません")
        elif path == "/api/state":
            body = json.dumps(build_state(), ensure_ascii=False).encode("utf-8")
            self._send(body, "application/json; charset=utf-8")
        else:
            self.send_error(404)

    def log_message(self, *args) -> None:
        pass      # アクセスログは出さない（コンソールが埋まるため）


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--open", action="store_true", help="ブラウザを自動で開く")
    args = ap.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    try:
        server = Server(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        # 二重起動や他アプリとのポート衝突。生の例外を出しても分かりにくいので言い換える。
        print(f"ポート {args.port} を使えませんでした: {exc}")
        print(f"すでにダッシュボードが起動している可能性があります。まず {url} を開いてみてください。")
        print(f"別のポートで動かすなら: python dashboard.py --port 8788")
        return
    print(f"運用状況ダッシュボード: {url}")
    print("Ctrl+C で終了します。（このサーバは表示専用で、売買には一切関与しません）")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n終了しました。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
