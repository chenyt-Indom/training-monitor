#!/usr/bin/env python3
"""通用 YOLO 训练监控 —— 终端实时查看每轮指标、趋势、ETA 与过拟合判断。

特点
----
* 自动发现最新的训练目录，训练换版本时自动跟着切换
* 每轮显示 box/cls/dfl loss、P、R、mAP50、mAP50-95、趋势箭头
* ASCII 曲线直接画在终端里，不用等 results.png
* 首轮还没跑完（results.csv 不存在）时，从训练日志里读实时步数进度
* 自动判断过拟合，给出「健康 / 正常波动 / 过拟合」结论
* Windows / Linux / macOS 通用，终端不支持 emoji 时自动降级为 ASCII 符号

用法
----
    python yolo_monitor.py                          # 自动探测 ./runs/detect 等常见位置
    python yolo_monitor.py --runs-dir D:/fire_runs  # 指定训练输出根目录，可写多次
    python yolo_monitor.py --once                   # 只输出一次快照，便于脚本/日志
    python yolo_monitor.py --interval 5             # 5 秒刷新一次（默认 3 秒）
    python yolo_monitor.py --ascii                  # 强制用纯 ASCII 符号

监控只会读取文件，不会影响训练进程。随时 Ctrl+C 退出即可。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ────────────────────────────────────────────────────────────
# 终端符号：老终端（如 Windows cp936）编不出 emoji 和方块字，自动降级
# ────────────────────────────────────────────────────────────

def _encodable(text: str) -> bool:
    enc = sys.stdout.encoding or "utf-8"
    try:
        text.encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


UNICODE_OK = _encodable("📊█░▲▼🏆")


class Glyphs:
    """一组按终端能力选择的符号。"""

    def __init__(self, unicode_ok: bool = True):
        self.unicode_ok = unicode_ok
        u = unicode_ok
        self.bar_full = "█" if u else "#"
        self.bar_empty = "░" if u else "-"
        self.up3 = "▲▲▲" if u else "+++"
        self.up2 = " ▲▲" if u else " ++"
        self.up1 = "  ▲" if u else "  +"
        self.flat = "  ▬" if u else "  ="
        self.dn1 = "  ▼" if u else "  -"
        self.dn2 = " ▼▼" if u else " --"
        self.dn3 = "▼▼▼" if u else "---"
        self.best = "🏆" if u else "*"
        self.chart = "📊" if u else "[#]"
        self.curve = "📈" if u else "[/]"
        self.wait = "🔄" if u else "[~]"
        self.rule = "─" if u else "-"


G = Glyphs(UNICODE_OK)


# ────────────────────────────────────────────────────────────
# 发现训练目录
# ────────────────────────────────────────────────────────────

# ultralytics 的标准输出布局：runs/{task}/{name}
DEFAULT_TASKS = ("detect", "segment", "pose", "obb", "classify")


def default_roots() -> list[str]:
    """没指定 --runs-dir 时，在常见位置找。"""
    roots: list[str] = []
    for task in DEFAULT_TASKS:
        roots.append(os.path.join("runs", task))
    roots.append("runs")

    cfg = Path.home() / ".yolo_monitor.json"
    if cfg.is_file():
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
            roots.extend(str(p) for p in data.get("runs_dirs", []))
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    return roots


def looks_like_run(path: str) -> bool:
    """训练目录的必要条件：有 weights 子目录。

    这样基准测试、探测目录、半途失败的临时目录都不会被误认为训练。
    """
    return os.path.isdir(os.path.join(path, "weights"))


def freshness(run_dir: str) -> float:
    """一次训练有多「新」。

    优先用 results.csv 的时间，但刚启动、首轮还没跑完的训练没有这个文件，
    只给 0 分会让监控一直停在几个月前的旧任务上，所以退回目录时间兜底。
    """
    csv = os.path.join(run_dir, "results.csv")
    ts = os.path.getmtime(csv) if os.path.exists(csv) else 0.0
    try:
        return max(ts, os.path.getmtime(run_dir))
    except OSError:
        return ts


def collect_runs(roots: list[str]) -> list[str]:
    found: list[str] = []
    for base in roots:
        if not os.path.isdir(base):
            continue
        # 允许直接把某个具体的 run 目录传进来，而不是只接父目录
        if looks_like_run(base):
            found.append(base)
            continue
        found.extend(glob.glob(os.path.join(base, "*")))
    found = [d for d in found if looks_like_run(d)]
    return sorted(set(found), key=freshness, reverse=True)


def read_args(run_dir: str) -> dict:
    """从 args.yaml 读几个关键超参。"""
    cfg = {"epochs": None, "batch": "?", "lr0": "?", "patience": "?", "freeze": "null", "imgsz": "?"}
    path = os.path.join(run_dir, "args.yaml")
    if not os.path.isfile(path):
        return cfg
    want = set(cfg)
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip()
            if key in want:
                cfg[key] = val.strip()
    try:
        cfg["epochs"] = int(cfg["epochs"])
    except (TypeError, ValueError):
        cfg["epochs"] = None
    return cfg


# ────────────────────────────────────────────────────────────
# 首轮未完时的实时进度
# ────────────────────────────────────────────────────────────

def tail_text(path: str, nbytes: int = 8192) -> str:
    """只读文件末尾。

    训练日志里进度条会刷新上万次，整份读进来既慢又占内存。
    """
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - nbytes))
            return fh.read().decode("utf-8", errors="ignore")
    except OSError:
        return ""


# 一行 ultralytics 训练日志：
#    1/20  7.24G  1.233  1.039  0.01281  22  960: 14% --- 1287/8720 4.7it/s 5:22<26:38
_PROGRESS_RE = re.compile(
    r"(\d+)/(\d+)\s+[\d.]+G.*?(\d+)/(\d+)\s+([\d.]+)it/s(?:\s+\S+<(\S+))?")


def live_progress(run_dir: str) -> dict | None:
    """从训练日志尾部捞实时进度。

    ultralytics 要等每轮验证跑完才写 results.csv，首轮结束前监控只能显示
    「启动中」，干等半小时看不到任何数字。日志里有进度条，可以提前显示。
    日志通常被重定向到训练目录的同级目录。
    """
    parent = os.path.dirname(os.path.abspath(run_dir).rstrip("\\/"))
    logs = glob.glob(os.path.join(parent, "*.log"))
    if not logs:
        return None
    log = max(logs, key=os.path.getmtime)
    hits = _PROGRESS_RE.findall(tail_text(log))
    if not hits:
        return None
    ep, n_ep, step, n_step, its, eta = hits[-1]
    return {"ep": int(ep), "n_ep": int(n_ep), "step": int(step),
            "n_step": int(n_step), "its": float(its), "eta": eta or "?",
            "log": os.path.basename(log)}


# ────────────────────────────────────────────────────────────
# 系统信息
# ────────────────────────────────────────────────────────────

def gpu_info() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=3,
        ).decode().strip()
        return out.splitlines()[0] if out else "N/A"
    except (OSError, subprocess.SubprocessError):
        return "N/A"


def ram_info() -> str:
    try:
        import psutil
        return f"{psutil.virtual_memory().percent}%"
    except Exception:
        return "N/A"


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


# ────────────────────────────────────────────────────────────
# 渲染
# ────────────────────────────────────────────────────────────

def read_results(csv_path: str) -> list[list[str]]:
    with open(csv_path, encoding="utf-8", errors="ignore") as fh:
        rows = [line.strip().split(",") for line in fh.readlines()[1:]]
    rows = [r for r in rows if r and r[0].strip()]
    # 有些版本会在续训时重写早期轮次，按 epoch 去重只留最后一次
    seen = {}
    for r in rows:
        try:
            seen[int(r[0])] = r
        except ValueError:
            continue
    return [seen[k] for k in sorted(seen)]


def trend_arrow(cur: float, prev: float) -> str:
    d = cur - prev
    if d > 0.005:
        return G.up3
    if d > 0.002:
        return G.up2
    if d > 0.0005:
        return G.up1
    if d > -0.0005:
        return G.flat
    if d > -0.002:
        return G.dn1
    if d > -0.005:
        return G.dn2
    return G.dn3


def verdict(maps: list[float], boxes: list[float]) -> str:
    u = G.unicode_ok
    if len(maps) >= 6 and all(maps[-i] <= maps[-i - 1] for i in range(1, 6)) \
            and (maps[-6] - maps[-1]) / max(maps[-6], 1e-6) > 0.02:
        return "⚠ 疑似过拟合（连续 5 轮下降且跌幅 >2%）" if u else \
               "!! 疑似过拟合（连续 5 轮下降且跌幅 >2%）"
    if maps[-1] > maps[0] and boxes and boxes[-1] < boxes[0]:
        return "✅ 健康（loss 降 + mAP 升）" if u else "OK 健康（loss 降 + mAP 升）"
    return "· 正常波动"


def render_header(cfg: dict, title: str, elapsed_min: float, width: int = 94) -> None:
    print("=" * width)
    print(f"  {title}  训练监控  {time.strftime('%H:%M:%S')}  已监控 {elapsed_min:.0f}分")
    keys = ["epochs", "batch", "imgsz", "lr0", "patience", "freeze"]
    parts = [f"{k}={cfg[k]}" for k in keys if cfg.get(k) not in (None, "?")]
    if parts:
        print("  " + " | ".join(parts))
    print("=" * width)


def render(cfg: dict, rows: list[list[str]], n_total: int, title: str,
           elapsed_min: float, width: int = 94) -> None:
    render_header(cfg, title, elapsed_min, width)

    print(f"  {'Ep':>3s}  {'box':>7s}  {'cls':>7s}  {'dfl':>7s}  "
          f"{'P':>6s}  {'R':>6s}  {'mAP50':>7s}  {'mAP95':>7s}  {'v_box':>7s}  趋势")
    print("  " + "-" * (width - 4))

    maps: list[float] = []
    boxes: list[float] = []
    best_so_far = -1.0

    for i, r in enumerate(rows):
        def num(idx, default=0.0):
            try:
                return float(r[idx])
            except (IndexError, ValueError):
                return default

        try:
            ep = int(r[0])
        except (IndexError, ValueError):
            continue

        box, cls, dfl = num(2), num(3), num(4)
        prec, rec = num(5), num(6)
        m50, m95, vbox = num(7), num(8), num(9)
        maps.append(m95)
        boxes.append(box)

        arrow = " --" if i == 0 else trend_arrow(m95, maps[i - 1])
        is_best = m95 > best_so_far
        best_so_far = max(best_so_far, m95)
        tag = f" {G.best}" if is_best else ""

        print(f"  {ep:>3d}  {box:7.4f}  {cls:7.4f}  {dfl:7.4f}  "
              f"{prec:5.1%}  {rec:5.1%}  {m50:7.4f}  {m95:7.4f}  {vbox:7.4f}  {arrow}{tag}")

    if not maps:
        print("\n  results.csv 里还没有有效数据行\n")
        return

    # ASCII 曲线
    if len(maps) >= 2:
        print(f"\n  {G.rule * (width - 6)}")
        print(f"  {G.curve} mAP50-95 曲线")
        print(f"  {G.rule * (width - 6)}")
        lo = max(0.0, min(maps) - 0.02)
        hi = min(1.0, max(maps) + 0.05)
        cw = max(20, width - 16)
        peak = max(maps)
        for i, m in enumerate(maps):
            filled = int((m - lo) / (hi - lo) * cw) if hi > lo else 0
            mark = f" <{G.best}" if abs(m - peak) < 1e-9 else ""
            print(f"  E{i + 1:>3d} ({m:.4f}) |{G.bar_full * filled}{mark}")
        print(f"  {G.rule * (width - 6)}")
        print(f"        {lo:.2f}{' ' * max(1, cw - 9)}{hi:.2f}\n")

        inc = sum(1 for i in range(1, len(maps)) if maps[i] > maps[i - 1] + 0.0005)
        dec = sum(1 for i in range(1, len(maps)) if maps[i] < maps[i - 1] - 0.0005)
        flat = len(maps) - 1 - inc - dec
        print(f"  {inc}升 {dec}降 {flat}平 | {verdict(maps, boxes)}")
        best_ep = rows[maps.index(peak)][0]
        print(f"  {G.best} 最佳: E{best_ep}  mAP50-95={peak:.4f}")

    # 总进度 + ETA
    ep = len(rows)
    pct = ep / max(n_total, 1) * 100
    bar_w = 30
    filled = min(bar_w, int(pct / 100 * bar_w))
    bar = G.bar_full * filled + G.bar_empty * (bar_w - filled)
    print(f"  {G.chart} [{bar}] {pct:.0f}% ({ep}/{n_total})", end="")
    if 1 < ep < n_total:
        try:
            avg_min = float(rows[ep - 1][1]) / 60.0 / ep
        except (IndexError, ValueError, ZeroDivisionError):
            avg_min = elapsed_min / max(ep, 1)
        if avg_min > 0:
            remain = int((n_total - ep) * avg_min)
            eta = time.strftime("%H:%M", time.localtime(time.time() + remain * 60))
            print(f"  |  {avg_min:.1f}分/轮  |  剩余 ~{remain}分  |  预计 {eta}")
        else:
            print()
    else:
        print()
    print()


def render_waiting(cfg: dict, title: str, run_dir: str, elapsed_min: float,
                   width: int = 94) -> None:
    render_header(cfg, title, elapsed_min, width)
    gpu, ram = gpu_info(), ram_info()
    prog = live_progress(run_dir)
    if not prog:
        print(f"\n  {G.wait} 训练启动中 ({title}) | GPU:{gpu} | RAM:{ram}\n")
        return
    pct = prog["step"] / max(prog["n_step"], 1) * 100
    bw = 30
    filled = min(bw, int(pct / 100 * bw))
    bar = G.bar_full * filled + G.bar_empty * (bw - filled)
    print(f"\n  {G.wait} 第 {prog['ep']}/{prog['n_ep']} 轮进行中 | GPU:{gpu} | RAM:{ram}")
    print(f"  {G.chart} [{bar}] {pct:3.0f}%  {prog['step']}/{prog['n_step']} 步"
          f"  {prog['its']:.1f} it/s  本轮剩余 {prog['eta']}")
    print(f"  日志: {prog['log']}（首轮验证结束后转为表格视图）\n")


# ────────────────────────────────────────────────────────────
# 主循环
# ────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="通用 YOLO 训练监控：每轮指标 / 趋势 / mAP 曲线 / ETA / 过拟合判断")
    ap.add_argument("--runs-dir", action="append", default=None, metavar="DIR",
                    help="训练输出根目录，可重复指定多次。默认自动探测 ./runs/detect 等")
    ap.add_argument("--interval", type=float, default=3.0, metavar="SEC",
                    help="刷新间隔秒数，默认 3")
    ap.add_argument("--once", action="store_true", help="只输出一次快照后退出")
    ap.add_argument("--no-clear", action="store_true", help="不清屏，把每次刷新都追加输出")
    ap.add_argument("--ascii", action="store_true", help="强制使用纯 ASCII 符号")
    ap.add_argument("--width", type=int, default=94, help="分隔线宽度，默认 94")
    a = ap.parse_args()

    if a.ascii:
        global G
        G = Glyphs(False)

    roots = a.runs_dir or default_roots()
    runs = collect_runs(roots)
    if not runs:
        print("未找到训练目录。")
        print(f"  已查找: {', '.join(roots)}")
        print("  用 --runs-dir 指定训练输出根目录，例如：")
        print("    python yolo_monitor.py --runs-dir D:/fire_runs")
        return 1

    run_dir = runs[0]
    title = os.path.basename(run_dir)
    cfg = read_args(run_dir)
    n_total = cfg["epochs"] or 100

    print(f"\n  监控: {title} ({n_total} 轮)  {a.interval:.0f} 秒刷新  Ctrl+C 退出\n")
    print(f"  目录: {run_dir}\n")
    if not a.once:
        time.sleep(1)

    start = time.time()
    while True:
        if not a.no_clear and not a.once:
            clear_screen()
        elapsed = (time.time() - start) / 60

        latest = collect_runs(roots)
        if latest and latest[0] != run_dir:
            run_dir = latest[0]
            title = os.path.basename(run_dir)
            cfg = read_args(run_dir)
            n_total = cfg["epochs"] or 100
            print(f"\n  自动切换至: {title}\n")

        csv_path = os.path.join(run_dir, "results.csv")
        if os.path.isfile(csv_path):
            try:
                rows = read_results(csv_path)
            except OSError:
                rows = []
            if rows:
                render(cfg, rows, n_total, title, elapsed, a.width)
            else:
                render(cfg, [], n_total, title, elapsed, a.width)
        else:
            render_waiting(cfg, title, run_dir, elapsed, a.width)

        if a.once:
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已退出监控。训练进程不受影响。")
        sys.exit(0)
