"""通用训练监控 v3 - 全量数据 + ASCII mAP可视化"""
import os, time, glob

LOG_BASE = r"D:\training_data\logs"
all_dirs = sorted(glob.glob(os.path.join(LOG_BASE, "person_v*")), reverse=True)
if not all_dirs:
    print("未检测到训练数据"); input("按回车退出..."); exit()

VERSION = os.path.basename(all_dirs[0])
CSV = os.path.join(all_dirs[0], "results.csv")
args_yaml = os.path.join(LOG_BASE, VERSION, "args.yaml")
if not os.path.exists(args_yaml):
    for d in all_dirs:
        c = os.path.join(d, "results.csv")
        if os.path.exists(c): CSV = c; VERSION = os.path.basename(d); break

start = time.time()
n_total = 30
if os.path.exists(args_yaml):
    with open(args_yaml) as f:
        for line in f:
            if 'epochs:' in line:
                try: n_total = int(line.strip().split(':')[1].strip())
                except: pass; break

print(f"\n  {VERSION} ({n_total}轮)  3秒刷新  Ctrl+C退出\n")
time.sleep(1)

while True:
    os.system('cls')
    elapsed = (time.time() - start) / 60

    # ===================== 参数区 =====================
    print("=" * 100)
    print(f"  {VERSION}  训练监控  {time.strftime('%H:%M:%S')}  已监控 {elapsed:.0f}分")
    args = {}
    if os.path.exists(args_yaml):
        with open(args_yaml) as f:
            for line in f:
                if ':' in line:
                    k, v = line.strip().split(':', 1)
                    k, v = k.strip(), v.strip()
                    if k in ['epochs','dropout','weight_decay','patience','warmup_epochs',
                             'close_mosaic','lr0','workers','cache','batch','imgsz','freeze']:
                        args[k] = v
    parts = []
    for k in ['epochs','batch','workers','freeze','cache','dropout','patience']:
        if k in args: parts.append(f'{k}={args[k]}')
    print(f"  {' | '.join(parts)}")
    print("=" * 100)

    if os.path.exists(CSV):
        lines = [l.strip().split(',') for l in open(CSV).readlines()[1:]]
        if not lines: continue
        ep = int(lines[-1][0])

        # ===================== 全量数据表 =====================
        print(f"  {'Ep':>3s}  {'box':>7s}  {'cls':>7s}  {'dfl':>7s}  {'P':>6s}  {'R':>6s}  {'mAP50':>7s}  {'mAP95':>7s}  {'v_box':>7s}   {'趋势':>5s}")
        print("  " + "-" * 96)

        # 去重: 每轮只保留最后一次记录
        seen = {}
        for d in lines:
            seen[int(d[0])] = d
        lines = [seen[k] for k in sorted(seen.keys())]

        maps = []; boxes_list = []; vboxes = []
        for i, d in enumerate(lines):
            e      = int(d[0])
            box    = float(d[2])
            cls    = float(d[3])
            dfl    = float(d[4])
            P      = float(d[5])
            R      = float(d[6])
            m50    = float(d[7])
            m95    = float(d[8])
            vbox   = float(d[9]) if len(d) > 9 else 0.0
            maps.append(m95)
            boxes_list.append(box)
            vboxes.append(vbox)

            arrow = " --"
            if i >= 1:
                diff = m95 - maps[i-1]
                if   diff > 0.005: arrow = "▲▲▲"
                elif diff > 0.002: arrow = " ▲▲"
                elif diff > 0.0005: arrow = "  ▲"
                elif diff > -0.0005: arrow = "  ▬"
                elif diff > -0.002: arrow = "  ▼"
                elif diff > -0.005: arrow = " ▼▼"
                else: arrow = "▼▼▼"

            tag = " 🏆" if m95 >= max(maps) and i > 0 else ""
            # 每行显示全部指标
            print(f"  {e:>3d}  {box:7.4f}  {cls:7.4f}  {dfl:7.4f}  {P:5.1%}  {R:5.1%}  {m50:7.4f}  {m95:7.4f}  {vbox:7.4f}  {arrow}{tag}")

        # ===================== ASCII mAP可视化 =====================
        if len(maps) >= 2:
            print(f"\n  {'─'*94}")
            print(f"  📈 mAP50-95 曲线:")
            print(f"  {'─'*94}")

            m_min = max(0, min(maps) - 0.02)
            m_max = min(1.0, max(maps) + 0.05)
            chart_width = 80

            # 每轮画一个柱状图
            for i, m in enumerate(maps):
                bar_len = int((m - m_min) / (m_max - m_min) * chart_width) if m_max > m_min else 0
                bar = "█" * bar_len
                marker = " ◄"
                if m >= max(maps): marker = " ◄🏆"
                print(f"  E{i+1:>3d} ({m:.4f}) │{bar}{marker}")

            print(f"  {'─'*94}")
            print(f"        0.00{' '*(chart_width-10)}{m_max:.2f}")
            print()

            # ===================== 统计 =====================
            inc = sum(1 for i in range(1, len(maps)) if maps[i] > maps[i-1] + 0.0005)
            dec = sum(1 for i in range(1, len(maps)) if maps[i] < maps[i-1] - 0.0005)
            flat = len(maps) - 1 - inc - dec

            status = ""
            if maps[-1] > maps[0] and boxes_list[-1] < boxes_list[0]:
                status = "✅ 健康 (box↓ + mAP↑)"
            elif len(maps) >= 4 and all(maps[-i] < maps[-i-1] for i in range(1,4)):
                status = "⚠️ 过拟合信号 (mAP连降3轮)"
            else:
                status = "🟡 正常波动"

            best = max(maps); best_ep = lines[maps.index(best)][0]
            print(f"  {inc}升 {dec}降 {flat}平 │ {status}")
            print(f"  🏆 最佳: E{best_ep}  mAP50-95={best:.4f}")
            print(f"  📊 进度: {ep}/{n_total}", end="")

            if ep > 1 and ep < n_total:
                avg = elapsed / ep
                rem = int((n_total - ep) * avg)
                finish_ts = time.localtime(time.time() + rem * 60)
                print(f"  │  {avg:.1f}分/轮  │  剩余 ~{rem}分  │  预计 {time.strftime('%H:%M', finish_ts)}")
            else:
                print()
            print()

        else:
            print(f"\n  {ep}/{n_total} | 等待更多数据...\n")
    else:
        try:
            import subprocess
            gpu = subprocess.check_output('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits', shell=False).decode().strip()
        except: gpu = "N/A"
        try:
            import psutil; ram = f"{psutil.virtual_memory().percent}%"
        except: ram = "N/A"
        print(f"\n  🔄 训练启动中 | GPU:{gpu} | RAM:{ram}\n")

    time.sleep(3)
