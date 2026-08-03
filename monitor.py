"""通用训练监控 v2 - 自动检测版本 + 可视化图表"""
import os, time, glob

LOG_BASE = r"D:\training_data\logs"
all_dirs = sorted(glob.glob(os.path.join(LOG_BASE, "person_v*")), reverse=True)

if not all_dirs:
    print("未检测到训练数据"); input("按回车退出..."); exit()

# 选最新: 优先有args.yaml的(活跃训练), 其次有CSV的
VERSION = os.path.basename(all_dirs[0])
CSV = os.path.join(all_dirs[0], "results.csv")
args_yaml = os.path.join(LOG_BASE, VERSION, "args.yaml")
if not os.path.exists(args_yaml):
    # 不是活跃版本, 找最近有CSV的
    for d in all_dirs:
        csv = os.path.join(d, "results.csv")
        if os.path.exists(csv):
            CSV = csv; VERSION = os.path.basename(d); break
start = time.time()

# 总轮数
n_total = 30
args_yaml = os.path.join(LOG_BASE, VERSION, "args.yaml")
if os.path.exists(args_yaml):
    with open(args_yaml) as f:
        for line in f:
            if 'epochs:' in line:
                try: n_total = int(line.strip().split(':')[1].strip())
                except: pass; break

print(f"\n  {VERSION} ({n_total}轮)  3秒刷新  Ctrl+C退出\n")
time.sleep(1)

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOT = True
except:
    HAS_PLOT = False

while True:
    os.system('cls')
    elapsed = (time.time() - start) / 60

    # === 文本区 ===
    print("=" * 90)
    print(f"  {VERSION}  训练监控  {time.strftime('%H:%M:%S')}  已监控 {elapsed:.0f}分")
    args = {}
    if os.path.exists(args_yaml):
        with open(args_yaml) as f:
            for line in f:
                if ':' in line:
                    k, v = line.strip().split(':', 1)
                    k, v = k.strip(), v.strip()
                    if k in ['epochs','dropout','weight_decay','patience','warmup_epochs','close_mosaic',
                             'lr0','workers','cache','batch','imgsz','freeze']:
                        args[k] = v
    parts = []
    for k in ['epochs','batch','workers','freeze','dropout','patience']:
        if k in args: parts.append(f'{k}={args[k]}')
    print(f"  {' | '.join(parts)}")
    print("=" * 90)

    if os.path.exists(CSV):
        lines = [l.strip().split(',') for l in open(CSV).readlines()[1:]]
        ep = int(lines[-1][0])

        # 表头
        print(f"  {'Ep':>3s} {'box':>7s} {'cls':>7s} {'dfl':>7s} {'P':>6s} {'R':>6s} {'mAP50':>7s} {'mAP95':>7s}  {'趋势':>5s}")
        print("  " + "-" * 78)

        maps = []; boxes_list = []
        for i, d in enumerate(lines):
            e, box = int(d[0]), float(d[2])
            cls_loss, dfl = float(d[3]), float(d[4])
            P, R = float(d[5]), float(d[6])
            m50, m95 = float(d[7]), float(d[8])
            maps.append(m95); boxes_list.append(box)

            arrow = " --"
            if i >= 1:
                diff = m95 - maps[i-1]
                if diff > 0.005: arrow = "▲▲↑"
                elif diff > 0.001: arrow = " ▲ ↑"
                elif diff > -0.001: arrow = " ▬ →"
                elif diff > -0.005: arrow = " ▼ ↓"
                else: arrow = "▼▼↓"

            tag = " 🏆" if m95 >= max(maps) and i > 0 else ""
            print(f"  {e:>3d} {box:7.4f} {cls_loss:7.4f} {dfl:7.4f} {P:6.2%} {R:6.2%} {m50:7.4f} {m95:7.4f} {arrow}{tag}")

        # 统计
        if len(maps) >= 2:
            inc = sum(1 for i in range(1, len(maps)) if maps[i] > maps[i-1] + 0.001)
            dec = sum(1 for i in range(1, len(maps)) if maps[i] < maps[i-1] - 0.001)
            flat = len(maps) - 1 - inc - dec
            print(f"\n  {inc}升 {dec}降 {flat}平 | ", end="")

            if maps[-1] > maps[0] and boxes_list[-1] < boxes_list[0]:
                print("✅ 健康 (box↓ + mAP↑)")
            elif len(maps) >= 4 and all(maps[-3:][i] < maps[-4:][i] for i in range(3)):
                print("⚠️ 过拟合信号 (mAP连降)")
            else:
                print("🟡 正常波动")

            best = max(maps); best_ep = lines[maps.index(best)][0]
            print(f"  🏆 E{best_ep}={best:.4f}  |  {ep}/{n_total} 完成", end="")

            if ep > 1 and ep < n_total:
                avg = elapsed / ep
                rem = int((n_total - ep) * avg)
                finish = time.localtime(time.time() + rem * 60)
                print(f"  |  {avg:.1f}分/轮  |  剩余 ~{rem}分  |  {time.strftime('%H:%M', finish)}")

            # === 可视化 ===
            if HAS_PLOT and len(maps) >= 2:
                epochs = np.arange(1, len(maps) + 1)
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                fig.suptitle(f'{VERSION}  Training Progress (Epoch {ep}/{n_total})', fontsize=14, fontweight='bold')

                # 左: Loss
                ax1 = axes[0]
                ax1.plot(epochs, boxes_list, 'b-o', markersize=5, label='box_loss')
                ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
                ax1.set_title('Box Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)
                ax1.axvline(x=5, color='gray', linestyle='--', alpha=0.5, label='warmup end')
                ax1.axvline(x=8, color='red', linestyle='--', alpha=0.5, label='mosaic off')

                # 右: mAP
                ax2 = axes[1]
                ax2.plot(epochs, maps, 'g-o', markersize=5, label='mAP50-95')
                ax2.plot(epochs, [float(d[7]) for d in lines], 'c--o', markersize=3, alpha=0.6, label='mAP50')
                ax2.set_xlabel('Epoch'); ax2.set_ylabel('mAP')
                ax2.set_title('Detection Accuracy'); ax2.legend(); ax2.grid(True, alpha=0.3)
                ax2.axhline(y=best, color='gold', linestyle='--', alpha=0.7, label=f'Best={best:.3f}')
                ax2.set_ylim(bottom=max(0, min(maps)-0.05), top=min(1.0, max(maps)+0.1))

                plt.tight_layout()
                chart_path = os.path.join(LOG_BASE, VERSION, "training_chart.png")
                plt.savefig(chart_path, dpi=100, bbox_inches='tight')
                plt.close()
                print(f"\n  📈 图表已保存: {chart_path}")
            print()
        else:
            print(f"\n  {ep}/{n_total} | 首轮刷新中...\n")
    else:
        try:
            import subprocess
            gpu = subprocess.check_output('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits', shell=False).decode().strip()
        except:
            gpu = "N/A"
        try:
            import psutil
            ram = f"{psutil.virtual_memory().percent}%"
        except:
            ram = "N/A"
        print(f"\n  🔄 训练启动中 | GPU:{gpu} | RAM:{ram}")
        print()

    time.sleep(3)
