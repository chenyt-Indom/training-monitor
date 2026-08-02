"""通用训练监控 - 自动检测V3/V4/V5/... 全量指标"""
import os, time, subprocess, glob

# 自动检测最新训练版本
LOG_BASE = r"D:\training_data\logs"
versions = []
for d in glob.glob(os.path.join(LOG_BASE, "person_v*")):
    if os.path.isdir(d):
        csv = os.path.join(d, "results.csv")
        if os.path.exists(csv):
            versions.append((csv, os.path.basename(d)))
versions.sort(key=lambda x: x[1])

if not versions:
    print("未检测到训练数据")
    input("按回车退出...")
    exit()

CSV, VERSION = versions[-1]
start = time.time()

# 预读训练参数
n_total = 30
args_yaml = os.path.join(LOG_BASE, VERSION, "args.yaml")
if os.path.exists(args_yaml):
    with open(args_yaml) as f:
        for line in f:
            if 'epochs:' in line:
                try: n_total = int(line.strip().split(':')[1].strip())
                except: pass
                break

print(f"已绑定: {VERSION} ({n_total}轮)")
while True:
    os.system('cls')
    elapsed = (time.time() - start) / 60
    print("=" * 78)
    print(f"  {VERSION} 训练监控  {time.strftime('%H:%M:%S')}  已监控 {elapsed:.0f}分")
    
    # 读取训练参数
    args_yaml = os.path.join(LOG_BASE, VERSION, "args.yaml")
    if os.path.exists(args_yaml):
        args = {}
        with open(args_yaml) as f:
            for line in f:
                if ':' in line:
                    k, v = line.strip().split(':', 1)
                    k = k.strip(); v = v.strip()
                    if k in ['epochs','dropout','weight_decay','patience','warmup_epochs','close_mosaic',
                             'lr0','workers','cache','batch','imgsz']:
                        args[k] = v
        print(f"  epochs={n_total} | dropout={args.get('dropout','?')} | wd={args.get('weight_decay','?')} | "
              f"patience={args.get('patience','?')} | warmup={args.get('warmup_epochs','?')}+{args.get('close_mosaic','?')}")
    print("=" * 78)

    if os.path.exists(CSV):
        lines = [l.strip().split(',') for l in open(CSV).readlines()[1:]]
        
        print(f"  {'Ep':<4s} {'box':>7s} {'cls':>7s} {'dfl':>7s} {'P':>6s} {'R':>6s} {'mAP50':>7s} {'mAP95':>7s} {'v_box':>7s} {'趋势':>6s}")
        print("  " + "-" * 76)

        maps = []
        for i, d in enumerate(lines):
            e    = int(d[0])
            box  = float(d[2])
            cls  = float(d[3])
            dfl  = float(d[4])
            P    = float(d[5])
            R    = float(d[6])
            m50  = float(d[7])
            m95  = float(d[8])
            vbox = float(d[9]) if len(d) > 9 else 0
            maps.append(m95)

            arrow = " --"
            if i >= 1:
                diff = m95 - maps[i-1]
                if diff > 0.005: arrow = " ▲▲"
                elif diff > 0.001: arrow = " ▲ "
                elif diff > -0.001: arrow = " ▬ "
                elif diff > -0.005: arrow = " ▼ "
                else: arrow = " ▼▼"

            tag = " 🏆" if m95 >= max(maps) and len(maps) > 1 else ""
            print(f"  {e:<4d} {box:7.4f} {cls:7.4f} {dfl:7.4f} {P:6.2%} {R:6.2%} {m50:7.4f} {m95:7.4f} {vbox:7.4f} {arrow}{tag}")

        if len(maps) >= 2:
            inc = sum(1 for i in range(1, len(maps)) if maps[i] > maps[i-1] + 0.001)
            dec = sum(1 for i in range(1, len(maps)) if maps[i] < maps[i-1] - 0.001)
            flat = len(maps) - 1 - inc - dec
            print(f"\n  {inc}升 {dec}降 {flat}平", end="  ")

            boxes = [float(d[2]) for d in lines]
            box_dropping = boxes[-1] < boxes[-2]
            map_dropping = len(maps) >= 3 and maps[-1] < maps[-2] and maps[-2] < maps[-3]
            if maps[-1] > maps[0] and boxes[-1] < boxes[0]:
                print("✅ 健康 (box↓ + mAP↑)")
            elif map_dropping and box_dropping:
                print("⚠️ 过拟合 (box↓ 但 mAP连降)")
            else:
                print("🟡 观察中")

            best = max(maps); best_ep = lines[maps.index(best)][0]
            ep = int(lines[-1][0])
            print(f"  🏆 最佳 E{best_ep}  mAP50-95={best:.4f}")
            print(f"  📊 {ep}/{n_total} 完成", end="")
            if ep > 0 and ep < n_total:
                avg_min = elapsed / ep
                remaining = int((n_total - ep) * avg_min)
                finish = time.localtime(time.time() + remaining * 60)
                print(f"  |  {avg_min:.1f}分/轮  |  剩余 ~{remaining}分  |  预计 {time.strftime('%H:%M', finish)}")
            else:
                print("  |  ✅ 训练完成!")

    else:
        print(f"\n  🔄 训练启动中... (等待首轮验证)")

    time.sleep(3)
