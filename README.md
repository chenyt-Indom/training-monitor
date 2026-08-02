# YOLO 训练通用监控 (Universal Training Monitor)

自动检测 `D:\training_data\logs\` 下最新训练版本，实时显示全量指标。

## 功能

- 自动绑定：无需配置，启动即检测最新训练 (person_v3/v4/v5/...)
- 全量显示：box_loss, cls_loss, dfl_loss, Precision, Recall, mAP50, mAP50-95, val_box_loss
- 趋势分析：每轮 ▲▲/▲/▬/▼/▼▼ 箭头 + 升/降/平统计
- 过拟合检测：box↓ 但 mAP连降 → 警告
- 时间预估：每轮耗时、剩余时间、预计完成时刻
- 超参显示：自动读取 args.yaml

## 使用

双击 `monitor.bat` 或：
```
python monitor.py
```

## 配置

默认监控 `D:\training_data\logs\person_v*` 目录。修改 `LOG_BASE` 变量适配其他路径。
