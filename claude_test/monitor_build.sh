#!/bin/bash
# 监控 DGL 编译进度 / Monitor DGL Build Progress

echo "=========================================="
echo "DGL 编译进度监控 / DGL Build Progress Monitor"
echo "=========================================="
echo ""
echo "按 Ctrl+C 退出监控（不会停止编译）"
echo "Press Ctrl+C to exit monitor (build will continue)"
echo ""

LOG_FILE="/home/qlyu/AA_peptide/pepland/claude_test/dgl_build.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "日志文件不存在: $LOG_FILE"
    echo "Log file not found: $LOG_FILE"
    exit 1
fi

# 持续显示日志尾部 / Continuously show log tail
tail -f "$LOG_FILE"
