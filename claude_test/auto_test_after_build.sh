#!/bin/bash
# 编译完成后自动测试脚本
# Auto Test Script After Build Completes

set -e

LOG_FILE="/home/qlyu/AA_peptide/pepland/claude_test/dgl_build.log"
TEST_DIR="/home/qlyu/AA_peptide/pepland/claude_test"
DOCS_DIR="${TEST_DIR}/docs"

echo "=========================================="
echo "等待 DGL 编译完成..."
echo "Waiting for DGL build to complete..."
echo "=========================================="

# 等待编译完成的函数
wait_for_build() {
    while true; do
        if grep -q "DGL build completed" "$LOG_FILE" 2>/dev/null; then
            echo "✓ DGL 编译完成！"
            return 0
        elif grep -q "Error\|Failed\|fatal" "$LOG_FILE" 2>/dev/null; then
            echo "✗ 编译失败，请检查日志"
            return 1
        fi
        sleep 30  # 每30秒检查一次
    done
}

# 更新测试文档
update_doc() {
    local status="$1"
    local test_name="$2"
    local details="$3"

    echo "" >> "${DOCS_DIR}/测试说明.md"
    echo "### ${test_name}" >> "${DOCS_DIR}/测试说明.md"
    echo "- **时间**: $(date '+%Y-%m-%d %H:%M:%S')" >> "${DOCS_DIR}/测试说明.md"
    echo "- **状态**: ${status}" >> "${DOCS_DIR}/测试说明.md"
    echo "- **说明**: ${details}" >> "${DOCS_DIR}/测试说明.md"
    echo "" >> "${DOCS_DIR}/测试说明.md"
}

# 主流程
if wait_for_build; then
    echo ""
    echo "开始自动测试流程..."
    echo ""

    cd "$TEST_DIR"

    # 测试 1: DGL 基础功能
    echo "=== 测试 1: DGL + CUDA 基础功能 ==="
    if CUDA_VISIBLE_DEVICES=3 python test_dgl_cuda.py > test_results/test1_dgl_basic.log 2>&1; then
        update_doc "✅ 通过" "1.1 DGL + CUDA 基础测试" "DGL 在 GPU 3 上运行正常，支持图操作和消息传递"
        echo "✓ 测试 1 通过"
    else
        update_doc "❌ 失败" "1.1 DGL + CUDA 基础测试" "见日志: test_results/test1_dgl_basic.log"
        echo "✗ 测试 1 失败"
    fi

    sleep 2

    # 测试 2: 模型组件
    echo ""
    echo "=== 测试 2: 模型组件 ==="
    if CUDA_VISIBLE_DEVICES=3 python test_model_cuda.py > test_results/test2_model_components.log 2>&1; then
        update_doc "✅ 通过" "2.1 模型组件测试" "Performer、Pooling、特征编码器等组件在 GPU 上运行正常"
        echo "✓ 测试 2 通过"
    else
        update_doc "❌ 失败" "2.1 模型组件测试" "见日志: test_results/test2_model_components.log"
        echo "✗ 测试 2 失败"
    fi

    echo ""
    echo "=========================================="
    echo "自动测试完成，详细结果见:"
    echo "${DOCS_DIR}/测试说明.md"
    echo "=========================================="

else
    echo "编译失败，跳过测试"
    update_doc "❌ 失败" "DGL 编译" "编译失败，见日志: dgl_build.log"
fi
