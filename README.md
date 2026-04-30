# Agent Core Architecture

一个"可运行、可测试、可解释"的标准 Agent 核心架构最简实现。

## 架构概览

本实现包含以下五个核心层级，各层之间通过显式的输入输出对象进行交互，实现了松散耦合：

```
┌─────────────────────────────────────────────────────────────┐
│                      执行层 (Orchestrator)                    │
│         感知 → 记忆检索 → 规划 → 工具调用 → 记忆写回 → 输出     │
└─────────────────────────────────────────────────────────────┘
         │           │           │           │           │
         ▼           ▼           ▼           ▼           ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   感知层     │ │   记忆层     │ │   规划层     │ │   工具层     │
│ 输入验证归一化 │ │ 短期/长期记忆 │ │ 任务拆解规划 │ │ 工具注册执行 │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
```

## 功能特性

### 1. 感知层 (Perception Layer)
- 输入验证：空输入、超长输入、非法字符检测
- 输入归一化：去除多余空白、规范化格式
- 生成结构化的 `TaskInput` 对象
- Trace 记录验证状态

### 2. 记忆层 (Memory Layer)
- 短期记忆：会话内有效
- 长期记忆：磁盘持久化（JSON 格式）
- 简单召回算法：关键词匹配、相似度评分
- Schema 版本控制：防止数据格式演进破坏

### 3. 规划层 (Planning Layer)
- 多步骤计划生成
- Action 类型：调用工具、写入记忆、产出最终回答、请求补充信息
- 失败重试策略（可配置）
- 可中止条件（连续失败 N 次后停止）

### 4. 工具层 (Tool Layer)
- 通用工具注册与调用框架
- 内置三个工具：
  - **calculator**: 确定性纯函数工具（四则运算）
  - **unstable_api**: 模拟外部不稳定情况（随机超时/抛错/慢响应）
  - **memory_rw**: 记忆读写工具
- 超时控制、异常捕获
- 结构化错误类型：参数错误、超时、运行时错误、工具不存在

### 5. 执行层 (Execution Layer / Orchestrator)
- 完整执行链路：感知 → 记忆检索 → 规划 → 工具调用 → 记忆写回 → 输出
- Dry-run 模式：仅生成计划，不实际执行工具
- max_steps 限制：防止死循环
- 结构化执行结果：最终回答、是否成功、失败原因、Trace 路径

### 6. 追踪系统 (Tracing)
- 每步记录：阶段名称、输入摘要、输出摘要、耗时
- 额外信息：是否命中记忆、是否调用工具、工具参数及结果/错误
- 控制台输出 + JSONL 文件持久化
- Trace 回放功能

## 快速开始

### 环境要求

- Python 3.10+

### 安装

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/macOS:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 或使用可编辑模式安装
pip install -e .
```

### 运行 Demo

Demo 会展示：
1. 成功的计算器工具调用
2. 记忆写入和读取
3. 不稳定 API 的错误处理

```bash
python -m cli.main demo
```

### 基本命令

#### 1. run - 执行用户输入

```bash
# 简单计算
python -m cli.main run --input "Calculate 15 plus 7"

# 保存到记忆
python -m cli.main run --input "Remember that my favorite color is blue"

# 从记忆读取
python -m cli.main run --input "Recall my favorite color"

# 调用不稳定 API（可能失败）
python -m cli.main run --input "Search for information about AI"

# 带自定义参数
python -m cli.main run \
    --input "Calculate 100 divided by 4" \
    --max-steps 20 \
    --timeout 10.0 \
    --max-retries 5 \
    --abort-threshold 3
```

#### 2. dry-run - 仅生成计划

```bash
python -m cli.main dry-run --input "Calculate 10 multiplied by 5"
```

#### 3. replay-trace - 回放 Trace

```bash
# 先运行一次产生 trace
python -m cli.main run --input "Calculate 1 plus 1" --trace-path ./data/my_trace.jsonl

# 回放 trace
python -m cli.main replay-trace --trace-file ./data/my_trace.jsonl
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行测试并显示详细输出
pytest -v

# 运行特定测试文件
pytest tests/test_perception.py -v
pytest tests/test_tools.py -v
pytest tests/test_memory.py -v
pytest tests/test_execution.py -v

# 生成覆盖率报告
pytest --cov=agent tests/
```

## 目录结构

```
11-agent-core-architecture/
├── agent/                      # 核心模块
│   ├── __init__.py
│   ├── types.py               # 核心类型定义（接口、数据结构）
│   ├── perception.py          # 感知层
│   ├── memory.py              # 记忆层
│   ├── tools.py               # 工具层
│   ├── planning.py            # 规划层
│   ├── execution.py           # 执行层（Orchestrator）
│   └── tracing.py             # 追踪系统
├── cli/                       # 命令行入口
│   ├── __init__.py
│   └── main.py
├── tests/                     # 单元测试
│   ├── __init__.py
│   ├── test_perception.py
│   ├── test_memory.py
│   ├── test_tools.py
│   └── test_execution.py
├── data/                      # 数据目录（运行时生成）
│   ├── memory_*.json          # 记忆持久化文件
│   └── trace_*.jsonl          # Trace 文件
├── pyproject.toml             # 项目配置
├── requirements.txt           # 依赖列表
└── README.md                  # 本文档
```

## 各层接口说明

### 类型定义 (`agent/types.py`)

所有层都通过以下协议（Protocol）进行交互：

- `PerceptionProtocol`: 感知层接口
- `PlanningProtocol`: 规划层接口
- `ToolRegistryProtocol`: 工具层接口
- `MemoryProtocol`: 记忆层接口

核心数据类：

- `UserInput`: 用户原始输入
- `Context`: 上下文信息（对话历史、记忆检索结果、可用工具）
- `TaskInput`: 规范化的任务输入
- `Action`: 单个动作
- `Plan`: 执行计划
- `MemoryItem`: 记忆项
- `ToolSchema`: 工具 Schema
- `ToolCallRequest`: 工具调用请求
- `ToolResult`: 工具执行结果
- `TraceEvent`: 追踪事件
- `ExecutionResult`: 执行结果

### 执行流程

```
1. 感知层处理输入
   └── 验证 → 归一化 → 生成 TaskInput

2. 记忆检索
   └── 使用用户输入查询相关记忆

3. 规划层生成计划
   └── 分析输入 → 生成多步骤 Action 列表

4. 执行计划步骤
   ├── 调用工具（如有）
   ├── 写入记忆（如有）
   └── 产出最终回答

5. 失败处理
   ├── 连续失败计数
   ├── 未达阈值 → 重试/重新规划
   └── 达到阈值 → 中止并返回失败原因
```

## 测试覆盖

测试文件覆盖以下场景：

### `test_perception.py`
- 空输入验证
- 有效输入验证
- 超长输入验证
- 输入归一化
- 截断处理

### `test_memory.py`
- 短期/长期记忆写入
- 按类型检索
- 按时间范围检索
- 相似度评分算法
- 持久化与加载
- Schema 版本控制

### `test_tools.py`
- 工具参数校验（必填、类型、范围）
- 计算器工具（四则运算、除零错误）
- 不稳定 API（强制成功/错误/超时）
- 工具注册与发现
- 错误类型区分（参数错误/超时/运行时错误）

### `test_execution.py`
- 有效/无效输入执行
- Dry-run 模式
- max_steps 限制
- Trace 记录
- 工具超时处理
- Trace 文件格式验证

## 设计原则

1. **松散耦合**: 层与层之间仅通过 Protocol 接口和数据对象交互，不直接导入具体实现
2. **可测试性**: 所有组件都是纯 Python 类，易于实例化和测试
3. **可解释性**: Trace 详细记录每一步执行，JSONL 格式支持后续分析
4. **可扩展性**: 新工具只需继承 `BaseTool`，新规划逻辑只需实现 `PlanningProtocol`
5. **健壮性**: 所有可能的错误都被捕获并结构化，不会导致整个进程崩溃

## License

MIT
