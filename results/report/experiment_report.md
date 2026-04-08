# 错题本实验报告
## BFCL V4 工具调用基准测试

**日期**：2026-04-08
**基准**：Berkeley Function Calling Leaderboard V4
**测试集**：782 条数据，覆盖 11 个类别

> [English Version](experiment_report_en.md)

---

## 1. 概述

本项目提出**错题本**方法，通过纠错式 Few-Shot 提示改进大模型的工具调用准确率。核心思想：与其向模型展示正确示例，不如展示**常见错误及其修正**。

### 核心结果

| 方法 | Volcengine Auto | Gemma4 26B |
|------|----------------|------------|
| Zero-shot 基线 | 68.0% | 76.3% |
| 纯错题（5道纠错） | **79.0% (+11.0pp)** | — |
| 混合 5+5（正确+纠错） | 76.3% (+8.3pp) | 77.6%* (+1.3pp) |
| 交替排列（正→错→正→错） | **79.0% (+11.0pp)** | — |
| 纯正面示例（5道正确） | 64.7% (-3.3pp) | — |

*Gemma4 错题本结果基于 634/782 条数据

**主要发现**：纠错示例是唯一的提升因素。仅展示正确示例反而拖累了性能（-3.3pp）。提升幅度在模型并行调用能力较弱时最为显著。

---

## 2. 评估流程中的 Bug 修复

在 BFCL 官方 AST checker 中发现并修复了两个关键 bug：

### Bug 1：`underscore_to_dot` 默认值错误
- **文件**：`eval/bfcl_eval/constants/model_config.py`
- **问题**：默认值为 `True`，导致 `func.name` 在验证时被转换为 `func_name`，触发大量假阳性的 `wrong_func_name` 错误
- **影响**：所有模型准确率虚低约 25 个百分点
- **修复**：改为 `False`

### Bug 2：Java/JS 类型字符串化
- **文件**：`eval/bfcl_ast_checker.py`
- **问题**：Java/JS 类型检查器期望字符串输入（为文本输出设计），但结构化 JSON 工具调用产生原生类型值（int、bool 等）
- **影响**：所有 Java/JavaScript 类别准确率为 0%
- **修复**：添加 `_stringify_value()` 自动将原生类型转为字符串

---

## 3. 评测模型

| 模型 | 类型 | 说明 |
|------|------|------|
| Gemma4 26B | 本地 (Ollama) | 原生工具调用，启用思考模式 |
| Volcengine Auto (ark-code-latest) | 云端 API | 火山引擎编程模型 |
| Doubao-Seed-2.0-Code | 云端 API | 分别测试开启/关闭思考模式 |
| Claude Opus 4.6 | Sub-agent | 文本格式工具调用，100 条抽样 |

---

## 4. 总体准确率对比

![总体准确率](01_overall_accuracy.png)

---

## 5. 错题本方法

### 流程
1. **Pool 推理**：用目标模型在 pool.jsonl（训练数据）上跑推理
2. **错误分类**：用 AST checker 将预测分类为不同错误类型
3. **K-Medoids 选择**：按错误类型比例选出 k 个最具代表性的错题
4. **构建提示**：将「错误→纠正」配对作为 Few-Shot 上下文
5. **评估**：在留出的测试集上测试

### 提示结构（纯错题模式，效果最佳）
```
System: You are a helpful assistant that calls tools accurately...

[纠错示例 1]
User: <问题>
Assistant: <错误的 tool_call>
User: "这个工具调用有误：<错误描述>。正确的调用应该是：<正确调用>"
Assistant: <正确的 tool_call>

[纠错示例 2-5]
...

[目标问题]
User: <实际要回答的问题>
```

---

## 6. 消融实验

![消融实验](02_ablation_study.png)

| 变体 | 准确率 | 相比 Zero-shot |
|------|--------|---------------|
| 纯错题（5道纠错） | **79.0%** | **+11.0pp** |
| 交替排列（正→错交替） | **79.0%** | **+11.0pp** |
| 混合 5+5（正确+纠错） | 76.3% | +8.3pp |
| Zero-shot 基线 | 68.0% | — |
| 纯正面示例（5道正确） | 64.7% | -3.3pp |

### 关键洞察
1. **纠错示例是唯一的提升驱动力**——正面示例完全没有贡献
2. **纯正面 Few-Shot 反而降低性能**——可能因为上下文长度开销而没有提供有用信号
3. 提升**集中在并行调用类别**（从 0% 提升到 80%+）

---

## 7. 各类别分析

![类别热力图](03_category_heatmap.png)

![错题本提升幅度](04_error_notebook_delta.png)

### Volcengine Auto：纯错题效果
- **parallel**：0% → 81.5%（+81.5pp）
- **parallel_multiple**：18.3% → 84.8%（+66.5pp）
- **live_parallel_multiple**：37.5% → 100%（+62.5pp）
- simple/live_simple 类别有轻微下降（-2 到 -5pp）

### Gemma4 26B：错题本效果
- **parallel**：78.3% → 97.2%（+18.9pp）
- **sql**：16.7% → 36.8%（+20.2pp）
- **simple**：88.3% → 96.4%（+8.1pp）
- 但 **live_parallel**：80% → 50%（-30pp）——针对错误模型的纠错示例反而造成混乱

### 为什么两个模型效果差异大？
Gemma4 在并行调用上本身就很强（78.3% zero-shot），因此针对并行的纠错示例带来的是边际递减效应，有时甚至产生干扰。而 Volcengine Auto 在并行上是 0%，纠错示例带来了变革性的提升。

**启示**：错题本在针对模型**实际薄弱环节**时效果最佳。

---

## 8. 错误分布

![Volcengine 错误分布](05_volcengine_errors.png)

![Gemma4 错误分布](06_gemma4_errors.png)

---

## 9. Claude Opus 4.6 基准测试（100 条抽样）

Claude 在分层抽样的 100 条子集上通过 sub-agent 评估（文本格式工具调用，无标准答案泄漏）。

| 模型 | 准确率（同 100 条） |
|------|-------------------|
| **Claude Opus 4.6** | **87.0%** |
| Gemma4 26B | 75.0% |
| Doubao-Code（思考） | 69.0% |
| Volcengine Auto | 60.0% |

Claude 在 simple、multiple、parallel、java、live_parallel 和 live_parallel_multiple 类别上达到 100%。

---

## 10. 思考模式分析

| 模型 | 开启思考 | 关闭思考 | 差异 |
|------|---------|---------|------|
| Doubao-Seed-2.0-Code | 73.9% | 67.6% | +6.3pp |

思考模式提升了工具调用准确率，尤其是需要多步推理的并行调用。

---

## 11. 结论与后续方向

### 结论
1. **错题本方法有效**：Volcengine Auto 提升 +11.0pp（68% → 79.0%）
2. **只有纠错示例起作用**：正面示例不必要甚至有害
3. **针对弱点效果最佳**：模型有明确失败模式时（如并行调用 0%）效果最显著
4. **修复了两个重大评估 bug**：之前所有模型的准确率数字都偏低约 25pp

### 可能的后续方向
- [ ] 在完整 782 条测试集上运行 Claude Opus（待 API 额度）
- [ ] 尝试不同的 k 值（k=3, k=8, k=10）
- [ ] 按类别定制错题本（如针对 sql 的专项纠错）
- [ ] 仅用 Gemma4 弱项类别的错误来构建错题本
- [ ] 在 Doubao-Seed-2.0-Code 上测试错题本
