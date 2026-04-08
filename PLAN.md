# 基于互预测性的示例选择 - 实施计划

## 项目概述

验证"基于互预测性的示例选择"方法能否提升本地模型的tool calling能力。
对比三个条件：zero-shot / random k-shot / selected k-shot（本方法）。

## 基础信息

- 模型：Gemma 4 26B-A4B
  - 退火scoring：vLLM + cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit（conda env: vllm_bench）
  - 最终评估：Ollama + gemma4:26b（GGUF Q4_K_M）
- 数据集：BFCL V4（使用V3继承的非agentic single-turn类别：simple/parallel/multiple/live）
- Embedding模型：Qwen3-Embedding-0.6B（预处理阶段，GPU运行）
- 硬件：RTX 4090 24GB

## 方法论限制（报告中需注明）

退火scoring使用vLLM + AWQ量化，最终评估使用Ollama + GGUF量化。
两者是同一base模型的不同4-bit量化版本，可能存在细微偏差。

---

## 步骤 1：下载BFCL数据，了解实际格式

### 1.1 环境准备
- 在 vllm_bench conda环境中安装 bfcl 包（`pip install bfcl`）
- 或直接从HuggingFace下载数据：`gorilla-llm/Berkeley-Function-Calling-Leaderboard`

### 1.2 下载数据
- 从HuggingFace下载：`gorilla-llm/Berkeley-Function-Calling-Leaderboard`
- gold answer在 `possible_answer/` 目录中
- 初期只使用single-turn类别：
  - V1 non-live: simple, multiple, parallel, parallel_multiple（AST + Exec）
  - V2 live: live_simple, live_multiple, live_parallel, live_parallel_multiple, live_irrelevance, live_relevance
- 暂不纳入multi-turn类别（格式复杂，few-shot构造方式不同，可作为后续扩展）

### 1.3 格式分析
- 读取几条样本，确认字段结构：question, function, answer
- 确认gold answer的JSON格式（函数名、参数结构）
- 确认不同类别之间格式的差异
- 记录总数据量

### 1.4 数据划分
- 将数据分为两部分：
  - **pool**（约70%）：用于退火搜索subset + random baseline采样
  - **test**（约30%）：用于最终三条件对比评估
- 划分时保证各类别在pool和test中比例一致（stratified split）
- 使用固定random seed（如42）确保可复现
- 保存划分结果到 `data/pool.jsonl` 和 `data/test.jsonl`

### 输出
- `data/` 目录下的原始数据和划分后数据
- `notebooks/01_data_exploration.py`：数据格式分析脚本

---

## 步骤 1.5：实现LLM请求缓存层

### 1.5.1 SQLite缓存设计
- 脚本：`cache/llm_cache.py`
- 使用本地SQLite数据库：`cache/llm_cache.db`
- 表结构：
  ```sql
  CREATE TABLE llm_cache (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      backend TEXT NOT NULL,          -- 'ollama' 或 'vllm'
      model TEXT NOT NULL,            -- 模型名称
      prompt_hash TEXT NOT NULL,      -- prompt内容的SHA256哈希
      params_hash TEXT NOT NULL,      -- 请求参数（temperature/max_tokens等）的哈希
      prompt TEXT NOT NULL,           -- 完整prompt内容
      response TEXT NOT NULL,         -- 完整返回数据（JSON字符串）
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(backend, model, prompt_hash, params_hash)
  );
  ```
- 缓存逻辑：
  - 请求前：用(backend, model, prompt_hash, params_hash)查询缓存
  - 命中：直接返回缓存的response，跳过LLM调用
  - 未命中：调用LLM，将结果存入数据库后返回
- 提供统一接口，Ollama和vLLM的调用都经过此缓存层

### 1.5.2 注意事项
- temperature=0时缓存有效（确定性输出）；temperature>0时应跳过缓存
- vLLM的prompt_logprobs结果也需缓存（退火过程中大量重复scoring）
- 始终缓存完整response（包括完整prompt_logprobs数据），不做提取后压缩
- 预计总缓存空间约1GB
- 提供手动清除缓存的方法（按backend/model/时间等条件）

### 输出
- `cache/llm_cache.py`
- `cache/llm_cache.db`（运行时生成）

---

## 步骤 2：实现zero-shot baseline + AST评估流程

### 2.1 构建Ollama推理pipeline
- 脚本：`eval/ollama_generate.py`
- 功能：读取test.jsonl，逐条（或并发）发送给Ollama gemma4:26b
- Ollama API参数：
  - model: "gemma4:26b"
  - temperature: 0
  - num_ctx: 动态计算（根据实际prompt token数 + 输出buffer，步骤1.3确认后设定）
  - num_predict: 1024（thinking + tool call）
- 并发策略：
  - 单条prompt较短时，Ollama可同时处理多个请求（OLLAMA_NUM_PARALLEL）
  - 需根据实际prompt长度和24GB VRAM测试最大并发数（预计2-4并发）
  - 先实现串行版本确保正确性，再加并发优化
- 输出：`results/zero_shot_responses.jsonl`（每条包含id, prompt, response, latency）

### 2.2 Prompt格式
- 使用BFCL标准的tool calling prompt格式
- 包含：system prompt（工具定义）+ user query
- 需要研究BFCL官方使用的prompt template，尽量与之一致

### 2.3 AST评估器
- 脚本：`eval/ast_evaluator.py`
- 功能：
  - 从模型response中提取tool call JSON（需处理thinking部分的剥离）
  - 与gold answer做AST结构匹配：
    - 函数名完全匹配
    - 参数名完全匹配
    - 参数值匹配（考虑类型转换、字符串规范化）
  - 参考BFCL官方的评估逻辑（可能直接复用bfcl包的checker）
- 输出：每条的pass/fail + 总体accuracy

### 2.4 运行zero-shot测试
- 在test set上运行完整的zero-shot评估
- 记录：总accuracy、各类别accuracy、运行时间
- 保存结果：`results/zero_shot_eval.json`

### 输出
- `eval/ollama_generate.py`
- `eval/ast_evaluator.py`
- `results/zero_shot_responses.jsonl`
- `results/zero_shot_eval.json`

---

## 步骤 3：预处理embedding

### 3.1 安装依赖
- 在 vllm_bench 环境中安装 sentence-transformers
- 下载 Qwen3-Embedding-0.6B 模型

### 3.2 计算embedding
- 脚本：`preprocess/compute_embeddings.py`
- 对pool中所有样本的question字段计算embedding
- 注意Qwen3-Embedding的instruction格式要求
- GPU运行，预期几秒完成

### 3.3 缓存
- 保存embedding矩阵为numpy文件：`data/pool_embeddings.npy`
- 保存对应的id映射：`data/pool_ids.json`
- 预计算pairwise余弦相似度矩阵（如果pool不太大的话）

### 输出
- `preprocess/compute_embeddings.py`
- `data/pool_embeddings.npy`
- `data/pool_ids.json`

---

## 步骤 4：实现teacher forcing打分

### 4.1 vLLM推理封装
- 脚本：`scoring/teacher_forcing.py`
- 功能：给定context + gold_answer，计算 log P(gold_answer | context, thinking)
- 两步走实现：
  - **Pass 1（生成thinking）**：
    - 输入：tool_definitions + [k-1 examples] + question_i（apply_chat_template）
    - 生成：max_tokens=600, temperature=0
    - 提取thinking部分（<think>...</think>标签之间的内容）
  - **Pass 2（teacher forcing）**：
    - 输入：Pass 1的完整输入 + thinking内容 + gold_json_i
    - 参数：SamplingParams(max_tokens=1, prompt_logprobs=1)
    - 提取gold_json_i对应位置的logprob求和
- 返回：log_prob_sum, mean_log_prob, gold_len_tokens

### 4.2 Tokenization处理
- 正确识别context与gold_answer的token边界
- 使用 llm.get_tokenizer() 分别编码context和full_prompt
- gold token起始位置 = len(context_ids)

### 4.3 单元测试
- 用几条BFCL样本手动验证：
  - gold answer的logprob是否合理（不是-inf，不是0）
  - 格式正确的答案logprob > 格式错误的答案logprob
  - 有相关few-shot时logprob > 无few-shot时logprob

### 输出
- `scoring/teacher_forcing.py`
- `tests/test_teacher_forcing.py`

---

## 步骤 5：实现互预测性得分

### 5.1 互预测性计算
- 脚本：`scoring/mutual_predictability.py`
- 功能：给定subset S = {(x_1,y_1), ..., (x_k,y_k)}，计算：
  P_θ(S) = Σ_{i=1}^{k} log P_θ(y_i | tool_defs, S\{i}, x_i)
- 实现：
  - 对每个i，构建context = tool_definitions + S中除第i个之外的所有example
  - 调用teacher_forcing.score()获取log_prob
  - 求和

### 5.2 性能考虑
- 每次计算P_θ(S)需要2k次vLLM调用（k次thinking生成 + k次teacher forcing）
- k=5时：10次调用/iteration
- k=10时：20次调用/iteration
- 优化：当退火只flip一个样本时，只需要重算受影响的score（增量更新）

### 输出
- `scoring/mutual_predictability.py`

---

## 步骤 6：实现逻辑一致性惩罚（多样性）

### 6.1 多样性惩罚函数
- 脚本：`scoring/diversity_penalty.py`
- 功能：给定subset S，计算：
  I(S) = Σ_i Σ_{j≠i} cosine_similarity(emb(x_i), emb(x_j))
- 使用步骤3中预计算的embedding
- 纯numpy运算，无需调用模型

### 6.2 总体scoring函数
- 脚本：`scoring/objective.py`
- 功能：
  U(S) = α · P_θ(S) - I(S)
- α为超参数，需要根据两项的量级初步calibrate

### 输出
- `scoring/diversity_penalty.py`
- `scoring/objective.py`

---

## 步骤 7：实现模拟退火

### 7.1 退火算法
- 脚本：`search/simulated_annealing.py`
- 状态：当前subset S（k个样本的index集合）
- 邻域操作：随机替换S中的一个样本（用pool中另一个样本替换）
- 接受准则：
  - ΔU > 0：直接接受
  - ΔU < 0：以概率 exp(ΔU / T) 接受
- 温度调度：指数衰减，T(t) = T_0 · decay^t
- 超参数：
  - k（subset大小）：待定，建议先试5
  - T_0（初始温度）：根据U(S)的量级calibrate
  - decay：0.995 或类似
  - max_iterations：500-1000
  - α：互预测性与多样性的权重平衡

### 7.2 增量更新优化
- 当替换一个样本时，只重新计算：
  - 新样本的teacher forcing score
  - 新样本对其他k-1个样本的影响（它们的context变了）
  - 新样本涉及的pairwise相似度
- 实际上替换1个样本需要重算全部k个互预测性score（因为所有样本的context都变了）
- 但多样性惩罚只需要更新涉及被替换样本的pair

### 7.3 日志和可视化
- 记录每次迭代的U(S), P_θ(S), I(S), temperature, 接受/拒绝
- 保存到 `results/annealing_log.jsonl`
- 可视化脚本：`analysis/plot_annealing.py`

### 7.4 运行退火
- 保存最终选出的subset：`results/selected_subset.json`

### 输出
- `search/simulated_annealing.py`
- `results/annealing_log.jsonl`
- `results/selected_subset.json`
- `analysis/plot_annealing.py`

---

## 步骤 8：实现random k-shot baseline

### 8.1 随机采样
- 脚本：`eval/random_baseline.py`
- 从pool中随机采样k个样本作为few-shot examples
- 重复多次（如30次）取平均，消除采样方差
- 每次采样使用不同random seed

### 8.2 评估
- 对每组随机样本，在test set上用Ollama运行评估
- 计算平均accuracy和标准差
- 保存结果：`results/random_kshot_eval.json`

### 输出
- `eval/random_baseline.py`
- `results/random_kshot_responses/`（多组结果）
- `results/random_kshot_eval.json`

---

## 步骤 9：最终三条件对比评估

### 9.1 Selected k-shot评估
- 使用步骤7选出的subset作为few-shot examples
- 在test set上用Ollama运行评估
- 保存结果：`results/selected_kshot_eval.json`

### 9.2 汇总对比
- 脚本：`analysis/compare_conditions.py`
- 对比三个条件的：
  - 总体accuracy
  - 各BFCL类别的accuracy
  - 统计显著性检验（McNemar test或类似）
- 生成对比表格和图表

### 9.3 分析
- selected比random好多少？
- 哪些类别提升最大？
- 退火选出的examples有什么共同特征？
- 多样性项的影响分析（ablation：去掉多样性项，只用互预测性）

### 输出
- `results/selected_kshot_eval.json`
- `analysis/compare_conditions.py`
- `analysis/final_report_data.json`

---

## 目录结构

```
mutual_predictability_selection/
├── PLAN.md                          # 本文件
├── cache/                           # LLM请求缓存
│   ├── llm_cache.py                 # 缓存层实现
│   └── llm_cache.db                 # SQLite数据库（运行时生成）
├── data/                            # 数据
│   ├── raw/                         # BFCL原始数据
│   ├── pool.jsonl                   # 退火搜索用
│   ├── test.jsonl                   # 最终评估用
│   ├── pool_embeddings.npy          # 预计算embedding
│   └── pool_ids.json                # embedding对应的样本id
├── preprocess/                      # 预处理
│   └── compute_embeddings.py
├── scoring/                         # 打分
│   ├── teacher_forcing.py           # teacher forcing核心实现
│   ├── mutual_predictability.py     # 互预测性得分
│   ├── diversity_penalty.py         # 多样性惩罚
│   └── objective.py                 # 总体目标函数 U(S)
├── search/                          # 搜索
│   └── simulated_annealing.py       # 模拟退火
├── eval/                            # 评估
│   ├── ollama_generate.py           # Ollama批量推理
│   ├── ast_evaluator.py             # AST结构匹配评估
│   └── random_baseline.py           # random k-shot baseline
├── analysis/                        # 分析
│   ├── plot_annealing.py            # 退火过程可视化
│   └── compare_conditions.py        # 三条件对比
├── results/                         # 结果
│   ├── zero_shot_responses.jsonl
│   ├── zero_shot_eval.json
│   ├── annealing_log.jsonl
│   ├── selected_subset.json
│   ├── selected_kshot_eval.json
│   └── random_kshot_eval.json
└── tests/                           # 测试
    └── test_teacher_forcing.py
```

---

## 待确认事项（运行后确定）

1. **subset大小k**：初步设为5，后续可做k=3/5/10的对比实验
2. **α超参数**：需要先跑步骤4和6，观察P_θ和I的量级后calibrate
3. **退火迭代次数**：初步500次，根据收敛情况调整
4. **random baseline重复次数**：30次取平均
5. **num_ctx具体值**：步骤1.3分析实际数据后确定
6. **Ollama并发数**：步骤2实现后根据VRAM占用测试确定
