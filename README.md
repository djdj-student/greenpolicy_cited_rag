# greenpolicy_cited_rag（GreenPolicyHub｜环保政策知识库｜强制引用可审计 RAG）

> 公开仓库说明：本仓库**不包含**任何真实法律/政策原文与运行日志。
> - 你的本地语料请放在 `data/raw/`（默认被 `.gitignore` 忽略，不会上传 GitHub）
> - Agent 运行记录与 bad case 记录写入 `data/runs/`、`data/badcases/`（同样默认忽略）
> - 背景图 `static/bg.jpg` 为可选本地文件（默认忽略）

## 目标与能力

在“双碳”目标与绿色转型加速背景下，环保政策更新频繁，企业合规/地方执行/公众查询存在“信息滞后、变更难追踪、解读不准”的痛点。本项目用**纯文本 RAG**构建环保政策知识库，提供：

- 实时问答：输入问题/场景 → 返回答案 + **强制引用原文片段**（降低幻觉）
- 变更追踪：新旧版本对比，高亮新增/删除/修改；支持“2026 年后政策变更汇总”
- 智能合规：基于 Agentic 路由（LangGraph）进行问答/变更/合规建议的自动分流

技术栈：LlamaIndex + bge-m3 Embedding + Chroma 向量库 +（可选）LangGraph Agent；评估脚手架预留给 RAGAS。

## 快速开始

### 1) 环境与依赖

建议 Python 3.10+。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如需评估：

```bash
pip install -r requirements-eval.txt
```

### 2) 配置（LLM / Embedding / 目录）

复制环境变量模板：

```bash
cp .env.example .env
```

- 默认使用 Ollama：`POLICYHUB_LLM_PROVIDER=ollama`，并配置 `OLLAMA_MODEL`
- 或使用 OpenAI：`POLICYHUB_LLM_PROVIDER=openai` + `OPENAI_API_KEY`
- 或使用 DeepSeek（OpenAI 兼容接口）：`POLICYHUB_LLM_PROVIDER=deepseek` + `DEEPSEEK_API_KEY`

如遇 `httpx.ReadTimeout`（常见于 Ollama 冷启动或生成较慢）：可在 `.env` 中提高 `OLLAMA_REQUEST_TIMEOUT`，并设置 `OLLAMA_KEEP_ALIVE=5m` 减少冷启动。

Embedding 默认 `BAAI/bge-m3`，首次运行会下载模型，可能较慢。

如果你的机器有 NVIDIA GPU（CUDA 可用），可在 `.env` 设置 `EMBEDDING_DEVICE=cuda` 加速“问题向量”的实时计算（否则默认用 CPU）。

### 3) 准备政策文本

把政策原文（`.txt` 或 `.md`）放到 `data/raw/`（该目录默认不会被提交到 GitHub）。

推荐命名：

- `政策名_版本标签_YYYY-MM-DD.txt`
- 例如：`生态环境法典_草案_2026-01-10.txt`

系统会用文件名推断：
- `policy_id`：第一个下划线字段
- `version_label`：日期前一段（如“草案/通过稿”）
- `version_date`：文件名中的 `YYYY-MM-DD`

另外，对“法典/法规类 Markdown”（存在大量 `第X条` 结构）会启用更适合的切分：
- 先读取 `# / ## / ### / ####` 作为编/分编/章/节上下文
- 再按 `第X条` 精确切分为“每条一段”的独立单元（条内超长再二次切分）
- 会清理文本中的 `（留空）` 排版标记，减少噪声
- `heading_path`（编/分编/章/节等结构路径）会作为元数据展示，但不会混入引用原文片段

### 4) 构建索引并启动 Demo

```bash
streamlit run app.py
```

说明：索引/Embedding 会在你首次点击“提问”时自动初始化（首次可能较慢）。

## 使用方式

### 问答（强制引用）

- 输入问题
- 选择“直接 RAG 提问”或“Agent 提问”
- 输出会展示答案与“引用原文”（含来源与原文片段，可核验）

> 备注：当前 UI Demo 以“问答 + 引用 + Agent 可复盘过程”为主；变更/合规能力可在后续按需要扩展到 UI。

## 目录结构

- `app.py`：Streamlit 单页 Demo
- `policyhub/`：核心库
  - `settings.py`：环境变量配置
  - `documents.py`：纯文本加载 + 元数据推断
  - `indexing.py`：LlamaIndex + Chroma 索引构建
  - `rag.py`：RAG 问答（带引用片段返回）
  - `agent.py`：Agent（可审计引用 + 自检重试 + 量化诊断 + 运行落盘）
- `data/raw/`：你的本地政策原文（默认忽略，不上传）

## 评估（RAGAS）

本仓库已预留评估依赖（`requirements-eval.txt`）。建议流程：

1) 准备评测集（问答对 + 期望引用范围），例如 `data/eval/questions.jsonl`
2) 编写 `scripts/run_eval.py` 读取评测集，跑 RAGAS 指标：faithfulness / answer relevancy / context precision

仓库内提供了一个最小示例：`data/eval/sample.jsonl`（基于占位文档，仅用于冒烟测试）。

> 当前仓库先提供“可运行的 Demo 与脚手架”，你的“召回率>85%、延迟<2s、bad case 优化记录”建议在你跑完真实政策集与评测集后，把结果补到本 README 的“实验记录”段落。

## 下一步可扩展

- 版本管理：接入官方发布渠道，自动抓取新版本并触发 diff
- 文档结构化：对条款/章节做规则抽取，提高对条款编号检索
- 行业工具：碳足迹计算、行业自查清单生成
