# AI Agents 路线图：从零基础到大师 🧠

![AI Agents logo](./docs/logo.png)  

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Focus](https://img.shields.io/badge/Focus-AI_Agents-green?style=for-the-badge)](./docs)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)](./LICENSE)
[![Stars](https://img.shields.io/github/stars/HelloGGX/AI-Agents-Zero-to-Hero?style=for-the-badge)](https://github.com/HelloGGX/AI-Agents-Zero-to-Hero/stargazers)

> 一个全面的、实践导向的AI Agents学习路线图，从零基础起步，受roadmap.sh启发。构建实用技能，帮助你获得第一份AI工作。

英文版请参阅 [README.md](./README.md)。

## 项目概述

这个开源仓库是你掌握AI Agents的结构化指南。无论你是完全新手，还是缺乏实际经验的AI工程师，这个项目记录了一个逐步的学习历程。它涵盖从基础到高级主题，包括代码示例、教程和项目，帮助你构建简历组合以获得工作机会。

核心焦点：将理论知识转化为可行动的技能。不再需要分散的资源——一切都在一处，实现系统化学习。

![AI Agents 路线图示意图](./docs/roadmap-diagram.png)  


## 适合人群

- **初学者**：零基础知识？从这里开始，获得清晰、渐进的路径。
- **求职者**：获得实际经验，轻松通过面试并获得AI职位。
- **自学者**：那些愿意投资高级资源进行更深入、指导性学习的人。

核心内容免费开源。对于高级视频课程、独家项目和社区支持，请查看我们的[会员选项](https://patreon.com/HelloGGX) 或 [课程](https://teachable.com/HelloGGX)。

## 核心功能

- **结构化路线图**：分为阶段，镜像roadmap.sh（例如，LLM基础、代理架构、构建代理）。
- **代码示例**：基于Python的概念实现，如ReAct循环、工具调用和RAG集成。
- **实践项目**：构建真实代理（例如，个人助理、数据分析器），用于简历展示。
- **资源**：精选链接、速查表和测验，用于自我评估。
- **更新**：基于2026年AI趋势的定期添加（例如，新模型、伦理指南）。

## 为什么选择这个项目？

- **解决痛点**：从“演示陷阱”到工程化设计，涵盖面试主题如代理幻觉、多代理协调和超长上下文。
- **工程实践**：强调解耦、事件驱动模式和可扩展性，实现可维护代码。
- **开源学习**：章节特定的代码路径，便于fork和实验。

## 学习路线图

按照这个分阶段方法构建专业知识：

| 阶段 | 涵盖主题 | 资源 |
|------|----------|------|
| **阶段1: 先决条件** | 后端基础、Git、REST API、LLM基础（Transformer、Tokenization、Embeddings、RAG基础）。 | `./docs/prerequisites.md`，代码在 `./src/prerequisites/` |
| **阶段2: AI Agents 101** | 什么是代理/工具？代理循环（感知、推理、行动、观察）。Prompt Engineering技巧。 | `./docs/agents-101.md`，示例在 `./src/agents-101/` |
| **阶段3: 工具与内存** | 工具定义、示例（Web Search、Code Execution）。代理内存（短期/长期、RAG、总结）。 | `./docs/tools-memory.md`，实现在 `./src/tools-memory/` |
| **阶段4: 架构** | ReAct、Chain of Thought、MCP、RAG代理、DAG/Tree-of-Thought。 | `./docs/architectures.md`，高级代码在 `./src/architectures/` |
| **阶段5: 构建与框架** | 手动构建（LLM API、Function Calling）。框架（LangChain、AutoGen、CrewAI）。 | `./docs/building.md`，项目在 `./src/projects/` |
| **阶段6: 评估与安全** | 测试指标、调试工具（LangSmith）。伦理（Prompt Injection、偏差防护）。 | `./docs/evaluation-security.md`，工具在 `./src/evaluation/` |

对于互动测验和视频演练，请升级到高级访问。

## 入门指南

### 先决条件
- Python 3.10+
- 基本命令行知识
- OpenAI API密钥（用于示例）

### 安装
1. 克隆仓库：
   ```bash
   git clone https://github.com/HelloGGX/AI-Agents-Zero-to-Hero
   cd ai-agents-roadmap
   ```
2. 设置环境：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. 配置 `.env`：
   ```
   OPENAI_API_KEY=your-key-here
   ```

### 快速示例
运行一个基本代理演示：
```bash
python src/agents-101/simple-agent.py
```
这演示了一个简单任务的ReAct循环。

探索 `./src/projects/` 以获取完整构建。

## 贡献指南

我们欢迎贡献来改进路线图！
- Fork并创建PR以进行增强（例如，新示例、修复）。
- 遵循PEP8风格；使用 `pytest` 测试。
- 详见 [CONTRIBUTING.md](./CONTRIBUTING.md) 指南。

## 社区与支持

- **讨论**：加入 [GitHub Discussions](https://github.com/HelloGGX/ai-agents-roadmap/discussions) 以提问。
- **高级社区**：通过 [Patreon](https://patreon.com/HelloGGX) 访问Q&A和直播会话。
- **联系**：Gavin  
  - X/Twitter: [@gavincoding](https://x.com/gavincoding?s=21)  
  - 邮箱: gavin_cat@outlook.com

## 相关资源

- 原始灵感： [roadmap.sh AI Agents](https://roadmap.sh/ai-agents)
- 框架： [LangChain](https://github.com/langchain-ai/langchain), [CrewAI](https://github.com/joaomdmoura/crewAI)
- 其他路线图：来自roadmap.sh的AI Engineer、MLOps

## 许可证

MIT许可证。详见 [LICENSE](./LICENSE)。

---

如果这个仓库对你的学习有帮助，请Star ⭐️ 或分享给他人。对于加速学习的高级内容，请探索我们的[课程和会员](https://teachable.com/HelloGGX)。让我们共同普及AI Agents教育！🚀