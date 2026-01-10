import copy
import os
import sys
import json
import logging
import textwrap
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tenacity import retry, wait_random_exponential, stop_after_attempt
import tiktoken
from tqdm import tqdm
from IPython.display import display, Markdown

# 加载环境变量
load_dotenv()

# ==========================================
# 1. 配置与工具层 (Configuration & Utils)
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("System")


class AppConfig:
    """集中管理应用程序配置"""

    def __init__(self):
        self.api_key = os.getenv("MODELSCOPE_API_KEY")
        self.base_url = os.getenv(
            "MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1"
        )
        self.model_id = os.getenv("MODEL_ID", "deepseek-ai/DeepSeek-V3.2")
        self.embedding_model = "Qwen/Qwen3-Embedding-8B"
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_env = "us-east-1"
        self.index_name = "genai-mas-mcp-ch3"
        self.embedding_dim = int(os.getenv("EMBEDDING_DIM", 1024))

        self.ns_context = os.getenv("NAMESPACE_CONTEXT", "context_blueprints")
        self.ns_knowledge = os.getenv("NAMESPACE_KNOWLEDGE", "knowledge_base")

        self.max_retries = int(os.getenv("MAX_RETRIES", 3))
        self.retry_delay = int(os.getenv("RETRY_DELAY", 2))

        if not self.api_key or not self.pinecone_api_key:
            raise ValueError("Critical Config Error: API Keys are missing.")


@dataclass
class McpMessage:
    """标准消息协议封装"""

    sender: str
    content: str | Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    protocol_version: str = "1.0"

    @property
    def dict_content(self) -> Dict[str, Any]:
        """安全地获取字典格式的内容，自动处理 JSON 解析"""
        if isinstance(self.content, dict):
            return self.content
        try:
            return json.loads(self.content)
        except (json.JSONDecodeError, TypeError):
            return {}

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# ==========================================
# 2. 基础服务层 (Infrastructure Services)
# ==========================================


class LLMService:
    """封装 LLM 交互逻辑，包含重试机制和流式输出"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def get_embedding(self, text: str) -> List[float]:
        """生成文本嵌入向量"""
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(
            input=[text],
            model=self.config.embedding_model,
            dimensions=self.config.embedding_dim,
        )
        return response.data[0].embedding

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成嵌入向量"""
        texts = [t.replace("\n", " ") for t in texts]
        response = self.client.embeddings.create(
            input=texts,
            model=self.config.embedding_model,
            dimensions=self.config.embedding_dim,
        )
        return [item.embedding for item in response.data]

    def chat_completion(
        self, system_prompt: str, user_content: str, json_mode: bool = True
    ) -> str:
        """执行对话请求"""
        logger.info(f"Calling LLM ({self.config.model_id})...")
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_id,
                response_format=(
                    {"type": "json_object"} if json_mode else {"type": "text"}
                ),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                stream=True,
            )

            return self._process_stream(response)
        except Exception as e:
            logger.error(f"LLM Call Failed: {e}")
            raise

    def _process_stream(self, response) -> str:
        """处理流式响应并打印"""
        full_content = []
        print(f"\n{'='*15} AI Stream Start {'='*15}")
        for chunk in response:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content or ""
            reasoning = getattr(chunk.choices[0].delta, "reasoning_content", "")

            if reasoning:
                print(reasoning, end="", flush=True)
            if content:
                print(content, end="", flush=True)
            full_content.append(content)
        print(f"\n{'='*15} AI Stream End {'='*15}\n")
        return "".join(full_content)


class VectorDBService:
    """封装 Pinecone 向量数据库的所有操作"""

    def __init__(self, config: AppConfig, llm_service: LLMService):
        self.config = config
        self.llm = llm_service
        self.pc = Pinecone(api_key=config.pinecone_api_key)
        self.index = self._initialize_index()

    def _initialize_index(self):
        """初始化或连接索引"""
        if self.config.index_name not in self.pc.list_indexes().names():
            logger.info(f"Creating index '{self.config.index_name}'...")
            self.pc.create_index(
                name=self.config.index_name,
                dimension=self.config.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=self.config.pinecone_env),
            )
            time.sleep(5)  # 等待就绪
        return self.pc.Index(self.config.index_name)

    def clear_namespace(self, namespace: str):
        """清空指定命名空间"""
        stats = self.index.describe_index_stats()
        if (
            namespace in stats.namespaces
            and stats.namespaces[namespace].vector_count > 0
        ):
            logger.info(f"Clearing namespace: {namespace}")
            self.index.delete(delete_all=True, namespace=namespace)
            time.sleep(2)  # 简单等待删除生效

    def upsert_data(self, vectors: List[Dict], namespace: str):
        """上传向量数据"""
        if vectors:
            self.index.upsert(vectors=vectors, namespace=namespace)
            logger.info(f"Upserted {len(vectors)} vectors to {namespace}")

    def query(self, query_text: str, namespace: str, top_k: int = 1) -> List[Any]:
        """语义检索"""
        try:
            vector = self.llm.get_embedding(query_text)
            response = self.index.query(
                vector=vector,
                namespace=namespace,
                top_k=top_k,
                include_metadata=True,
            )
            return response["matches"]
        except Exception as e:
            logger.error(f"Pinecone Query Error: {e}")
            return []


# ==========================================
# 3. 智能体层 (Agent Layer)
# ==========================================


class BaseAgent(ABC):
    def __init__(self, llm_service, vector_db=None):
        self.llm = llm_service
        self.vector_db = vector_db

    @abstractmethod
    def process_task(self, message: McpMessage) -> McpMessage:
        pass


class ContextLibrarianAgent(BaseAgent):
    """负责查找并提供语义蓝图/上下文模板"""

    def __init__(self, llm_service: LLMService, vector_db: VectorDBService):
        super().__init__(llm_service)
        self.vector_db = vector_db
        self.namespace = os.getenv("NAMESPACE_CONTEXT") or "ContextLibrary"

    def process_task(self, message: McpMessage) -> McpMessage:
        logger.info("[Librarian] Activating...")
        query = message.dict_content.get("intent_query", "")

        matches = self.vector_db.query(query, self.namespace, top_k=1)

        if matches:
            match = matches[0]
            logger.info(
                f"[Librarian] Found blueprint: {match['id']} (Score: {match['score']:.2f})"
            )
            blueprint = match["metadata"]["blueprint_json"]
        else:
            logger.warning("[Librarian] No blueprint found, using default.")
            blueprint = json.dumps({"instruction": "Generate content neutrally."})

        return McpMessage(sender="Librarian", content={"blueprint": blueprint})


class ResearcherAgent(BaseAgent):
    """负责从知识库中检索事实并进行总结"""

    def __init__(self, llm_service: LLMService, vector_db: VectorDBService):
        super().__init__(llm_service)
        self.vector_db = vector_db
        self.namespace = os.getenv("NAMESPACE_KNOWLEDGE") or "KnowledgeStore"

    def process_task(self, message: McpMessage) -> McpMessage:
        logger.info("[Researcher] Activating...")
        topic = message.dict_content.get("topic_query", "")

        matches = self.vector_db.query(topic, self.namespace, top_k=3)
        if not matches:
            return McpMessage(
                sender="Researcher", content={"facts": "No specific data found."}
            )

        source_texts = [m["metadata"]["text"] for m in matches]

        # 总结检索到的内容
        sys_prompt = """You are an expert research synthesis AI.
        Synthesize the provided source texts into a concise, bullet-pointed summary relevant to the user's topic. Focus strictly on the facts provided in the sources. Do not add outside information."""

        user_prompt = f"Topic: {topic}\n\nSources:\n" + "\n\n---\n\n".join(source_texts)

        facts = self.llm.chat_completion(sys_prompt, user_prompt, json_mode=False)
        return McpMessage(sender="Researcher", content={"facts": facts})


class WriterAgent(BaseAgent):
    """根据蓝图和事实生成最终内容"""

    def process_task(self, message: McpMessage) -> McpMessage:
        logger.info("[Writer] Activating...")
        data = message.dict_content
        facts = data.get("facts")
        blueprint = data.get("blueprint")
        previous_content = data.get("previous_content")

        if not blueprint:
            raise ValueError("Writer requires 'blueprint' in the input content.")

        if facts:
            source_material = facts
            source_label = "RESEARCH FINDINGS"
        elif previous_content:
            source_material = previous_content
            source_label = "PREVIOUS CONTENT (For Rewriting)"
        else:
            raise ValueError("Writer requires either 'facts' or 'previous_content'")

        sys_prompt = f"""You are an expert content generation AI.
            Your task is to generate content based on the provided SOURCE MATERIAL.
            Crucially, you MUST structure, style, and constrain your output according to the rules defined in the SEMANTIC BLUEPRINT provided below.

            --- SEMANTIC BLUEPRINT (JSON) ---
            {blueprint}
            --- END SEMANTIC BLUEPRINT ---

            Adhere strictly to the blueprint's instructions, style guides, and goals. The blueprint defines HOW you write; the source material defines WHAT you write about.
            """

        user_prompt = f"""
            --- SOURCE MATERIAL ({source_label}) ---
            {source_material}
            --- END SOURCE MATERIAL ---

            Generate the content now, following the blueprint precisely.
            """

        output = self.llm.chat_completion(sys_prompt, user_prompt, json_mode=False)

        return McpMessage(sender="Writer", content={"output": output})


class AgentRegistry:
    def __init__(self) -> None:
        self.registry = {
            "Librarian": ContextLibrarianAgent,
            "Researcher": ResearcherAgent,
            "Writer": WriterAgent,
        }

    def get_handler(self, agent_name):
        """Retrieves the function associated with an agent name."""
        handler = self.registry.get(agent_name)
        if not handler:
            raise ValueError(f"Agent '{agent_name}' not found in registry")
        return handler

    def get_capabilities_description(self):
        """
        Returns a structured description of the agents for the Planner LLM.
        This is crucial for the Planner to understand how to use the agents.
        """
        return """
        Available Agents and their required inputs:

        1. AGENT: Librarian
           ROLE: Retrieves Semantic Blueprints (style/structure instructions).
           INPUTS:
             - "intent_query": (String) A descriptive phrase of the desired style or format.
           OUTPUT: The blueprint structure (JSON string).

        2. AGENT: Researcher
           ROLE: Retrieves and synthesizes factual information on a topic.
           INPUTS:
             - "topic_query": (String) The subject matter to research.
           OUTPUT: Synthesized facts (String).

        3. AGENT: Writer
           ROLE: Generates or rewrites content by applying a Blueprint to source material.
           INPUTS:
             - "blueprint": (String/Reference) The style instructions (usually from Librarian).
             - "facts": (String/Reference) Factual information (usually from Researcher). Use this for new content generation.
             - "previous_content": (String/Reference) Existing text (usually from a prior Writer step). Use this for rewriting/adapting content.
           OUTPUT: The final generated text (String).
        """


AGENT_TOOLKIT = AgentRegistry()
print("智能体注册初始化成功")


class Planner(BaseAgent):
    """
    Analyzes the goal and generates a structured Execution Plan using the LLM.
    """

    print("[Engine: Planner] Analyzing goal and generating execution plan...")

    def process_task(self, message: McpMessage) -> McpMessage:
        data = message.dict_content
        goal = data.get("goal")
        capabilities = data.get("capabilities")
        sys_prompt = f"""
        You are the strategic core of the Context Engine. Analyze the user's high-level goal and create a structured Execution Plan using the available agents.

        --- AVAILABLE CAPABILITIES ---
        {capabilities}
        --- END CAPABILITIES ---

        INSTRUCTIONS:
        1. The plan MUST be a JSON list of objects, where each object is a "step".
        2. You MUST use Context Chaining. If a step requires input from a previous step, reference it using the syntax $$STEP_X_OUTPUT$$.
        3. Be strategic. Break down complex goals (like sequential rewriting) into distinct steps. Use the correct input keys ('facts' vs 'previous_content') for the Writer agent.

        EXAMPLE GOAL: "Write a suspenseful story about Apollo 11."
        EXAMPLE PLAN (JSON LIST):
        [
            {{"step": 1, "agent": "Librarian", "input": {{"intent_query": "suspenseful narrative blueprint"}}}},
            {{"step": 2, "agent": "Researcher", "input": {{"topic_query": "Apollo 11 landing details"}}}},
            {{"step": 3, "agent": "Writer", "input": {{"blueprint": "$$STEP_1_OUTPUT$$", "facts": "$$STEP_2_OUTPUT$$"}}}}
        ]

        EXAMPLE GOAL: "Write a technical report on Juno, then rewrite it casually."
        EXAMPLE PLAN (JSON LIST):
        [
            {{"step": 1, "agent": "Librarian", "input": {{"intent_query": "technical report structure"}}}},
            {{"step": 2, "agent": "Researcher", "input": {{"topic_query": "Juno mission technology"}}}},
            {{"step": 3, "agent": "Writer", "input": {{"blueprint": "$$STEP_1_OUTPUT$$", "facts": "$$STEP_2_OUTPUT$$"}}}},
            {{"step": 4, "agent": "Librarian", "input": {{"intent_query": "casual summary style"}}}},
            {{"step": 5, "agent": "Writer", "input": {{"blueprint": "$$STEP_4_OUTPUT$$", "previous_content": "$$STEP_3_OUTPUT$$"}}}}
        ]

        Respond ONLY with the JSON list.
        """
        plan_json = ""
        try:
            plan_json = self.llm.chat_completion(sys_prompt, goal, json_mode=True)
            plan = json.loads(plan_json)

            # 如果是 dict，尝试提取 plan 或 steps
            if isinstance(plan, dict):
                if "plan" in plan and isinstance(plan["plan"], list):
                    plan = plan["plan"]
                elif "steps" in plan and isinstance(plan["steps"], list):
                    plan = plan["steps"]
                else:
                    raise ValueError(
                        "Planner returned a dict, but missing 'plan' or 'steps' key."
                    )
            # 如果不是 list，直接报错
            elif not isinstance(plan, list):
                raise ValueError("Planner did not return a valid JSON list structure.")

            return McpMessage(sender="Planer", content={"plan": plan})
        except Exception as e:
            print(
                f"[Engine: Planner] Failed to generate a valid plan. Error: {e}. Raw LLM Output: {plan_json}"
            )
            raise e


def resolve_dependencies(input_params, state):
    """
    Helper function to replace $$REF$$ placeholders with actual data from the execution state.
    This implements Context Chaining.
    """
    # Use copy.deepcopy to ensure the original plan structure is not modified
    resolved_input = copy.deepcopy(input_params)

    # Recursive function to handle potential nested structures
    def resolve(value):
        if isinstance(value, str) and value.startswith("$$") and value.endswith("$$"):
            ref_key = value[2:-2]
            if ref_key in state:
                # Retrieve the actual data (string) from the previous step's output
                print(f"[Engine: Executor] Resolved dependency {ref_key}.")
                return state[ref_key]
            else:
                raise ValueError(
                    f"Dependency Error: Reference {ref_key} not found in execution state."
                )
        elif isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve(v) for v in value]
        return value

    return resolve(resolved_input)


class ExecutionTrace:
    """Logs the entire execution flow for debugging and analysis."""

    def __init__(self, goal):
        self.goal = goal
        self.plan = None
        self.steps = []
        self.status = "Initialized"
        self.final_output = None
        self.start_time = time.time()

    def log_plan(self, plan):
        self.plan = plan

    def log_step(self, step_num, agent, planned_input, mcp_output, resolved_input):
        """Logs the details of a single execution step."""
        self.steps.append(
            {
                "step": step_num,
                "agent": agent,
                # The raw input definitions from the plan (including $$REFS$$)
                "planned_input": planned_input,
                # Crucial for debugging: What exact context did the agent receive?
                "resolved_context": resolved_input,
                "output": mcp_output.content,
            }
        )

    def finalize(self, status, final_output=None):
        self.status = status
        self.final_output = final_output
        self.duration = time.time() - self.start_time

    def display_trace(self):
        """Displays the trace in a readable format."""
        display(
            Markdown(
                f"### Execution Trace\n**Goal:** {self.goal}\n**Status:** {self.status} (Duration: {self.duration:.2f}s)"
            )
        )
        if self.plan:
            # Display the raw plan JSON
            display(
                Markdown(f"#### Plan:\n```json\n{json.dumps(self.plan, indent=2)}\n```")
            )

        display(Markdown("#### Execution Steps:"))
        for step in self.steps:
            print(f"--- Step {step['step']}: {step['agent']} ---")
            print("  [Planned Input]:", step["planned_input"])
            # print("  [Resolved Context]:", textwrap.shorten(str(step['resolved_context']), width=150))
            print(
                "  [Output Snippet]:", textwrap.shorten(str(step["output"]), width=150)
            )
            print("-" * 20)


def context_engine(goal):
    """
    上下文引擎的入口点，管理计划和执行
    """
    print(f"\n=== [Context Engine] Starting New Task ===\n Goal: {goal}\n")
    trace = ExecutionTrace(goal)
    registry = AGENT_TOOLKIT
    config = AppConfig()
    llm_service = LLMService(config)
    vector_db = VectorDBService(config, llm_service)

    # 2. 数据注入 (实际生产中通常是单独的 Pipeline，这里为了演示保留)
    # Uncomment the line below to reset DB on run
    # setup_knowledge_base(config, vector_db, llm_service)
    try:
        capabilities = registry.get_capabilities_description()
        planner = Planner(llm_service)
        mcp_to_planner = McpMessage(
            sender="context_engine",
            content={"goal": goal, "capabilities": capabilities},
        )
        mcp_from_planner = planner.process_task(mcp_to_planner)
        plan = mcp_from_planner.dict_content.get("plan")
        trace.log_plan(plan)
    except Exception as e:
        trace.finalize("Failed during Planning")
        return None, trace

    # Phase 2: Execute
    # State stores the raw outputs (strings) of each step: { "STEP_X_OUTPUT": data_string }
    state = {}

    if not plan:
        return

    for step in plan:
        step_num = step.get("step")
        agent_name = step.get("agent")
        planned_input = step.get("input")

        print(f"\n[Engine: Executor] 开始步骤 {step_num}: {agent_name}")

        try:
            handler = registry.get_handler(agent_name)

            resolved_input = resolve_dependencies(planned_input, state)
            if isinstance(resolved_input, dict):
                cur_agent = handler(llm_service, vector_db)
                mcp_output = cur_agent.process_task(
                    McpMessage(sender="Engine", content=resolved_input)
                )

                output_data = mcp_output.dict_content
                """
                每个输出都存储在状态字典中，形成短期记忆，供后续步骤调用。
                """
                state[f"STEP_{step_num}_OUTPUT"] = output_data
                """
                追踪器会记录过程中的每个细节：调用了哪个智能体、它收到了什么输入、上下文是如何解析的以及产生了什么结果。
                """
                trace.log_step(
                    step_num, agent_name, planned_input, mcp_output, resolved_input
                )
                print(f"[Engine: Executor] Step {step_num} completed.")
        except Exception as e:
            error_message = f"Execution failed at step {step_num} ({agent_name}): {e}"
            print(f"[Engine: Executor] ERROR: {error_message}")
            trace.finalize(f"Failed at Step {step_num}")
            # Return the trace for debugging the failure
            return None, trace

    final_output = state.get(f"STEP_{len(plan)}_OUTPUT")
    trace.finalize("Success", final_output)
    print("\n=== [Context Engine] Task Complete ===")
    return final_output, trace


# ==========================================
# 4. 数据准备工具 (Data Ingestion Setup)
# ==========================================


def setup_knowledge_base(
    config: AppConfig, vector_db: VectorDBService, llm: LLMService
):
    """(可选) 初始化向量数据库数据。仅在初次运行或需要重置时调用"""
    print(">>> Setting up Knowledge Base...")

    # A. 准备上下文蓝图 (Blueprints)
    vector_db.clear_namespace(config.ns_context)
    blueprints = _get_raw_blueprints()  # 获取原始数据（此处省略具体数据定义以节省空间）

    vectors_ctx = []
    for item in tqdm(blueprints, desc="Embedding Blueprints"):
        vec = llm.get_embedding(item["description"])
        vectors_ctx.append(
            {
                "id": item["id"],
                "values": vec,
                "metadata": {
                    "blueprint_json": item["blueprint"],
                    "description": item["description"],
                },
            }
        )
    vector_db.upsert_data(vectors_ctx, config.ns_context)

    # B. 准备知识库事实 (Knowledge Chunks)
    vector_db.clear_namespace(config.ns_knowledge)
    raw_text = _get_raw_knowledge_text()  # 获取原始文本
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(raw_text)

    chunks = []
    chunk_size, overlap = 400, 50
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_str = (
            tokenizer.decode(tokens[i : i + chunk_size]).replace("\n", " ").strip()
        )
        if chunk_str:
            chunks.append(chunk_str)

    vectors_knw = []
    # 批量处理 embedding
    batch_size = 50
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding Knowledge"):
        batch = chunks[i : i + batch_size]
        embeddings = llm.get_embeddings_batch(batch)
        for j, emb in enumerate(embeddings):
            vectors_knw.append(
                {"id": f"chunk_{i+j}", "values": emb, "metadata": {"text": batch[j]}}
            )
    vector_db.upsert_data(vectors_knw, config.ns_knowledge)
    print(">>> Knowledge Base Ready.\n")


def _get_raw_blueprints():
    # 返回原始代码中的 context_blueprints 列表
    # 为保持简洁，此处引用原数据结构
    return [
        {
            "id": "blueprint_suspense_narrative",
            "description": "A precise Semantic Blueprint designed to generate suspenseful and tense narratives, suitable for children's stories. Focuses on atmosphere, perceived threats, and emotional impact. Ideal for creative writing.",
            "blueprint": json.dumps(
                {
                    "scene_goal": "Increase tension and create suspense.",
                    "style_guide": "Use short, sharp sentences. Focus on sensory details (sounds, shadows). Maintain a slightly eerie but age-appropriate tone.",
                    "participants": [
                        {
                            "role": "Agent",
                            "description": "The protagonist experiencing the events.",
                        },
                        {
                            "role": "Source_of_Threat",
                            "description": "The underlying danger or mystery.",
                        },
                    ],
                    "instruction": "Rewrite the provided facts into a narrative adhering strictly to the scene_goal and style_guide.",
                }
            ),
        },
        {
            "id": "blueprint_technical_explanation",
            "description": "A Semantic Blueprint designed for technical explanation or analysis. This blueprint focuses on clarity, objectivity, and structure. Ideal for breaking down complex processes, explaining mechanisms, or summarizing scientific findings.",
            "blueprint": json.dumps(
                {
                    "scene_goal": "Explain the mechanism or findings clearly and concisely.",
                    "style_guide": "Maintain an objective and formal tone. Use precise terminology. Prioritize factual accuracy and clarity over narrative flair.",
                    "structure": [
                        "Definition",
                        "Function/Operation",
                        "Key Findings/Impact",
                    ],
                    "instruction": "Organize the provided facts into the defined structure, adhering to the style_guide.",
                }
            ),
        },
        {
            "id": "blueprint_casual_summary",
            "description": "A goal-oriented context for creating a casual, easy-to-read summary. Focuses on brevity and accessibility, explaining concepts simply.",
            "blueprint": json.dumps(
                {
                    "scene_goal": "Summarize information quickly and casually.",
                    "style_guide": "Use informal language. Keep it brief and engaging. Imagine explaining it to a friend.",
                    "instruction": "Summarize the provided facts using the casual style guide.",
                }
            ),
        },
    ]


def _get_raw_knowledge_text():
    return """
Space exploration is the use of astronomy and space technology to explore outer space. The early era of space exploration was driven by a "Space Race" between the Soviet Union and the United States. The launch of the Soviet Union's Sputnik 1 in 1957, and the first Moon landing by the American Apollo 11 mission in 1969 are key landmarks.

The Apollo program was the United States human spaceflight program carried out by NASA which succeeded in landing the first humans on the Moon. Apollo 11 was the first mission to land on the Moon, commanded by Neil Armstrong and lunar module pilot Buzz Aldrin, with Michael Collins as command module pilot. Armstrong's first step onto the lunar surface occurred on July 20, 1969, and was broadcast on live TV worldwide. The landing required Armstrong to take manual control of the Lunar Module Eagle due to navigational challenges and low fuel.

Juno is a NASA space probe orbiting the planet Jupiter. It was launched on August 5, 2011, and entered a polar orbit of Jupiter on July 5, 2016. Juno's mission is to measure Jupiter's composition, gravitational field, magnetic field, and polar magnetosphere to understand how the planet formed. Juno is the second spacecraft to orbit Jupiter, after the Galileo orbiter. It is uniquely powered by large solar arrays instead of RTGs (Radioisotope Thermoelectric Generators), making it the farthest solar-powered mission.

A Mars rover is a remote-controlled motor vehicle designed to travel on the surface of Mars. NASA JPL managed several successful rovers including: Sojourner, Spirit, Opportunity, Curiosity, and Perseverance. The search for evidence of habitability and organic carbon on Mars is now a primary NASA objective. Perseverance also carried the Ingenuity helicopter.
"""


# ==========================================
# 5. 主程序入口 (Main Execution)
# ==========================================


def main():
    try:
        goal_1 = "Write a short, suspenseful scene for a children's story about the Apollo 11 moon landing, highlighting the danger."

        # Run the Context Engine
        # Ensure the Pinecone index is populated (from Ch3 notebook) for this to work.
        result_1, trace_1 = context_engine(goal_1)

        if result_1:
            print("\n******** FINAL OUTPUT 1 **********\n")
            print(result_1)
            print("\n\n" + "=" * 50 + "\n\n")
            # Optional: Display the trace to see the engine's process
            # trace_1.display_trace()

    except Exception as e:
        logger.exception("Fatal Error in Main Loop")
        sys.exit(1)


if __name__ == "__main__":
    main()
