import os
import sys
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import tiktoken

# 引入必要的 OpenAI 异常类
from openai import OpenAI, APIError, APIConnectionError, RateLimitError

load_dotenv()
# ==========================================
# 0. 基础设施层 (Infrastructure Layer)
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("System")


class AppConfig:
    def __init__(self):
        # 替换为你的 API Key
        self.api_key = os.getenv("MODELSCOPE_API_KEY")
        self.base_url = os.getenv(
            "MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1"
        )
        self.model_id = os.getenv("MODEL_ID", "deepseek-ai/DeepSeek-V3.2")
        self.max_retries = int(os.getenv("MAX_RETRIES", 3))
        self.retry_delay = int(os.getenv("RETRY_DELAY", 2))
        self._validate()

    def _validate(self):
        if not self.api_key:
            raise ValueError("Critical Config Error: API Key is missing.", self.api_key)


@dataclass
class McpMessage:
    sender: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    protocol_version: str = "1.0"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    def validate(self) -> bool:
        """检查消息核心字段是否完整"""
        if not self.sender or self.content is None:
            logger.error(f"MCP 校验失败：来自 {self.sender} 的内容为空")
            return False
        return True


class LLMService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    def _print_stream(self, text: str):
        print(text, end="", flush=True)

    def chat_completion(
        self, system_prompt: str, user_content: str, enable_thinking: bool = True
    ) -> str:
        logger.info(f"正在调用 LLM... (最大重试次数: {self.config.max_retries})")

        for attempt in range(1, self.config.max_retries + 1):
            try:
                return self._execute_call(system_prompt, user_content, enable_thinking)
            except (APIConnectionError, RateLimitError, APIError) as e:
                logger.warning(f"尝试 {attempt}/{self.config.max_retries} 失败: {e}")
                if attempt == self.config.max_retries:
                    raise e
                time.sleep(self.config.retry_delay)
            except Exception as e:
                logger.error(f"不可恢复的错误: {e}")
                raise e
        return ""

    def _execute_call(
        self, system_prompt: str, user_content: str, enable_thinking: bool
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            stream=True,
            extra_body={"enable_thinking": enable_thinking},
        )

        full_content = []
        has_printed_separator = False
        print(f"\n{'='*15} LLM 响应开始 {'='*15}\n")

        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", "") or ""
            content = delta.content or ""

            if reasoning:
                self._print_stream(reasoning)
            elif content:
                if not has_printed_separator:
                    print(f"\n\n{'='*15} 最终回答 {'='*15}\n")
                    has_printed_separator = True
                self._print_stream(content)
                full_content.append(content)

        print(f"\n\n{'='*15} 响应结束 {'='*15}\n")
        return "".join(full_content)


print("xxxxxxxxx", os.getenv("PINECONE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
spec = ServerlessSpec(cloud="aws", region="us-east-1")

# INDEX_NAME 就是为这个 Pinecone 索引设定的 “专属名称”，相当于数据库中 “表名” 的角色 ——
# 通过这个名称，智能体（如 Researcher Agent、Context Librarian Agent）能精准定位到需要查询或写入数据的向量集合，确保数据流转的准确性。
INDEX_NAME = "genai-mas-mcp-ch3"
dimension_str = os.getenv("EMBEDDING_DIM")
if dimension_str is None:
    raise ValueError("EMBEDDING_DIM environment variable is not set.")
dimension = int(dimension_str)

if INDEX_NAME not in pc.list_indexes().names():
    print(f"Index '{INDEX_NAME}' not found, Creating new Serverless index...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=dimension,
        metric="cosine",  # 余弦相似度
        spec=spec,
    )
    while True:
        status = pc.describe_index(INDEX_NAME).status
        if status is not None and status["ready"]:
            break
        print("Waiting for index to be ready...")
        time.sleep(1)

    print("Index created successfully. It is new and empty")
else:
    print(
        f"Index '{INDEX_NAME}' already exists. Clearing namespaces for a fresh start.."
    )
    index = pc.Index(INDEX_NAME)
    namespaces_to_clear = [
        os.getenv("NAMESPACE_KNOWLEDGE"),
        os.getenv("NAMESPACE_CONTEXT"),
    ]
    for namespace in namespaces_to_clear:
        stats = index.describe_index_stats()
        if (
            namespace in stats.namespaces
            and stats.namespaces[namespace].vector_count > 0
        ):
            print(f"Clearing namespace '{namespace}'...")
            index.delete(delete_all=True, namespace=namespace)

            # 这里有个棘手的问题。清理函数可能是异步的，程序可能会进展得太快。
            # 这意味着，缓慢的删除操作可能在我们的上传开始后才完成，而那些新向量可能会被清除掉。
            # 这并非必然，但无疑是我们不愿承担的风险。因此，我们强制系统等待，只有当向量数量真正为零时才继续执行：
            while True:
                stats = index.describe_index_stats()
                if (
                    namespace not in stats.namespaces
                    or stats.namespaces[namespace].vector_count == 0
                ):
                    print(f"Namespace '{namespace}' cleared successfully.")
                    break
                print(f"Waiting for namespace '{namespace}' to clear...")
                time.sleep(5)  # Poll every 5 seconds
        else:
            print(
                f"Namespace '{namespace}' is already empty or does not exist. Skipping."
            )

index = pc.Index(INDEX_NAME)

context_blueprints = [
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
# @title 4.Data Preparation: The Knowledge Base (Factual RAG)
# -------------------------------------------------------------------------
# We use sample data related to space exploration.

knowledge_data_raw = """
Space exploration is the use of astronomy and space technology to explore outer space. The early era of space exploration was driven by a "Space Race" between the Soviet Union and the United States. The launch of the Soviet Union's Sputnik 1 in 1957, and the first Moon landing by the American Apollo 11 mission in 1969 are key landmarks.

The Apollo program was the United States human spaceflight program carried out by NASA which succeeded in landing the first humans on the Moon. Apollo 11 was the first mission to land on the Moon, commanded by Neil Armstrong and lunar module pilot Buzz Aldrin, with Michael Collins as command module pilot. Armstrong's first step onto the lunar surface occurred on July 20, 1969, and was broadcast on live TV worldwide. The landing required Armstrong to take manual control of the Lunar Module Eagle due to navigational challenges and low fuel.

Juno is a NASA space probe orbiting the planet Jupiter. It was launched on August 5, 2011, and entered a polar orbit of Jupiter on July 5, 2016. Juno's mission is to measure Jupiter's composition, gravitational field, magnetic field, and polar magnetosphere to understand how the planet formed. Juno is the second spacecraft to orbit Jupiter, after the Galileo orbiter. It is uniquely powered by large solar arrays instead of RTGs (Radioisotope Thermoelectric Generators), making it the farthest solar-powered mission.

A Mars rover is a remote-controlled motor vehicle designed to travel on the surface of Mars. NASA JPL managed several successful rovers including: Sojourner, Spirit, Opportunity, Curiosity, and Perseverance. The search for evidence of habitability and organic carbon on Mars is now a primary NASA objective. Perseverance also carried the Ingenuity helicopter.
"""

tokenizer = tiktoken.get_encoding("cl100k_base")


def chunk_text(text, chunk_size=400, overlap=50):
    tokens = tokenizer.encode(text)
    chunks = []
    #令牌序列：0—100—200—300—350—400—500—600—700—750—800—900—1000
    #第1片段：[0──────────────400)  （覆盖0-399令牌）
    #第2片段：        [350──────────────750)  （覆盖350-749令牌，与第1片段重叠50个）
    #第3片段：                [700──────────────1000)  （覆盖700-999令牌，与第2片段重叠50个）
    for i in range(0, len(tokens), chunk_size - overlap):
        