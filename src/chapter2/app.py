import os
import sys
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, Any
from dotenv import load_dotenv

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


# ==========================================
# 1. 领域模型层 (Domain Models)
# ==========================================


@dataclass
class McpMessage:
    sender: str
    content: str
    # field 是 Python 标准库 dataclasses 模块提供的工具函数，专门用于精细化配置 @dataclass 装饰的类的属性。
    # 若直接写 metadata: Dict = {}，会导致所有实例会共享同一个字典（Python 可变默认值的坑）；
    # field(default_factory=dict) 表示：每次创建 McpMessage 对象时，都会调用 dict() 生成一个新的空字典，避免共享问题。
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


# ==========================================
# 2. 服务层 (Service Layer)
# ==========================================


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


# ==========================================
# 3. 业务逻辑层 (Agents)
# ==========================================


class BaseAgent(ABC):
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    # 子类必须实现的统一接口
    @abstractmethod
    def process_task(self, message: McpMessage) -> McpMessage:
        pass


class ResearcherAgent(BaseAgent):
    def __init__(self, llm_service: LLMService):
        super().__init__(llm_service)
        self.knowledge_base = {
            "mediterranean diet": "地中海饮食包括大量水果、蔬菜、全谷物、豆类、坚果，以及橄榄油作为主要脂肪来源。"
        }

    def process_task(self, message: McpMessage) -> McpMessage:
        topic = message.content.strip().lower()
        context = self.knowledge_base.get(topic)

        if not context:
            return McpMessage(
                sender="Researcher", content="", metadata={"status": "not_found"}
            )

        summary = self.llm.chat_completion("请用一句话总结以下事实信息内容。", context)
        return McpMessage(sender="Researcher", content=summary)


class WriterAgent(BaseAgent):
    def process_task(self, message: McpMessage) -> McpMessage:
        start_time = time.time()
        system_prompt = (
            "你是一个专业的内容创作者。请根据研究要点写一篇短小精悍且吸引人的博客文章（约100字）。"
            "请使用中文，并配上一个引人注目的标题。"
        )
        blog_post = self.llm.chat_completion(system_prompt, message.content)
        return McpMessage(
            sender="Writer",
            content=blog_post,
            metadata={
                "prev_sender": message.sender,
                "time": f"{time.time()-start_time:.2f}s",
            },
        )


class ValidatorAgent(BaseAgent):
    def process_task(self, message: McpMessage) -> McpMessage:
        # --- 核心修复：解析 JSON 字符串 ---
        try:
            data = json.loads(message.content)
            source_summary = data.get("summary", "")
            draft_post = data.get("draft", "")
        except Exception as e:
            logger.error(f"Validator 解析 JSON 失败: {e}")
            return McpMessage(sender="Validator", content="fail: 数据格式错误")

        system_prompt = """
        你是一名严谨的核查员。判断 '草稿(DRAFT)' 与 '原始研究摘要(SOURCE SUMMARY)' 是否事实一致。
        - 如果一致且没有虚假内容，只回复单词: pass
        - 如果草稿包含了原始摘要中没有的虚假信息，回复: fail 并用一句话说明原因。
        """
        validation_context = (
            f"SOURCE SUMMARY:\n{source_summary}\n\nDRAFT:\n{draft_post}"
        )
        result = self.llm.chat_completion(system_prompt, validation_context)
        return McpMessage(sender="Validator", content=result)


class Orchestrator(BaseAgent):
    def process_task(self, message: McpMessage) -> McpMessage:
        # 1. 调度研究员
        researcher = ResearcherAgent(self.llm)
        mcp_res = researcher.process_task(message)
        if not mcp_res.validate() or not mcp_res.content:
            return McpMessage(
                sender="Orchestrator", content="工作流失败：研究员未找到相关信息。"
            )

        res_summary = mcp_res.content
        logger.info("[Orchestrator] 研究阶段完成。")

        # 2. 迭代撰稿与验证
        max_revisions = 2
        final_output = "无法生成通过验证的文章。"
        feedback = ""

        for i in range(max_revisions):
            logger.info(f"[Orchestrator] 正在进行第 {i+1} 次创作尝试...")

            # 准备创作上下文
            writer_input = res_summary
            if feedback:
                writer_input += f"\n\n请根据此反馈修改草稿: {feedback}"

            # 调度作家
            writer = WriterAgent(self.llm)
            mcp_draft = writer.process_task(
                McpMessage(sender="Orchestrator", content=writer_input)
            )

            # 调度核查员 (发送 JSON 格式内容)
            val_payload = json.dumps(
                {"summary": res_summary, "draft": mcp_draft.content}, ensure_ascii=False
            )
            validator = ValidatorAgent(self.llm)
            mcp_val = validator.process_task(
                McpMessage(sender="Orchestrator", content=val_payload)
            )

            if "pass" in mcp_val.content.lower():
                logger.info("[Orchestrator] 验证通过！")
                final_output = mcp_draft.content
                break
            else:
                feedback = mcp_val.content
                logger.warning(f"[Orchestrator] 验证未通过，反馈: {feedback}")

        return McpMessage(sender="Orchestrator", content=final_output)


# ==========================================
# 4. 主程序入口
# ==========================================


def main():
    try:
        config = AppConfig()
        llm = LLMService(config)
        orchestrator = Orchestrator(llm)

        input_msg = McpMessage(sender="User", content="Mediterranean Diet")
        print(f"任务启动: {input_msg.content}")

        final_msg = orchestrator.process_task(input_msg)

        print("\n" + "#" * 30)
        print("最终结果：")
        print(final_msg.content)
        print("#" * 30)

    except Exception as e:
        logger.exception("程序发生致命错误")
        sys.exit(1)


if __name__ == "__main__":
    main()
