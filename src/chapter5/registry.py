# 5.The Agent Registry (Final Hardened Version)
# -------------------------------------------------------------------------
# We now make one final, crucial upgrade to the AgentRegistry.
# We must ensure all dependencies, including namespaces, are passed through.
# -------------------------------------------------------------------------

import logging
import os
import agents
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


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


class AgentRegistry:
    def __init__(self):
        self.registry = {
            "Librarian": agents.agent_context_librarian,
            "Researcher": agents.agent_researcher,
            "Writer": agents.agent_writer,
        }

    # *** Updated signature to include namespace_context and namespace_knowledge ***
    def get_handler(
        self,
        agent_name,
        client,
        index,
        generation_model,
        embedding_model,
        embedding_dim,
        namespace_context,
        namespace_knowledge,
    ):
        handler_func = self.registry.get(agent_name)
        if not handler_func:
            logging.error(f"Agent '{agent_name}' not found in registry.")
            raise ValueError(f"Agent '{agent_name}' not found in registry.")

        if agent_name == "Librarian":
            # *** Inject the context namespace into the Librarian handler ***
            return lambda mcp_message: handler_func(
                mcp_message,
                client=client,
                index=index,
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
                namespace_context=namespace_context,
            )
        elif agent_name == "Researcher":
            # *** Inject the knowledge namespace into the Researcher handler ***
            return lambda mcp_message: handler_func(
                mcp_message,
                client=client,
                index=index,
                generation_model=generation_model,
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
                namespace_knowledge=namespace_knowledge,
            )
        elif agent_name == "Writer":
            return lambda mcp_message: handler_func(
                mcp_message, client=client, generation_model=generation_model
            )
        else:
            return handler_func

    def get_capabilities_description(self):
        """
        Returns a structured description of the agents for the Planner LLM.
        UPGRADE: Now includes an explicit instruction to use exact key names.
        """
        return """
        Available Agents and their required inputs.
        CRITICAL: You MUST use the exact input key names provided for each agent.

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
             - "facts": (String/Reference) Factual information (usually from Researcher).
             - "previous_content": (String/Reference) Existing text for rewriting.
           OUTPUT: The final generated text (String).
        """


# Initialize the global toolkit.
AGENT_TOOLKIT = AgentRegistry()
logging.info("✅ Agent Registry initialized and fully upgraded.")
