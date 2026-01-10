# @title 7.Execution (with the Fully Upgraded and Corrected Engine)
# -------------------------------------------------------------------------
# This is the final step where we run our fully upgraded and hardened engine.
# We now define all configuration variables here and pass them into the
# context_engine call, completing our transition to a fully
# dependency-aware and self-contained system.
# -------------------------------------------------------------------------

# Ensure required display imports are present (for Colab/Jupyter)
import logging
import os

# # 添加src目录到Python路径，以便正确导入模块
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from openai import OpenAI
from pinecone import Pinecone
from engine import context_engine
from registry import AppConfig
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# === CONTROL DECK: Context Reduction Workflow ===

goal = "What are the primary scientific objectives of the Juno mission, and what makes its design unique? Please cite your sources."


# --- Define all configuration variables before the call ---
config = AppConfig()
print(config)
client = OpenAI(base_url=config.base_url, api_key=config.api_key)
pc = Pinecone(api_key=config.pinecone_api_key)
# --- UPGRADE: Run the Context Engine with full dependency injection ---
# We now pass ALL dependencies—clients, configurations, and namespaces—into the engine.
result_1, trace_1 = context_engine(
    goal,
    client=client,
    pc=pc,
    index_name=config.index_name,
    generation_model=config.model_id,
    embedding_model=config.embedding_model,
    embedding_dim=config.embedding_dim,
    namespace_context=config.ns_context,
    namespace_knowledge=config.ns_knowledge,
)


if result_1:
    logging.info("\n******** 最终输出 1 **********")
    print(result_1)
