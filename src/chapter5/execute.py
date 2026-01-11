# @title 7.Execution (with the Fully Upgraded and Corrected Engine)
# -------------------------------------------------------------------------
# This is the final step where we run our fully upgraded and hardened engine.
# We now define all configuration variables here and pass them into the
# context_engine call, completing our transition to a fully
# dependency-aware and self-contained system.
# -------------------------------------------------------------------------

# Ensure required display imports are present (for Colab/Jupyter)
import logging

from dotenv import load_dotenv
from engine import context_engine

# # 添加src目录到Python路径，以便正确导入模块
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))
from openai import OpenAI
from pinecone import Pinecone
from registry import AppConfig

# 加载环境变量
load_dotenv()


logging.info("******** Example 1: Executing the Hardened Engine **********\n")

goal_1 = "Write a short, suspenseful scene for a children's story about the Apollo 11 moon landing, highlighting the danger."

# --- Define all configuration variables before the call ---
config = AppConfig()
print(config)
client = OpenAI(base_url=config.base_url, api_key=config.api_key)
pc = Pinecone(api_key=config.pinecone_api_key)
# --- UPGRADE: Run the Context Engine with full dependency injection ---
# We now pass ALL dependencies—clients, configurations, and namespaces—into the engine.
result_1, trace_1 = context_engine(
    goal_1,
    client=client,
    pc=pc,
    index_name=config.index_name,
    generation_model=config.model_id,
    embedding_model=config.embedding_model,
    embedding_dim=config.embedding_dim,
    # *** FIX: Pass the namespaces into the engine ***
    namespace_context=config.ns_context,
    namespace_knowledge=config.ns_knowledge,
)

if result_1:
    logging.info("\n******** FINAL OUTPUT 1 **********")
    print(result_1)

    # Optional: Display the detailed trace for debugging
    # print(f"\nTrace Status: {trace_1.status}")
    # import pprint
    # pp = pprint.PrettyPrinter(indent=2)
    # pp.pprint(trace_1.steps)
