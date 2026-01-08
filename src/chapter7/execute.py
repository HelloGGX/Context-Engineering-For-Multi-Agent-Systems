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
from helpers import count_tokens
from engine import context_engine
from registry import AppConfig
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# === CONTROL DECK: Context Reduction Workflow ===

# 1. Define a large piece of text that would be expensive or too long
# to use as direct context for the Writer agent.
large_text_from_researcher = """
Juno is a NASA space probe orbiting the planet Jupiter. It was launched from Cape Canaveral Air Force Station on August 5, 2011, as part of the New Frontiers program. Juno entered a polar orbit of Jupiter on July 5, 2016, to begin a scientific investigation of the planet. After completing its primary mission, it received a mission extension. Juno's mission is to measure Jupiter's composition, gravitational field, magnetic field, and polar magnetosphere. It is also searching for clues about how the planet formed, including whether it has a rocky core, the amount of water present within the deep atmosphere, mass distribution, and its deep winds, which can reach speeds up to 618 kilometers per hour (384 mph). Juno is the second spacecraft to orbit Jupiter, after the nuclear-powered Galileo orbiter, which orbited from 1995 to 2003. Unlike all earlier spacecraft to the outer planets, Juno is powered by solar arrays, which are commonly used by satellites orbiting Earth and working in the inner Solar System, whereas radioisotope thermoelectric generators are commonly used for missions to the outer Solar System and beyond. For Juno, however, the three largest solar array wings ever deployed on a planetary probe play an integral role in stabilizing the spacecraft and generating power.
"""

# 2. Define a goal that requires both using the large text AND a creative step.
# This forces the Planner to recognize the need for summarization before writing.
goal = f"""First, summarize the following text about the Juno probe to extract only the key facts about its scientific mission and instruments. Then, using that summary, write a short, suspenseful scene for a children's story about the probe's dangerous arrival at Jupiter.

--- TEXT TO USE ---
{large_text_from_researcher}
"""


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

original_text = large_text_from_researcher

summarized_text = trace_1.steps[0]["output"]["summary"]

original_tokens = count_tokens(original_text)
summarized_tokens = count_tokens(summarized_text)
reduction_percentage = (1 - (summarized_tokens / original_tokens)) * 100

print("--- 上下文减少分析 ---")
print(f"原始文本Token数: {original_tokens}")
print(f"摘要文本Token数: {summarized_tokens}")
print(f"Token减少比例: {reduction_percentage:.1f}%")

if result_1:
    logging.info("\n******** 最终输出 1 **********")
    print(result_1)

    # Optional: Display the detailed trace for debugging
    # print(f"\nTrace Status: {trace_1.status}")
    # import pprint
    # pp = pprint.PrettyPrinter(indent=2)
    # pp.pprint(trace_1.steps)
