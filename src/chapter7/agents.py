# 4.The Specialist Agents (Fully Upgraded with Full Dependency Injection)
# -------------------------------------------------------------------------
# We now complete the upgrade of our specialist agents.
# The final step is to pass the configuration variables (like model names
# AND namespaces) as arguments, making the agents fully self-contained and
# removing all reliance on global variables.
# -------------------------------------------------------------------------
import json
import logging

from helpers import (
    call_llm_robust,
    create_mcp_message,
    helper_sanitize_input,
    query_pinecone,
)


# === 4.1. Context Librarian Agent (Upgraded) ===
# *** Added 'namespace_context' argument ***
def agent_context_librarian(
    mcp_message, client, index, embedding_model, namespace_context, embedding_dim
):
    """
    Retrieves the appropriate Semantic Blueprint from the Context Library.
    UPGRADE: Now also accepts embedding_model and namespace configuration.
    """
    logging.info("【图书管理员】已激活，正在分析意图...")
    try:
        requested_intent = mcp_message["content"].get("intent_query")

        if not requested_intent:
            raise ValueError("Librarian requires 'intent_query' in the input content.")

        # UPGRADE: Pass all necessary dependencies to the hardened helper function.
        results = query_pinecone(
            query_text=requested_intent,
            # *** Use the passed argument instead of the global variable ***
            namespace=namespace_context,
            top_k=1,
            index=index,
            client=client,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
        )

        if results:
            match = results[0]
            logging.info(
                f"【图书管理员】找到蓝图 '{match['id']}' (相似度得分: {match['score']:.2f})"
            )
            blueprint_json = match["metadata"]["blueprint_json"]
            content = {"blueprint_json": blueprint_json}
        else:
            logging.warning("【图书管理员】未找到特定蓝图，返回默认设置。")
            content = {
                "blueprint_json": json.dumps(
                    {"instruction": "Generate the content neutrally."}
                )
            }

        return create_mcp_message("Librarian", content)

    except Exception as e:
        logging.error(f"【图书管理员】发生错误: {e}")
        raise e


# === 4.2. Researcher Agent (Upgraded) ===
# *** 'namespace_knowledge' argument ***
def agent_researcher(
    mcp_message,
    client,
    index,
    generation_model,
    embedding_model,
    embedding_dim,
    namespace_knowledge,
):
    """
    Retrieves and synthesizes factual information from the Knowledge Base.
    UPGRADE: Now accepts all necessary model and namespace configurations.
    """
    logging.info("【研究员】已激活，正在调查主题...")
    try:
        topic = mcp_message["content"].get("topic_query")

        if not topic:
            raise ValueError("Researcher requires 'topic_query' in the input content.")

        # UPGRADE: Pass all dependencies to the Pinecone helper.
        results = query_pinecone(
            query_text=topic,
            # *** Use the passed argument instead of the global variable ***
            namespace=namespace_knowledge,
            top_k=3,
            index=index,
            client=client,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
        )

        if not results:
            logging.warning("【研究员】未找到相关信息。")
            return create_mcp_message("Researcher", "No data found on the topic.")

        sanitized_texts = []
        sources = set()
        for match in results:
            try:
                clean_text = helper_sanitize_input(match["metadata"]["text"])
                sanitized_texts.append(clean_text)
                if "source" in match["metadata"]:
                    sources.add(match["metadata"]["source"])
            except ValueError as e:
                logging.warning(
                    f"[Researcher] A retrieved chunk failed sanitization and was skipped. Reason: {e}"
                )
                continue

        if not sanitized_texts:
            logging.error(
                "[Researcher] All retrieved chunks failed sanitization. Aborting."
            )
            return create_mcp_message(
                "Researcher",
                {
                    "answer": "Could not generate a reliable answer as retrieved data was suspect.",
                    "sources": [],
                },
            )

        logging.info(f"【研究员】找到 {len(results)} 个相关片段，正在综合分析...")

        system_prompt = """You are an expert research synthesis AI. Your task is to provide a clear, factual answer to the user's topic based *only* on the 208 High-Fidelity RAG and Defense: The NASA-Inspired Research Assistant provided source texts. After the answer, you MUST provide a "Sources" section listing the unique source document names you used"""
        source_material = "\n\n---\n\n".join(sanitized_texts)
        user_prompt = f"Topic: {topic}\n\nSources:\n{source_material}\n\n---\nSynthesize your answer and list the source documents now."

        # UPGRADE: Pass all dependencies to the LLM helper.
        findings = call_llm_robust(
            system_prompt, user_prompt, client=client, generation_model=generation_model
        )

        """
        final_output输出像下面:
        
        The Juno mission, launched by NASA in 2011, has three primary scientific objectives:
            1. Determine Jupiter's water abundance and core mass to verify planetary formation theories;
            2. Analyze Jupiter's atmosphere, including composition and cloud movements;
            3. Map Jupiter's magnetic and gravity fields to explore its deep structure and polar magnetosphere.
            Juno is unique for its polar orbit, allowing it to avoid Jupiter's hazardous radiation belts.

            **Sources:**
            - juno_mission_overview.txt
            - perseverance_rover_tools.txt
        """

        final_output = f"{findings}\n\n**Sources:**\n" + "\n".join(
            [f"- {s}" for s in sorted(list(sources))]
        )

        return create_mcp_message("Researcher", {"facts": final_output})

    except Exception as e:
        logging.error(f"【研究员】发生错误: {e}")
        raise e


# === 4.3. Writer Agent (Upgraded) ===
def agent_writer(mcp_message, client, generation_model):
    """
    Combines research with a blueprint to generate the final output.
    UPGRADE: Now accepts the generation_model configuration.
    """
    logging.info("【作家】已激活，正在将蓝图应用于源材料...")
    try:
        blueprint_data = mcp_message["content"].get("blueprint")
        facts_data = mcp_message["content"].get("facts")
        previous_content_data = mcp_message["content"].get("previous_content")

        blueprint_json_string = (
            blueprint_data.get("blueprint_json")
            if isinstance(blueprint_data, dict)
            else blueprint_data
        )
        facts = None
        if isinstance(facts_data, dict):
            facts = facts_data.get("facts")
            if facts is None:
                facts = facts_data.get("summary")
            if facts is None:
                facts = facts_data.get("answer_with_sources")
        elif isinstance(facts_data, str):
            facts = facts_data

        previous_content = previous_content_data

        if not blueprint_json_string:
            raise ValueError("Writer requires 'blueprint' in the input content.")

        if facts:
            source_material = facts
            source_label = "SOURCE FACTS"
        elif previous_content:
            source_material = previous_content
            source_label = "PREVIOUS CONTENT (For Rewriting)"
        else:
            raise ValueError("Writer requires either 'facts' or 'previous_content'.")

        system_prompt = f"""You are an expert content generation AI.
        Your task is to generate content based on the provided SOURCE MATERIAL.
        Crucially, you MUST structure, style, and constrain your output according to the rules defined in the SEMANTIC BLUEPRINT provided below.

        --- SEMANTIC BLUEPRINT (JSON) ---
        {blueprint_json_string}
        --- END SEMANTIC BLUEPRINT ---

        Adhere strictly to the blueprint's instructions, style guides, and goals. The blueprint defines HOW you write; the source material defines WHAT you write about.
        """

        user_prompt = f"""
        --- SOURCE MATERIAL ({source_label}) ---
        {source_material}
        --- END SOURCE MATERIAL ---

        Generate the content now, following the blueprint precisely.
        """

        # UPGRADE: Pass all dependencies to the robust LLM call.
        final_output = call_llm_robust(
            system_prompt, user_prompt, client=client, generation_model=generation_model
        )

        return create_mcp_message("Writer", final_output)

    except Exception as e:
        logging.error(f"【作家】发生错误: {e}")
        raise e


def agent_summarizer(mcp_message, client, generation_model):
    logging.info("[Summarizer]已被激活，处理上下文中...")
    try:
        text_to_summarize = mcp_message["content"].get("text_to_summarize")
        summary_objective = mcp_message["content"].get("summary_objective")
        if not text_to_summarize or not summary_objective:
            raise ValueError(
                "Summarizer requires 'text_to_summarize' and 'summary_objective' in the input content."
            )
        system_prompt = """You are an expert summarization AI. Your task is to reduce the provided text to its essential points, guided by the user's specific objective. The summary must be concise, accurate, and directly address the stated
goal."""
        user_prompt = f"""--- OBJECTIVE ---\n{summary_objective}\n\n--- TEXT TO SUMMARIZE ---\n{text_to_summarize}\n--- END TEXT ---\n\nGenerate the summary now."""
        summary = call_llm_robust(
            system_prompt, user_prompt, client=client, generation_model=generation_model
        )
        return create_mcp_message("Summarizer", {"summary": summary})
    except Exception as e:
        logging.error(f"【总结者】发生错误: {e}")
        raise e
