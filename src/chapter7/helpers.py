import logging
import re
import tiktoken
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import APIError  # Import specific error for better handling

# === Configure Production-Level Logging ===
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def process_stream(response) -> str:
    """处理流式响应并打印"""
    full_content = []
    print(f"\n{'=' * 15} AI Stream Start {'=' * 15}")
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
    print(f"\n{'=' * 15} AI Stream End {'=' * 15}\n")
    return "".join(full_content)


# === LLM Interaction (Hardened with Dependency Injection) ===
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_llm_robust(
    system_prompt, user_prompt, client, generation_model, json_mode=False
):
    """
    A centralized function to handle all LLM interactions with retries.
    UPGRADE: Now requires the 'client' and 'generation_model' objects to be passed in.
    """
    logging.info("尝试调用大语言模型...")
    try:
        response_format = {"type": "json_object"} if json_mode else {"type": "text"}
        # UPGRADE: Uses the passed-in client and model name for the API call.
        response = client.chat.completions.create(
            model=generation_model,
            response_format=response_format,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        )
        logging.info("大语言模型调用成功。")
        return process_stream(response)
    except APIError as e:
        logging.error(f"call_llm_robust中的OpenAI API错误: {e}")
        raise e
    except Exception as e:
        logging.error(f"call_llm_robust中发生意外错误: {e}")
        raise e


# === Embeddings (Hardened with Dependency Injection) ===
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embedding(text, client, embedding_model, embedding_dim):
    """
    Generates embeddings for a single text query with retries.
    UPGRADE: Now requires the 'client' and 'embedding_model' objects.
    """
    text = text.replace("\n", " ")
    try:
        # UPGRADE: Uses the passed-in client and model name.
        response = client.embeddings.create(
            input=[text],
            model=embedding_model,
            dimensions=embedding_dim,
        )
        return response.data[0].embedding
    except APIError as e:
        logging.error(f"get_embedding中的OpenAI API错误: {e}")
        raise e
    except Exception as e:
        logging.error(f"get_embedding中发生意外错误: {e}")
        raise e


# === Model Context Protocol (MCP) ===
def create_mcp_message(sender, content, metadata=None):
    """Creates a standardized MCP message."""
    return {
        "protocol_version": "2.0 (Context Engine)",
        "sender": sender,
        "content": content,
        "metadata": metadata or {},
    }


# === Pinecone Interaction (Hardened with Dependency Injection) ===
def query_pinecone(
    query_text, namespace, top_k, index, client, embedding_model, embedding_dim
):
    """
    Embeds the query text and searches the specified Pinecone namespace.
    UPGRADE: Now requires 'index', 'client', and 'embedding_model' objects.
    """
    logging.info(f"正在查询Pinecone命名空间 '{namespace}'...")
    try:
        # UPGRADE: Passes the necessary dependencies down to get_embedding.
        query_embedding = get_embedding(
            query_text,
            client=client,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
        )
        # UPGRADE: Uses the passed-in index object for the query.
        response = index.query(
            vector=query_embedding,
            namespace=namespace,
            top_k=top_k,
            include_metadata=True,
        )
        logging.info("Pinecone查询成功。")
        return response["matches"]
    except Exception as e:
        logging.error(f"Pinecone查询错误 (命名空间: {namespace}): {e}")
        raise e


# === Context Management Utility (New) ===
def count_tokens(text, model="gpt-4"):
    """Counts the number of tokens in a text string for a given model."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback for models that might not be in the tiktoken registry
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def helper_sanitize_input(text):
    """
    A simple sanitization function to detect and flag potential prompt injection
    patterns.
    Returns the text if clean, or raises a ValueError if a threat is detected.
    """
    injection_patterns = [
        r"ignore previous instructions",
        r"ignore all prior commands",
        r"you are now in.*mode",
        r"act as",
        r"print your instructions",
        r"sudo|apt-get|yum|pip install",
    ]
    for pattern in injection_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            raise ValueError(f"Input sanitization failed. Potential threat detected.")
    return text
