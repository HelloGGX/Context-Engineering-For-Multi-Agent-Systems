import os
from openai import OpenAI
from pinecone import Pinecone
from tenacity import retry, stop_after_attempt, wait_random_exponential
import tiktoken
from registry import AppConfig
from tqdm import tqdm

tokenizer = tiktoken.get_encoding("cl100k_base")
config = AppConfig()
client = OpenAI(base_url=config.base_url, api_key=config.api_key)
pc = Pinecone(api_key=config.pinecone_api_key)
index = pc.Index(config.index_name)


def chunk_text(text, chunk_size=400, overlap=50):
    tokens = tokenizer.encode(text)
    chunks = []  # 存储最终拆分的文本块
    # 循环逻辑：从token序列的0位置开始，按“步长=chunk_size - overlap”迭代
    for i in range(0, len(tokens), chunk_size - overlap):
        # 1. 截取当前块的token（从i到i+chunk_size，避免超出总长度）
        chunk_tokens = tokens[i : i + chunk_size]
        # 2. Token→文本解码（将token序列还原为人类可读的文本）
        chunk_text = tokenizer.decode(chunk_tokens)
        # 3. 基础清理（避免空行、多余空格影响检索精度）
        chunk_text = chunk_text.replace("\n", " ").strip()  # 换行符→空格，去除首尾空白
        # 4. 过滤空块（避免因拆分逻辑产生的空文本）
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embeddings_batch(texts, model=config.embedding_model):
    """Generates embeddings for a batch of texts using OpenAI, with retries."""
    # OpenAI expects the input texts to have newlines replaced by spaces
    texts = [t.replace("\n", " ") for t in texts]
    response = client.embeddings.create(
        input=texts, model=model, dimensions=config.embedding_dim
    )
    return [item.embedding for item in response.data]


if not os.path.exists("nasa_documents"):
    os.makedirs("nasa_documents")

knowledge_base = {}
doc_dir = "nasa_documents"
for filename in os.listdir(doc_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(doc_dir, filename), "r") as f:
            knowledge_base[filename] = f.read()

print(
    f"📚 Loaded {len(knowledge_base)} documents into the knowledge base."
)  # We use sample data related to space exploration.

print(f"\nProcessing and uploading Context Library to namespace: {config.ns_knowledge}")

batch_size = 100
total_vectors_uploaded = 0

for doc_name, doc_content in knowledge_base.items():
    knowledge_chunks = chunk_text(doc_content)

    for i in tqdm(
        range(0, len(knowledge_chunks), batch_size), desc=f" Uploading {doc_name}"
    ):
        batch_texts = knowledge_chunks[i : i + batch_size]
        batch_embeddings = get_embeddings_batch(batch_texts)
        batch_vectors = []
        for j, embedding in enumerate(batch_embeddings):
            chunk_id = f"{doc_name}_chunk_{total_vectors_uploaded + j}"

            batch_vectors.append(
                {
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": {"text": batch_texts[j], "source": doc_name},
                }
            )
        index.upsert(vectors=batch_vectors, namespace=config.ns_knowledge)

    total_vectors_uploaded += len(knowledge_chunks)

print(
    f"\n✅ Successfully uploaded {total_vectors_uploaded} knowledge vectors from {len(knowledge_base)} documents."
)
