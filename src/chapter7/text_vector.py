import re
from openai import OpenAI
from pinecone import Pinecone
from tenacity import retry, stop_after_attempt, wait_random_exponential
import tiktoken
from registry import AppConfig

tokenizer = tiktoken.get_encoding("cl100k_base")
config = AppConfig()
client = OpenAI(base_url=config.base_url, api_key=config.api_key)
pc = Pinecone(api_key=config.pinecone_api_key)
index = pc.Index(config.index_name)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embeddings_batch(texts, model=config.embedding_model):
    """Generates embeddings for a batch of texts using OpenAI, with retries."""
    # OpenAI expects the input texts to have newlines replaced by spaces
    texts = [t.replace("\n", " ") for t in texts]
    response = client.embeddings.create(
        input=texts, model=model, dimensions=config.embedding_dim
    )
    return [item.embedding for item in response.data]


query_embedding = get_embeddings_batch(["What is the Juno mission?"])[0]
results = index.query(
    vector=query_embedding,
    top_k=1,
    namespace=config.ns_knowledge,
    include_metadata=True,
)

if results["matches"]:
    top_match_metadata = results["matches"][0]["metadata"]
    print(top_match_metadata)
