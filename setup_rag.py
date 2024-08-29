from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone, ServerlessSpec
import cohere
import os
import json

# load the API key from the .env file
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

index_name = "rmp-ai-assitant"

#check if an index list has been created
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
)

data = json.load(open("csvjson.json"))
processed_data = []


for review in data["reviews"]:
    response = co.embed(
        texts=[review["comments"]], 
        model='embed-multilingual-light-v3.0',
        input_type='search_query'
    )
    embedding = response.embeddings[0]
    processed_data.append({
        "values": embedding,
        "id": review["professor_name"],
        "metadata": {
            "school": review["school_name"],
            "rating": review["star_rating"],
            "race": review["race"],
        }
    })

index = pc.Index(index_name)
upsert_response = index.upsert(vectors=processed_data, namespace="ns1")
print(f"Upserted count: {upsert_response['upserted_count']}")
print(index.describe_index_stats())
