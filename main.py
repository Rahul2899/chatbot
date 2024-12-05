import numpy as np
import faiss
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document


import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = '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'
os.environ["OPENAI_API_KEY"] = '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'



# Load the vectorstore using np.load
product_embeddings = np.load('embeddings.npy')

dimension = product_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)

index.add(product_embeddings)



# 
user_question = 'suggest me gaspre-heator for manual welding?'
new_data = [Document(page_content=user_question)]
vectorstore = Chroma.from_documents(documents=new_data, embedding=OpenAIEmbeddings())

result = vectorstore.get()
embeddings = vectorstore.get(ids=result['ids'], include=["embeddings"])['embeddings']
embeddings_array = np.array(embeddings)
question_embeddings = embeddings_array[0]


# Query the index
query_embedding = np.array([question_embeddings], dtype='float32')  # Example query embedding
k = 5  # Number of nearest neighbors to retrieve
distances, indices = index.search(query_embedding, k)

# Output the results
for idx, distance in zip(indices[0], distances[0]):
    print(f"Product Index: {idx}, Distance: {distance}")