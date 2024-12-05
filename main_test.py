import numpy as np
import faiss
from langchain_chroma import Chroma
from langchain_community.llms import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import os
import json

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = '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'
os.environ["OPENAI_API_KEY"] = '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'

# Load the JSON document
json_file_path = 'data.json'
with open(json_file_path, 'r') as file:
    json_data = json.load(file)

# Assuming json_data is a list of dictionaries
#original_documents = [Document(page_content=str(item['id']), metadata=item) for item in json_data]
# Assuming json_data is a list of strings or dictionaries as determined
original_documents = [Document(page_content=str(item['id']), metadata=item) if isinstance(item, dict) else Document(page_content=item, metadata={'id': item}) for item in json_data]


# Load embeddings and create FAISS index
product_embeddings = np.load('embeddings.npy')
dimension = product_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(product_embeddings)

# User query
user_question = 'Im using a MIG welding torch for some general metalwork, and Ive heard that using the right contact tip size can make a big difference in performance. My current setup isnt working well for the type of welds I need. Can you recommend the correct contact tip that would work best for a 2.4 mm wire?'
new_data = [Document(page_content=user_question)]
vectorstore = Chroma.from_documents(documents=new_data, embedding=OpenAIEmbeddings()) 

# Get query embedding
result = vectorstore.get()
embeddings = vectorstore.get(ids=result['ids'], include=["embeddings"])['embeddings']
embeddings_array = np.array(embeddings)
question_embeddings = embeddings_array[0]

# Query the FAISS index
query_embedding = np.array([question_embeddings], dtype='float32')
k = 5
distances, indices = index.search(query_embedding, k)

# Output the results
for idx, distance in zip(indices[0], distances[0]):
    print(f"Product Index: {idx}, Distance: {distance}")

# Retrieve relevant documents
retrieved_documents = [original_documents[i] for i in indices[0]]

# Extract product details (ID and metadata)
retrieved_product_info = "\n".join([f"ID: {doc.metadata['id']}, Product: {doc.metadata.get('product_name', 'N/A')}" for doc in retrieved_documents])

# Initialize the LLM
llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Construct the prompt for the LLM
contextual_info = "\n".join([f"Product ID: {doc.metadata['id']}\nProduct Name: {doc.metadata.get('product_name', 'N/A')}\nDescription: {doc.metadata.get('description', 'N/A')}" for doc in retrieved_documents])
prompt = f"Given the following product information:\n{contextual_info}\n\nAnswer the user's question: {user_question}\n\nPlease include the product ID and a clickable URL to the product page on https://accurateco2spares.com/product/<product_id>  and take this <product_id> from the product inforamtion I mentieond above, there is a key named id of product. IMPORTANT: if you don't know the answer, then don't give any answer, only answer if you are sure."

# Get the LLM's response
response = llm(prompt)

# Print the response
print(response)
