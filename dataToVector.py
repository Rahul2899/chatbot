import json
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = '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'
os.environ["OPENAI_API_KEY"] = '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'



# Read the JSON file
with open('merged_api.json', 'r') as file:
    data = json.load(file)


new_data = []

for product in data:
    doc = Document(page_content=str(product))
    new_data.append(doc)


# [
#     Document('produc 1 data') => [0.1, 0.2, 0.3, ...],
#     Document('produc 2 data') => [0.1, 0.2, 0.3, ...],
#     Document('produc 3 data') => [0.1, 0.2, 0.3, ...],
# ]

# print(new_data)

vectorstore = Chroma.from_documents(documents=new_data, embedding=OpenAIEmbeddings())

# with open('bk_vs.pkl', 'wb') as f:
#     pickle.dump(vectorstore, f)



# print(vectorstore.get())
result = vectorstore.get()

embeddings = vectorstore.get(ids=result['ids'], include=["embeddings"])['embeddings']
docs = vectorstore.get(ids=result['ids'], include=["embeddings", "documents"])['documents']


# print(embeddings[0])
# print(len(embeddings[0]))
embeddings_array = np.array(embeddings)
np.save('embeddings.npy', embeddings_array)

with open("docs_test.txt", "w") as text_file:
    text_file.write(str(docs))

# tsne = TSNE(n_components=2, random_state=42)
# embeddings_2d = tsne.fit_transform(embeddings_array)

# plt.figure(figsize=(10, 8))
# plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
# plt.title("t-SNE visualization of embeddings")
# plt.xlabel("t-SNE feature 1")
# plt.ylabel("t-SNE feature 2")
# plt.show()

# retriever = vectorstore.as_retriever(search_type="similarity")

# retriever_result = retriever.invoke("can you give me contact tip made up of copper?")

# print(retriever_result)