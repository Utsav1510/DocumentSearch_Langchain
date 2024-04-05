from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("pd.pdf")
pages=loader.load_and_split()

import transformers
import sentence_transformers
import huggingface_hub
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)


from langchain_community.vectorstores import FAISS
faiss_index=FAISS.from_documents(pages,embeddings)
docs=faiss_index.similarity_search("Conductor",k=1)
for doc in docs:
      print(str(doc.metadata["page"]) + ":", doc.page_content[:300])