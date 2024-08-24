import os
import pickle
from typing import List

from langchain_community.document_loaders import NotionDBLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import Document, VectorStoreIndex, Settings, PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike

NOTION_SECRET = ""
API_BASE = "http://{api}:5000/v1"
DB_IDS = []

Settings.context_window = 4096

template = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
"You are a helpful chatbot for resolving incidents at WeTransfer. Use the additional context below to formulate answers to questions:\n"
"---------------------\n"
"{context_str}"
"\n---------------------\n"
Given this information, please answer the question: {query_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
qa_template = PromptTemplate(template)


class NotionDBLoaderFix(NotionDBLoader):
    def load(self) -> List[Document]:
        page_summaries = self._retrieve_page_summaries(query_dict={"page_size": 100})

        return list(self.load_page(page_summary) for page_summary in page_summaries)


notion_docs = []
for i in range(3):
    path = f"data/{i}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            notion_docs.extend(pickle.load(f))
    else:
        loader = NotionDBLoaderFix(NOTION_SECRET, DB_IDS[i], request_timeout_sec=60)
        i_docs = loader.load()

        with open(path, "wb") as f:
            pickle.dump(i_docs, f)

        notion_docs.extend(i_docs)


markdown_splitter = RecursiveCharacterTextSplitter(
    separators=["#", "##", "###", "\\n\\n", "\\n", "."],
    chunk_size=1500,
    chunk_overlap=150)
split_docs = markdown_splitter.split_documents(notion_docs)
llama_docs = [Document.from_langchain_format(doc) for doc in split_docs]

llm = OpenAILike(
    is_chat_model=False,
    api_base=API_BASE,
    api_key="any",
    max_tokens=256,
    temperature=0.2
)

embed_model = OpenAIEmbedding(api_base=API_BASE, api_key="any")

index = VectorStoreIndex.from_documents(llama_docs, embed_model=embed_model, show_progress=True)

query_engine = index.as_query_engine(llm=llm, text_qa_template=qa_template)
result = query_engine.query("What should I do in case of a security incident?")
print(result)
