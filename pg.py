from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings



# os.environ["HF_TOKEN"] = "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"

loader = TextLoader("rag.txt", encoding='utf-8')
doc = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
texts = text_splitter.split_documents(doc)


model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


vector = embeddings.embed_query('Testing embeddings')
doc_vector = embeddings.embed_documents([t.page_content for t in texts[:5]])


conn = "postgresql+psycopg2://master:0r5VB[TL?>A/8,}<vkpmEwS)@65.20.77.132:32432/ems_ai"
collection_name = 'state_of_union_vectors'

# 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
db = PGVector.from_documents(embedding=embeddings, documents=texts, connection_string=conn, collection_name=collection_name)

