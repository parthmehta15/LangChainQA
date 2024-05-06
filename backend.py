

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch




def text_preprocessing(path, chunk_size, chunk_overlap):
        loader = DirectoryLoader(path, glob = "**/*pdf")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = text_splitter.split_documents(documents)
        return documents



def run_app(embeddings, index_name, path = 'data/', chunk_size = 500, chunk_overlap = 40, mode = 'LOAD', relevent_chunks = 4):
#     if mode=='INIT':
#             documents = text_preprocessing(path = path, chunk_size = chunk_size, chunk_overlap = chunk_overlap)
#             vectorstore = Pinecone.from_documents(documents, embeddings, index_name=index_name) 
#     else:
#             vectorstore = Pinecone.from_existing_index(index_name, embeddings)
# 
# 
        documents = text_preprocessing(path = path, chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        data_base = DocArrayInMemorySearch.from_documents(
                documents, 
                embeddings
                )



        retriever = data_base.as_retriever()
        # retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":relevent_chunks}) # Number of relevent similar chunks
        print(retriever)
        qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), retriever)
        #     print(qa)
        return qa



