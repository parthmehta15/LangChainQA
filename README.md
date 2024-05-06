# LangChainQA


Add pdfs to data folder and the talk to the pdf
Only need to run frontend.py


######  ENV file #############
# Once you add your API key below, make sure to not share it with anyone! The API key should remain private.
OPENAI_API_KEY=

PINECONE_ENV=

PINECONE_API_KEY=

PINECONE_INDEX_NAME= 

#Make sure to create an Index first on Pinecone.com (Embedding_size = 1536 for openAI, similarity cosine)

PINECONE_MODE=LOAD     #INIT (When running for first time) OR LOAD (If vector database already exists)


##########################
Only OPENAI API needed others can be ignored
