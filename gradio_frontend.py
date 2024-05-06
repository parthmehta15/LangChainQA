import os
import pinecone
from dotenv import load_dotenv,find_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings

from backend import run_app

import gradio as gr


load_dotenv(find_dotenv())

##PINE CONE
# pinecone.init(
#     api_key=os.getenv('PINECONE_API_KEY'),  
#     environment=os.getenv('PINECONE_ENV')  
# )
# index_name = os.getenv('PINECONE_INDEX_NAME')
# mode = os.getenv('PINECONE_MODE')
index_name = None
embeddings = OpenAIEmbeddings()

qa = run_app(embeddings=embeddings, index_name =index_name, path = 'data/', chunk_size = 500, chunk_overlap = 40, mode = 'LOAD', relevent_chunks = 4)


#Frontend
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    
    def respond(user_message, chat_history):
        print(user_message)
        if chat_history:
          chat_history = [tuple(sublist) for sublist in chat_history]


        response = qa({"question": user_message, "chat_history": chat_history})
        chat_history.append((user_message, response["answer"]))
        print(chat_history)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(debug=True, share=True)