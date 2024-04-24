"""
Queries:
1. Explain Transformers as in the article
2. My name is Ivan
3. What did I say?
4. What is the capital of France?
"""

import os
import openai

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from qdrant_client import QdrantClient

from custom_chat_history import CustomChatBufferHistory
from flask import Flask, render_template, request, jsonify, make_response
from dotenv import load_dotenv, find_dotenv



# .env file is read locally with OPENAI_API_KEY inside
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ.get('OPENAI_API_KEY')

try:
    qhost=os.environ['QDRANT_HOST']
    qport=int(os.environ['QDRANT_PORT'])
except KeyError:
    print (f"QDRANT_HOST,QDRANT_PORT dont  exist")
    qhost = ""
    qport = ""


app = Flask(__name__)



class LlmPdfRagGenerator:
    def __init__(self, llm, custom_chat_history, text_splitter):
        self.llm = llm
        self.custom_chat_history = custom_chat_history
        self.vectordb = None
        self.embeddings_model = None
        self.text_splitter = text_splitter

    def initialize(self):
        # More complicated initialization is happening here, divided to make code more modular
        self.initialize_embedding_model()
        self.initialize_vector_db_and_fill_it()


    def initialize_embedding_model(self):
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L12-v2',
            model_kwargs={"device": "cpu"},  # cuda
            # Set True to compute cosine similarity
            encode_kwargs={"normalize_embeddings": True})


    def initialize_vector_db_and_fill_it(self):
        print('Getting document chunks...')        
        docs = self.get_chunks(embeddings_model=self.embeddings_model, splitter=self.text_splitter)


        if qhost != "" and qport!="":
            self.vectordb = Qdrant.from_documents(
                    docs, self.embeddings_model,
                    host=qhost,
                    port=qport,
                    collection_name="my_documents",
                )
            print("by host port")
        else:
            self.vectordb = Qdrant.from_documents(
                    docs, self.embeddings_model,
                    location=":memory:",  # Local mode with in-memory storage only
                    collection_name="my_documents",
                )
            print("by in memory")
    
    @staticmethod
    def get_chunks(embeddings_model, splitter):
        pdf_file_url = 'https://arxiv.org/pdf/1706.03762.pdf'
        loader = PyPDFLoader(pdf_file_url)

        print('Text Splitter Name:', splitter)
        if splitter == 'character_splitter':
            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=50,
                                                  length_function=len)
        else:
            raise AssertionError(f'There is not such `splitter` {splitter}')
        return loader.load_and_split(text_splitter)




    def search_results(self, _input, k=3):
        # Similarity search against embedding
        search_results = self.vectordb.similarity_search(_input, k=k)

        # Combine content
        content_combined = ""
        for result in search_results:
            content_combined += (result.page_content + " \n")
        return {"content_combined": content_combined}



class OpenAIRagLLM(LlmPdfRagGenerator):
    def __init__(self, text_splitter):
        self.llm = self.connect_to_model('gpt-3.5-turbo')
        self.custom_chat_history = CustomChatBufferHistory(human_prefix_start='Human:', ai_prefix_start='AI:')
        self.prompt_template = """
        You are a helpful, respectful and honest assistant. You help to answer questions regarding `Attention is all you need` article.
        Answer the following Question based on the Context and Chat_history only. Give preferernce to Chat_history. If you don't know the answer, say 'I don't know'.
 
        Chat_history: {chat_history}

        Context: {content}

        Question: {question}
        """

        super().__init__(
            llm=self.llm,
            custom_chat_history=self.custom_chat_history,
            text_splitter=text_splitter
        )

    @staticmethod
    def connect_to_model(model_name):
        print("Load OpenAI LangChain LLM Object...")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(temperature=0, model=model_name, callbacks=[StreamingStdOutCallbackHandler()])

    def generate(self, _input):
        content = self.search_results(_input)
        history= self.custom_chat_history.memory
        print (history)
        csv_string =""
        for row in history:
            csv_string += str(row) + '\n'

        prompt = self.prompt_template.format(content=content["content_combined"],
                                            # chat_history=list(self.custom_chat_history.memory),
                                            chat_history=csv_string,
                                            question=_input)
        
        print('Final Prompt:', prompt, end='\n\n\n')
        response = self.llm.invoke(prompt)
        self.custom_chat_history.update_history(human_query=_input, ai_response=response)
        return response.content



@app.route("/openai")
def home_openai():
    openai_llm_rag.custom_chat_history.clean_memory()
    return render_template("index_openai.html")


@app.route("/generate", methods=['POST'])
def get_bot_response():
    data = request.json
    user_text = data.get('msg', '')
    index = data.get('index', '')


    if index == 'openai':
        try:
            response = openai_llm_rag.generate(user_text)
        except Exception as e:
            print(e)
            response = "Sorry! We have an outage at the moment, please be assured we will be back very soon."
    else:
        error_response = jsonify({"error": "Invalid html index"})
        return make_response(error_response, 400)

    return jsonify({"response": response})


if __name__ == '__main__':
    """
    1. Connecting to APIs & 2. Initialize LLM RAG Generator object
    """

    # passed in dockerfile
    text_splitter_name = os.environ.get('TEXT_SPLITTER', 'character_splitter')

    print('Initialize OpenAI GenAI Project')
    openai_llm_rag = OpenAIRagLLM(text_splitter=text_splitter_name)
    openai_llm_rag.initialize()

    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
