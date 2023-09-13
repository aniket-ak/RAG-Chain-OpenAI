# Import packages
import os
from dotenv import load_dotenv
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import FAISS
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.embeddings.cohere import CohereEmbeddings
import cohere as co
import speech_recognition as sr
import pyttsx3
import numpy as np
import gradio as gr

# set working directory
WORKING_DIR = os.getcwd() #"C:\\Users\\aniket.kulkarni\\OneDrive - Aligned Automation\\03_Initiatives\\03_NLP\\PAN_RAG\\"

def create_chunks(sample_text):
    """Split the input text into multiple chunks based on 
    chunk_size and overlap and return them as documents"""
    markdown_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = markdown_splitter.create_documents([sample_text])
    return docs

def initialize_db(docs):
    """Convert the documents into embeddings and then store them into FAISS vector store"""
    embeddings = CohereEmbeddings(model = "multilingual-22-12")
    db = FAISS.from_documents(docs, embeddings)
    return db

def make_prompt():
    """This function returns prompt to be used by the model in a specified format and template"""
    template = """You are an exceptional customer support chatbot that gently answer questions.

    You know the following context information.

    {chunks_formatted}

    Answer to the following question from a customer. Use only information from the previous context information. Do not invent stuff. Bulletize your answers for easy readability when the context contains bulletized list. Answer should be in {language}.

    Question: {query}

    Answer:"""

    prompt = PromptTemplate(
        input_variables=["chunks_formatted", "query","language"],
        template=template,
    )
    return prompt

def get_chunks(query, db):
    """Use similarity score and get the n (3) closest documents to the query input by user"""
    docs_answers = db.similarity_search(query,3)
    retrieved_chunks = [doc.page_content for doc in docs_answers]
    chunks_formatted = ''.join(retrieved_chunks)
    
    return chunks_formatted

def speech_to_text():
    """Records the speech from user's device and converts that into text"""
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("Calibrating...")
        r.adjust_for_ambient_noise(source, duration=5)
        # optional parameters to adjust microphone sensitivity
        r.energy_threshold = 400
        # r.pause_threshold=0.5
        
        text = ""
        print("listening now...")
        
        audio = r.listen(source, timeout=5, phrase_time_limit=30)
        print("Recognizing...")
        text = r.recognize_sphinx(audio)
        print(text)
    return text

def text_to_speech(response_text):
    """Given the response from the LLM in form of text, read out loud"""
    engine = pyttsx3.init()
    engine.say(response_text)
    engine.runAndWait()

def calculate_relevance_score(db, result):
    """Calculate the similarity score between result and the documents from knowledge 
    source. These scores are averaged to get a sense of closeness between the results and 
    the original source."""
    similar_docs = db.similarity_search_with_score(result,3)
    scores = np.array([similar_docs[i][1] for i in range(len(similar_docs))])
    average_score = scores.mean()
    return average_score


def main(option, query):
    """The main function"""
    
    # if the option is audio from UI, obtain the query by converting the speech to text
    if option == 'Audio':
        query = speech_to_text()
        print(query)

    # Read the .env file containing the API keys
    env_file = os.path.join(WORKING_DIR,".env")
    if os.path.exists(env_file):
        print("Successfully loaded the environment file", env_file)
        load_dotenv(env_file)
    else:
        raise FileNotFoundError("No .env file found (Refer to the .env.template file to place your OpenAI API key)")

    # Read in the knowledge source
    with open(os.path.join(WORKING_DIR, "KnowledgeDocument(pan_card_services).txt"), encoding='utf-8') as f:
        sample_text = f.read()

    # Detect language using Cohere. This is used to generate the response in the same language
    language=co.Client().detect_language([query]).results[0].language_name

    # Split the knowledge source to create chunks
    docs = create_chunks(sample_text)

    # Store the documents in vector store
    db = initialize_db(docs)

    # Find the n (=3) documents closest to the query
    chunks_formatted = get_chunks(query, db)

    # Define the LLM
    llm=OpenAI(model='text-davinci-003', temperature=0)

    # LLMChain for chaining LLM and prompt
    orig_chain = LLMChain(llm=llm, prompt=make_prompt())

    # Guardrails for making sure that the answers are ethical and fair
    ethical_principle = ConstitutionalPrinciple(
        name="Ethical Principle",
        critique_request="The model should only talk about ethical and fair things.",
        revision_request="Rewrite the model's output to be both ethical and fair.",
    )

    # Constitutional chain for guardrails
    constitutional_chain = ConstitutionalChain.from_llm(
        chain=orig_chain,
        constitutional_principles=[ethical_principle],
        llm=llm
    )

    # Obtain final response by running through both chains
    result = constitutional_chain.run({"chunks_formatted":chunks_formatted, "query":query, "language": language})
    
    # Replace ABC with SBNRI since in the ideal responses, ABC is never seen. This step
    # can also be done for the knowledge source; more of a design choice
    result = result.replace('ABC', 'SBNRI')
    
    # Calculate the relevance score
    score = calculate_relevance_score(db, result)

    return result, 100-score


if __name__ == "__main__":
    # Use gradio to make the UI
    gr.Interface(
        fn=main,
        inputs=[
            gr.Radio(["Audio", "Text"], label="Input method",value="Text"), 
            gr.Textbox(lines=2, placeholder="Please ask your question...",label='Question (Supports multiple language inputs)')], 
        outputs=[
            gr.Textbox(lines=5, placeholder="Answer...", label="Answer"), 
            gr.Number(label="Evaluation score (0 indicates the answer is very different and 100 indicates very accurate answer)")], 
        allow_flagging="never"
    ).launch()