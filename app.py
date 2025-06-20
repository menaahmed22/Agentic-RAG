
"""##importing libraries"""

import json
from google.colab import userdata
import google.generativeai as genai
from crewai import Agent,Task,Crew,Process,LLM
from tavily import TavilyClient
from crewai.tools import tool
from IPython.display import Markdown
import PyPDF2
from openai import OpenAI
from dotenv import load_dotenv
import os
from config import Gemini_api ,Routerai_key,tavily_api
# configure Based LLM
Based_llm = LLM(
    model="gemini/gemini-1.5-flash",
    temperature=0,
    provider="google_ai_studio",
    api_key=Gemini_api
    )

# configure Assest LLM
assest_llm = LLM(
    model="openrouter/deepseek/deepseek-r1-0528-qwen3-8b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=Routerai_key
)

# """## Tools"""

@tool("extract_pdf_text")
def extract_pdf_text(pdf_path: str) -> str:
    """
    Extracts text content from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page_num in range(len(reader.pages)):
                page_text = reader.pages[page_num].extract_text()
                if page_text:
                    text += page_text
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {e}"

search_client = TavilyClient(api_key=tavily_api) # using Tavily libraray to making web searching

@tool("web_search_tool")
def web_search_tool (question:str):
   """This tool is used for searching for any given ququestionery .used to find information through search engines"""
   return search_client.search(question,max_results=3)

# """##Agent 1,Task 1 ( Routing Question)"""

Router_Agent=Agent(
        role="Router Agent",
        goal="My objective is to accurately load, process, and prepare the content of user-provided PDFs for downstream retrieval and reasoning tasks."
             " I extract text, segment it into semantically meaningful chunks"
             "make it searchable using the PDFSearchTool."
             " My output empowers the rest of the system to ground answers in the uploaded document.",
        backstory=(
            "You are an assistant for question-answering tasks."
            "carefully ingest the PDF, extract all readable text, and organize it into manageable, information-rich segments that the system can later retrieve from."
            "extract and index the document's contents efficiently"
            "Analyze the given question {question} and pdf in {pdf_path}. and try to find answer in this pdf"
            "My work ensures that the system can first rely on the uploaded content before turning to external sources."
        ),
        verbose=True,
        allow_delegation=False,
        llm=Based_llm,
        tools=[extract_pdf_text]
)

Routing_task=Task(
    description=(
           "Analyze the given question {question} and extract information for the question {question} with the help of pdf_extractor_tool"),
     expected_output=(
            "Return just exactly one word:\n"
            "'answer_available' - if the question can be answered from our RAG knowledge base\n"
            "'answer_not_available' - if the question requires external information\n"
            "No additional explanation or preamble should be included."),
     agent=Router_Agent,
     tools=[extract_pdf_text],

)

# """##Agent 2,Task 2 (Reterivaling)"""

Reterival_agnet=Agent(
        role="Retriever Agent",
        goal="Use the suitable agent and tool to get answer",
        backstory=(
            "You are an assistant for question-answering tasks."
            "Use the suitable agent and tool to get answer"
            "You have to provide a clear concise answer."
        ),
        verbose=True,
        allow_delegation=False,
        llm=Based_llm,
        tools=[extract_pdf_text,web_search_tool]

)

Reterival_task=Task(
    description=(
            "Based on the response from the Router_Agent extract information for the question {question} with the help of the respective tool."
            "Use the extract_pdf_text to retrieve information from the vectorstore in case the Router_Agent final answer is 'answer_available'. "
            "Use the web_search_tool to retrieve information from the web  in case the Router_Agent final answer is 'answer_not_available'You should pass the input query {question} to the web_search_tool."
        ),
        expected_output=(
            "You should act excatly as the final answer of Router_Agent"
            "If the Router_Agent output is 'answer_available' then Analyze the given question {question} and pdf in {pdf_path}. and try to find answer in this pdf"
            "If the Router_Agent output  is 'answer_not_available' then use the web_search_tool to retrieve information from the web."
            "Return a claer and consise text as response."
            "\n"
            "\n mention the used tool"
        ),
        agent=Reterival_agnet,
        tools=[extract_pdf_text, web_search_tool],
)

# """##Agent 3,Task 3 (Grading)"""

Grader_agent=Agent(
        role="Grader Agent",
        goal=("My goal is to  verify the final answer before it reaches the user."
             "i have to make sure that the answer is relevant to the question."),
        backstory=("i have to make sure that the answer is relevant to the question."
                 " I ensure it is grounded in source evidence (PDF or web) and free of hallucinations or inconsistencies. "
                 "If the answer is inaccurate or unsupported, I flag it for correction or regeneration."),
       verbose=True,
       allow_delegation=False,
       llm=assest_llm,
)

Grader_task=Task(
    description=("Based on the response from the retriever task for the quetion {question}"
                 " evaluate whether the retrieved content is relevant to the question."),
    expected_output=(
            "Binary score  only 'yes' or 'no' score to indicate whether the answer is relevant to the question"
            "You must answer 'yes' if the response from the 'Reterival_task' is in alignment with the question asked."
            "You must answer 'no' if the response from the 'Reterival_task' is not in alignment with the question asked."
            "Do not provide any preamble or explanations except for 'yes' or 'no'."
        ),
        agent=Grader_agent,
        context=[Reterival_task],

)

# """####Agent 4,Task 4 (Formatter)"""

Formatter_agent = Agent(
    role="Answer Formatter",
    goal="Format the final answer into a clean, readable, structured format depending on context.",
    backstory=(
        "I am the Formatter Agent. Once an answer is verified, I take over and ensure it's presented clearly. "
        "I format answers as text, bullet points, tables, or JSON — depending on what's best for clarity and delivery."),
    allow_delegation=False,
    verbose=True,
    llm=Based_llm,
)

formatter_task = Task(
    description=("Based on the response from the Grader_agent give your final answer  "
                 "if Grader_agent  final answer is 'yes' Your job is to rephrase 'Reterival_agnet' final answer (quesetion's answer and used tool) for clarity and format it cleanly. "
                 "Use bullet points, tables, or summaries depending on the answer content. "
                 "Do not add or remove information, only format and improve clarity."
                 " elseif Grader_agent  final answer is 'no' Your final answer will be 'sorry i can not find answer' "),
    expected_output=(
        # "A JSON response with two keys: `format` (e.g., 'text', 'markdown', 'json') and `content` (formatted answer)."
        "the final answer organized as Markdowns and mandatory mention the used tool form Reterival_agnet "
        ),
    agent=Formatter_agent,
    context=[Reterival_task],
    llm=Based_llm,
 
)
import tempfile

crew_4=Crew(
  agents=[Router_Agent,Reterival_agnet,Grader_agent,Formatter_agent],
  tasks=[Routing_task,Reterival_task,Grader_task,formatter_task],
  verbose=False,
  process=Process.sequential)
output_dir = "./ai-agent-output"
os.makedirs(output_dir, exist_ok=True) 
# """## Creating the Crew"""
import streamlit as st
st.write('# Agentic RAG')
with st.form('Agentic RAG'):
   question_= st.text_input('what is your question')
   uploaded_file = st.file_uploader("upload your file")
   if uploaded_file is not None:
    # Save it temporarily to a file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name  # This is the actual file path
        # st.success(f"PDF uploaded successfully ")
        inputs={'question':f"{question_}",
           'pdf_path': f"{temp_path}" }
        with st.spinner("Generating Answer using Agentic RAG..."):
             results_4=crew_4.kickoff(inputs=inputs)
             st.write('# Answer is \n ',results_4.raw)
             st.success("Answer generated successfully!")
   st.form_submit_button('Get answer')




