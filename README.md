# 🧠 Agentic RAG: Multi-Agent PDF + Web Question Answering System

An intelligent multi-agent RAG (Retrieval-Augmented Generation) system powered by **CrewAI**, capable of answering user questions using both a **PDF document** and the **web** as information sources. The system routes, retrieves, verifies, and formats answers — returning a clean, Markdown-formatted response.

---

## 🚀 Features

- 🧠 Agent-based architecture using CrewAI
- 📄 Loads and searches PDF for relevant information
- 🌐 Falls back to web search if the PDF lacks the answer
- 🔍 Verifies accuracy before returning results
- 📝 Outputs final answers in Markdown format
- 🧩 Modular design: easily extend or customize agents/tools
## 📌 Use Case

Ask any question and provide a PDF. The system attempts to find the answer in the PDF. If not found, it automatically searches the web, verifies the response, and returns a polished Markdown-formatted answer.

## 💡 How It Works
1-Router Agent >>decides whether to use PDF or Web based on the question.

2-Retriever Agent >>fetches relevant content from the selected source.

3-Verifier Agent >>uses an LLM to check the factuality of the retrieved answer.

4-Formatter Agent >> outputs the final answer in Markdown.

## 📎 Example Output

![Streamlit-06-20-2025_05_49_PM](https://github.com/user-attachments/assets/d4f679cb-9f1e-4280-a3ee-13771957a803)

## 🛠️ Tech Stack
| Tool               | Purpose                                                   |
| ------------------ | --------------------------------------------------------- |
| **Python**         | Core programming language                                 |
| **Crew AI**        | Multi-agent orchestration and task flow                   |
| **Gemini**         | Large Language Model (LLM) for content generation         |
| **Tavily API**     | Real-time web search                                      |
| **Deepseek**       | Assistant Large Language Model (LLM) for grading answer   |
| **PyPDF2**         | Reading and Estract Text from PDF files                   |
| **streamlitui**    | generating ui                                             |


## Project Flowchart
![Editor _ Mermaid Chart-2025-06-20-141246](https://github.com/user-attachments/assets/36da6c4a-34c2-4d93-adeb-5af2aa9d46c2)

## 🙋‍♀️ Author
Mena Allah Ahmed — Data Scientist & AI Developer

Feel free to connect or ask questions!
