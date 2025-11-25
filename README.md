# InsightQA
RAG-Powered Test Case and Script Generation Agent

Live App:  
https://insightapp-sreeyasrikanth.streamlit.app/

InsightQA is a QA automation assistant that converts product documents into structured test cases and executable Selenium-style scripts.  
It uses Retrieval-Augmented Generation and a Llama3 LLM backend to generate grounded, document-based outputs.

---

## Features

### 1. Upload Knowledge Sources
Users can upload:
- A primary file (HTML UI, API spec, flow, etc.)
- Multiple supporting documents (PDF, TXT, MD, JSON, etc.)

Each upload creates a separate Knowledge Base (KB).

---

### 2. Vector-Based Knowledge Base
The system:
- Extracts and parses text  
- Chunks and embeds documents using Sentence Transformers  
- Stores vectors in ChromaDB  
- Maintains metadata for all uploaded files  

Each KB can be viewed, renamed, or deleted.

---

### 3. Test Case Generation (RAG)
Given a natural-language request such as:

“discount code validation”

InsightQA retrieves the most relevant context and generates structured JSON test cases including:
- Test ID  
- Test Scenario  
- Preconditions  
- Steps  
- Expected Result  
- Context references  

All test cases are grounded in the uploaded documents.

---

### 4. Script Generation
InsightQA can convert a selected test case into a runnable Python script.  
Locators and actions are inferred from the uploaded HTML file.

---

## Workflow Summary

1. Upload main and support files  
2. Build a Knowledge Base  
3. Enter a feature to test  
4. Retrieve relevant context  
5. Generate structured test cases  
6. Select a test case  
7. Generate the corresponding script  

---

## Technology Stack

- Streamlit  
- FastAPI  
- ChromaDB
- SQLite
- Sentence-Transformers  
- Llama3  

---

## License
MIT License


