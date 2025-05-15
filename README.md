Agentic RAG & Tool App
Overview
The Agentic RAG & Tool App is an intelligent, multi-agent system designed to handle document processing and real-time information retrieval. It combines Retrieval-Augmented Generation (RAG) for querying uploaded documents with live search capabilities (e.g., weather reports, cryptocurrency prices). Built with Python, Chainlit, Qdrant, and the Gemini API, this app provides a user-friendly chat interface for interacting with documents and external data sources.

Key Features

Document Processing
Upload PDF, DOCX, or image files to extract text and generate embeddings stored in Qdrant for semantic search.
RAG Workflow
Query documents via a RAG pipeline powered by Gemini embeddings and Qdrant vector search.
Live Search
Retrieve real-time data (e.g., weather, Bitcoin prices) using the Tavily API.
Multi-Agent System

Orchestration Agent: Manages user context and delegates tasks.
RAG Assistant: Handles document-related queries.
Live Search Assistant: Processes real-time search requests.


Interactive UI
Built with Chainlit for a seamless chat-based experience.
User Context Management
Store and retrieve user information (name, age, location, interests, preferences).


Prerequisites

Python ≥ 3.11
Docker (optional, for local Qdrant deployment)

API Keys

Gemini API (for embeddings and LLM)
Tavily API (for live search; up to 1000 free searches/month)
Qdrant API (if using a cloud instance)


Installation

Clone the Repository
bashgit clone https://github.com/your-username/agentic-rag-tool-app.git
cd agentic-rag-tool-app

Set Up a Virtual Environment
bashpython -m venv .venv
source .venv/bin/activate     # On Windows: .venv\Scripts\activate

Install Dependencies
bashpip install -r requirements.txt
Ensure your requirements.txt includes:
chainlit==1.2.0
qdrant-client==1.12.0
google-generativeai==0.8.3
langchain-text-splitters==0.3.0
python-magic==0.4.27
docx2txt==0.8
pytesseract==0.3.13
textract==1.6.5
PyPDF2==3.0.1
requests==2.32.3
python-dotenv==1.0.1

Configure Environment Variables
Create a .env in the project root:
GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=https://your-qdrant-instance:6333
QDRANT_API_KEY=your_qdrant_api_key
TAVILY_API_KEY=your_tavily_api_key
COLLECTION_NAME=gemini-embeddings
EMBED_MODEL=models/embedding-001
CHUNK_SIZE=800
CHUNK_OVERLAP=160

(Optional) Run Qdrant Locally
bashdocker run -d -p 6333:6333 qdrant/qdrant

Running the Application
bashchainlit run ui.py
Then open your browser to:
http://localhost:8000


Usage
Upload Documents
Use the chat UI to upload PDF, DOCX, or image files.
Files are saved to C:\temp_uploads and processed into embeddings stored in Qdrant.
Query Documents
Ask things like:
"What does my document say about AI?"
The RAG Assistant will retrieve relevant content.
Live Search
Ask real-time questions, e.g.:
"What's the latest weather in Sialkot?"
"Current Bitcoin price?"
The Orchestration Agent delegates to the Live Search Assistant.
Manage User Context
Set info:
"My name is Alice."
"I'm interested in crypto."
Get info:
"What's my name?"
Project Structure
agentic-rag-tool-app/
├── main.py           # Defines agents & system setup
├── tools.py          # RAG & live search tools
├── ui.py             # Chainlit UI implementation
├── .env              # Environment variables
├── requirements.txt  # Dependencies
├── README.md         # This file
└── C:\temp_uploads\  # Directory for uploaded files
Troubleshooting
Live Search Not Working
Verify Tavily API Key
bashcurl -X POST https://api.tavily.com/search \
  -H "Content-Type: application/json" \
  -d '{"api_key":"your_key","query":"weather Sialkot"}'
Check Network

Ensure no firewall/proxy blocks api.tavily.com.
Test connectivity: ping api.tavily.com.

Inspect Logs
bashpython -m litellm --debug
chainlit run ui.py
Update search_everything
Ensure it gracefully handles errors (see tools.py).
Other Issues

RAG Failures: Make sure Qdrant is running and the collection exists.
Gemini API Errors: Confirm your key and usage limits.
File Uploads: Ensure C:\temp_uploads is writable.

Contributing
We welcome contributions!

Fork the repo.
Create a branch:
bashgit checkout -b feature/your-feature

Commit and push:
bashgit commit -m "Add my feature"
git push origin feature/your-feature

Open a pull request.

For detailed guidelines, consider adding a CONTRIBUTING.md (e.g., via makeareadme.com).
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgements

Chainlit for the UI framework
Qdrant for vector storage
Google Gemini for embeddings and LLM
Tavily for live search capabilities

Contact
For issues or feature requests, open a GitHub issue or email your-email@example.com.
Happy coding! Let's make AI accessible and powerful. 🚀
