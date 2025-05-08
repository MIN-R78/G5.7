# PDF AI Assistant

A lightweight AI-powered assistant that answers questions about PDF documents using Azure OpenAI GPT-4.5. It supports both `PyPDF2` and `pdfplumber` for flexible text extraction, and uses FAISS for semantic search and retrieval. Users can interactively select the parser and target PDF at runtime, with support for summarization-style queries and multilingual input (English, Chinese, Korean).

## Features
- Dual PDF parser support (`PyPDF2` and `pdfplumber`)
- CLI-based dynamic PDF and parser selection
- Multilingual query handling (EN / 中文 / 한국어)
- Summarization mode for long-document questions
- FAISS-based vector similarity search
- Azure OpenAI GPT-4.5 integration

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/MIN-R78/G5.7.git
   cd G5.7
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
3. Run the app:

   ```bash
   python Google-4.py advanced
