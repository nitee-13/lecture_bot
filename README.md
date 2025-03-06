# 📚 Lecture Bot

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B.svg)](https://streamlit.io/)
[![Gemini](https://img.shields.io/badge/Gemini-API-4285F4.svg)](https://ai.google.dev/)

A powerful AI-powered lecture assistant that helps you understand and learn from your lecture materials using advanced RAG (Retrieval-Augmented Generation) capabilities and interactive knowledge graphs.

[Getting Started](#getting-started) •
[Features](#features) •
[Contributing](#contributing) •
[Project Status](#project-status)

</div>

## 🌟 Features

### 📄 Document Processing
- **Smart PDF Processing**: Upload and process lecture slides in PDF format
- **Advanced Text Extraction**: Extract text, mathematical equations, and diagrams from slides
- **Intelligent Chunking**: Smart text segmentation for better context understanding

### 🧠 AI-Powered Learning
- **Interactive Chat**: Ask questions about your lectures with mathematical equation rendering
- **Context-Aware Responses**: Get answers based on your lecture content
- **Quiz Generation**: Test your knowledge with AI-generated quizzes
- **Knowledge Graph Visualization**: Explore concepts and their relationships

### 🎯 Learning Tools
- **Concept Maps**: Visualize relationships between different topics
- **Equation Support**: Beautiful rendering of mathematical equations
- **Prerequisite Tracking**: Understand dependencies between concepts
- **Interactive Learning**: Engage with your lecture content in multiple ways

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- Google Gemini API key
- CUDA-capable GPU (recommended for faster processing)

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/lecture_bot.git
   cd lecture_bot
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables
   Create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your_api_key_here
   VECTOR_DB_PATH=data/embeddings
   KNOWLEDGE_GRAPH_PATH=data/knowledge_graph
   PDF_STORAGE_PATH=data/pdfs
   ```

4. Run the application
   ```bash
   streamlit run app.py
   ```

## 🎮 Usage

### 📤 Upload Page
- Upload your lecture PDFs
- View processing status and details
- Access document statistics and chunks

### 💬 Chat Interface
- Ask questions about your lectures
- Get context-aware responses
- View mathematical equations in beautiful format

### 📊 Knowledge Graph
- Visualize concept relationships
- Explore topic hierarchies
- Navigate through lecture content

### 📝 Quiz Generation
- Generate quizzes from lecture content
- Test your understanding
- Get immediate feedback

## 🏗️ Project Structure

```
lecture_bot/
├── app.py                 # Main Streamlit application
├── data/                  # Data storage
│   ├── pdfs/             # Uploaded PDFs
│   ├── embeddings/       # Vector embeddings
│   └── knowledge_graph/  # Knowledge graphs
├── src/                  # Core functionality
│   ├── embedding/        # Embedding generation
│   ├── knowledge_graph/  # Graph building
│   ├── llm/             # Language models
│   ├── pdf_processor/   # PDF processing
│   └── vector_store/    # Vector storage
└── ui/                  # UI components
    ├── components/      # Reusable UI elements
    └── pages/          # Streamlit pages
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### 🎯 Areas for Contribution
- **New Features**: Add support for more document types or learning tools
- **UI Improvements**: Enhance the user interface and experience
- **Performance**: Optimize processing and response times
- **Documentation**: Improve documentation and add examples
- **Testing**: Add unit tests and integration tests
- **Bug Fixes**: Fix issues and improve error handling

## 📈 Project Status

### 🟢 Current Status
- Core functionality is working
- Basic RAG implementation is complete
- Knowledge graph visualization is implemented
- Quiz generation is functional

### 🟡 In Progress
- Enhanced error handling
- Performance optimizations
- Additional document format support
- Improved UI/UX

### 🔴 Planned Features
- Multi-language support
- Collaborative learning features
- Advanced analytics
- Mobile app version
- API endpoints for external integration

## 📝 Issues and Pull Requests

### 🐛 Known Issues
- Large PDF processing can be slow
- Some complex equations might not render correctly
- Memory usage can be high for large documents

### 🔄 Pull Requests
We welcome pull requests that:
- Fix bugs
- Add new features
- Improve documentation
- Enhance performance
- Add tests

## 🙏 Acknowledgments

- Google Gemini API for powerful language models
- Streamlit for the beautiful UI framework
- FAISS for efficient similarity search
- PyMuPDF for PDF processing
- NetworkX for graph operations

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/yourusername/lecture_bot/issues) page
2. Create a new issue with detailed information
3. Join our community discussions

---

<div align="center">
Made with ❤️ for better learning experiences
</div>
