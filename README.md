# Lecture Bot

A personalized chatbot for lecture slides with RAG (Retrieval-Augmented Generation) capabilities and knowledge graph visualization.

## Features

- Upload and process lecture slides in PDF format
- Extract text and mathematical equations from slides
- Create and visualize knowledge graphs of lecture content
- Chat interface with mathematical equation rendering
- Quiz generation based on lecture content
- Personalized responses that reference lecture material

## Requirements

- Python 3.9+
- Google Gemini API key

## Setup

1. Clone the repository

   ```
   git clone https://github.com/yourusername/lecture_bot.git
   cd lecture_bot
   ```

2. Install dependencies

   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your Gemini API key:

   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

4. Run the application
   ```
   streamlit run app.py
   ```

## Usage

1. **Upload Page**: Upload your lecture PDFs. The system will process them and add the content to the knowledge base.

2. **Chat Page**: Ask questions about your lectures. The bot will provide responses based on the content of your slides, with proper formatting for mathematical equations.

3. **Knowledge Graph**: Visualize the concepts and relationships extracted from your lecture materials.

## Project Structure

The project is organized as follows:

- `app.py`: Main Streamlit application
- `data/`: Directory storing uploaded PDFs and processed data
- `src/`: Core functionality of the application
- `ui/`: Streamlit UI components

## Adding New Lectures

Upload new lecture slides through the upload interface. The system will automatically process the content and update the knowledge base.

## License

MIT
