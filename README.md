# Ollama Document Chat App with Training

This application allows you to chat with your documents using Ollama's local LLM models. Upload documents, train the model on their content, and then ask questions to receive AI-generated responses - all running locally on your machine.

## Features

- **Document Processing**: Upload and process documents in various formats (PDF, DOCX, TXT, CSV, Excel)
- **Ollama Integration**: Uses locally installed Ollama models with no need for external API keys
- **Model Fine-tuning**: Train the model on your uploaded documents to improve response quality
- **Multiple Model Support**: Switch between any models you have installed in Ollama
- **Customizable Generation**: Adjust temperature, top-p, max tokens, and repetition penalty
- **Chat Interface**: Simple and intuitive chat interface for interacting with your documents
- **Conversation History**: Keeps track of your conversation for context-aware responses

## Prerequisites

- **Python 3.9+**
- **Ollama**: Install from [ollama.ai](https://ollama.ai)
- **Llama 3.2 model**: Run `ollama pull llama3.2:latest` (or another model of your choice)

## Setup Instructions

1. **Clone or download this project**

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

4. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Create necessary directories**:
   ```bash
   mkdir -p templates data/uploads processed_data
   ```

6. **Place the HTML template in the templates directory**:
   Copy the index.html file to the templates directory.

7. **Ensure Ollama is running**:
   **Refer to Link and install Ollam model : https://ollama.com/library/llama3.3:latest**
   Open a terminal and run:
   ```bash
    brew install ollama
    ollama run llama3.2
   ollama --version
   #ollama run llama3.2
   ollama serve
   ```
   (This needs to be running in the background while using the app)

## Running the Application

1. **Start the application**:
   ```bash

   python main.py
   ```

2. **Access the web interface**:
   Open your browser and go to:
   ```
   http://localhost:9000
   ```

## Usage Guide

1. **Upload Documents**:
   - Click "Choose Files" to select documents
   - Click "Upload Files" to process them
   - You should see your files appear in the "Uploaded Files" section

2. **Initialize the Model**:
   - Choose your model from the dropdown (default is llama3.2:latest)
   - Click "Initialize Model"
   - Wait for initialization to complete

3. **Train the Model on Your Documents** (New Feature):
   - Set the number of training examples (recommended 10-20)
   - Click "Train on Documents"
   - Wait for the training to complete (progress bar will show status)
   - A new fine-tuned model will be created with the suffix "-docs" (e.g., "llama3.2-docs")

4. **Ask Questions**:
   - Type your question in the chat input
   - The model will generate a response based on your documents
   - If you've trained the model, responses should be more relevant to your documents

5. **Adjust Settings** (optional):
   - Modify temperature, top-p, tokens, and repetition penalty
   - Click "Update Settings" to apply changes

## Training Process Details

The training process creates a fine-tuned model from your uploaded documents by:

1. Chunking your documents into manageable pieces
2. Creating 80/20 train/validation split
3. Automatically generating questions and answers based on document content
4. Creating a custom Ollama model with these examples
5. Using the trained model for future responses

The trained model will show up with a "(fine-tuned)" label in the model dropdown and will be automatically used when initialized.

## Supported File Types

- PDF (.pdf)
- Word Documents (.docx)
- Text Files (.txt)
- CSV Files (.csv)
- Excel Files (.xlsx, .xls)

## Troubleshooting

- **Model initialization fails**: Make sure Ollama is running in another terminal with `ollama serve`
- **Model not found**: Ensure you've pulled the model with `ollama pull model_name`
- **Training takes too long**: Reduce the number of training examples or use a smaller model
- **Poor document parsing**: Check if your document is properly formatted and not scanned/image-based
- **Training error**: Make sure your documents contain sufficient text content