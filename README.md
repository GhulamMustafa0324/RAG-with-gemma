# RAG-with-Gemma: Streamlit Web App for Retrieval-Augmented Generation

This repository implements a Retrieval-Augmented Generation (RAG) model with a user-friendly Streamlit web app.

## Key Features

- **Two Chat Interfaces:**
  - Interact with a pre-loaded dataset for information retrieval and generation using RAG.
  - Upload a PDF document and chat with the AI using the uploaded content for context.

## Content Breakdown

- **app.py:** Main script to run the RAG model and web app. Note: Add your Hugging Face token for GEMMA 2B access on line 17.
- **colab.ipynb:** Jupyter notebook for running the RAG model in Google Colab (without a GPU) using a tunnel for Streamlit.
- **ingestion.py:** Script for data ingestion, preprocessing, and vector creation.
- **requirements.txt:** File listing dependencies needed to run the project.
- **.txt files:** These files contain links used for data ingestion.

## Running the Application

### Prerequisites

- Python with necessary libraries (specified in `requirements.txt`).

### With GPU

1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`

### Without GPU (Google Colab)

1. Upload `app.py` to your Google Drive.
2. Open `colab.ipynb` in Colab and run all cells. This uses a provided tunnel to run Streamlit.

## Data Ingestion

To preprocess and add data to the system, run: `python ingestion.py` (This process may take some time).

### Adding Your Own Data

Modify existing `.txt` files or create new ones with links to your data for ingestion using `ingestion.py`.

## Note

- Replace `YOUR_HUGGINGFACE_TOKEN` in `app.py` with your actual Hugging Face token to use the GEMMA 2B model.

This RAG-with-Gemma application offers a convenient way to interact with information retrieval and generation tasks through a web interface.
