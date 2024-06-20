# Simple RAG use case with Amphi ETL

# Import the Amphi pipeline

Just upload the Reddit_RAG_ChromaDB in your workspace and open it. You may adapt it to your input data.
Add an OpenAI API key and setup a perisisted directory for ChromaDB.

![image](https://github.com/tgourdel/simple-rag-amphi/assets/15718239/db9f8e23-ac9f-4a38-a9f9-f7c711ad2156)

# Start the Streamlit Chatbot

Go into the chatbot folder and install dependencies:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
