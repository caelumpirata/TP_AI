FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

RUN mkdir -p /app/uploads

# docker buildx build --platform linux/amd64 -t dinesh214/chatbot:feb09 --push .

# Expose the port the app runs on
EXPOSE 8501

RUN apt-get update && apt-get install -y git
RUN apt install -y build-essential

RUN pip install --user --no-cache-dir --upgrade pip

RUN pip install tabulate
RUN pip install langchain-community
RUN pip install pandasai==1.4.10
RUN pip install numexpr
RUN pip install --no-cache-dir mitosheet
RUN pip install streamlit_extras streamlit_modal
RUN pip install langchain_experimental streamlit matplotlib pandas streamlit_chat pygwalker langchain openai psycopg2 langchain_core langchain_groq
RUN pip install dask dask[dataframe]


COPY TP.py .
COPY style.css .
COPY numbers.txt .


CMD ["streamlit", "run", "TP.py"]