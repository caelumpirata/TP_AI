FROM python:3.11-buster

# Set the working directory to /app
WORKDIR /app

RUN mkdir -p /app/uploads

# docker buildx build --platform linux/amd64 -t dinesh214/chatbot:feb09 --push .

# Expose the port the app runs on
EXPOSE 8501

RUN apt-get update && apt-get install -y git
RUN apt install -y build-essential

RUN pip install --user --no-cache-dir --upgrade pip
RUN pip install streamlit-authenticator
RUN pip install tabulate
RUN pip install langchain-community
RUN pip install numexpr
RUN pip install --no-cache-dir mitosheet
RUN pip install streamlit_extras streamlit_modal
RUN pip install langchain_experimental streamlit matplotlib pandas streamlit_chat pygwalker langchain openai psycopg2 langchain_core langchain_groq


COPY Login.py .
COPY config.yaml .
COPY numbers.txt .


CMD ["streamlit", "run", "Login.py"]