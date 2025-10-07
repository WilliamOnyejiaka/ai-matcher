from dotenv import load_dotenv
import os

#* Load .env file
load_dotenv()

#* Access environment variables
PORT = os.getenv("PORT")
API_KEY = os.getenv("API_KEY")
REDIS_URL = os.getenv("REDIS_URL")
DATABASE_URL = os.getenv("DATABASE_URL")
RABBITMQ_URL = os.getenv("RABBITMQ_URL")
BATCH_SIZE = os.getenv("BATCH_SIZE") 
PINECONE_KEY = os.getenv("PINECONE_KEY")
