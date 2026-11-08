from dotenv import load_dotenv
import os

load_dotenv()


def get_llm_client():
    from openai import OpenAI

    client = OpenAI(
        base_url=os.getenv("API_URL"),
        api_key=os.getenv("API_KEY"),
    )

    return client
