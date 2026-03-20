import os
import openai

try:
    openapi_key = os.environ['OPENAI_API_KEY']
except KeyError:
    print("Error: OPEN_AI_KEY environment variable not set.")


