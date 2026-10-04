from huggingface_hub import InferenceClient
#import google.generativeai as genai
import os
from dotenv import load_dotenv
from config import HF_TOKEN, GOOGLE_API_KEY, GEMINI_MODEL
#from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub import InferenceClient
import requests

#client = InferenceClient(
#    api_key="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"
#)

#response = client.chat.completions.create(
#    model="Qwen/Qwen2.5-7B-Instruct",
#    messages=[
#        {
#            "role": "user",
#            "content": "Say hello in one short sentence."
#        }
#   ],
#    max_tokens=20,
#    temperature=0.2,
#)

#print(response.choices[0].message.content)

#system_prompt = 'You are helpful assistant'
#user_prompt = 'Give me captial of India'
#max_tokens= 800
#temperature =0.2


#genai.configure(api_key=GOOGLE_API_KEY)

#model = genai.GenerativeModel(
#        model_name=GEMINI_MODEL,
#        system_instruction=system_prompt
#        )

#response = model.generate_content(
#           user_prompt,
#            generation_config=genai.types.GenerationConfig(
#            max_output_tokens=max_tokens,
#            temperature=temperature,
#            ))  

#print(response.text)

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    model="Qwen/Qwen3-14B",
    token=HF_TOKEN,
)

response = client.chat_completion(
    messages=[
        {
            "role": "user",
            "content": "What is the capital of India?"
        }
    ],
    max_tokens=500,
    temperature=0.2,
)

print(response.choices[0].message.content)