import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN
)


def get_treatment_recommendation(disease_name):
    response = client.chat.completions.create(
        model="Qwen/Qwen3-4B-Instruct-2507",
        messages=[
            {
                "role": "user",
                "content": f"""
The predicted plant disease is: {disease_name}.

Provide:
1. Disease Description
2. Symptoms
3. Causes
4. Organic Treatment
5. Chemical Treatment
6. Prevention Tips

Keep the response well-structured.
"""
            }
        ],
        max_tokens=500,
    )

    return response.choices[0].message.content