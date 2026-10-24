import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN
)


def get_local_recommendation(disease_name):
    return f"""
### 🌱 Disease Description
{disease_name} is a plant disease that can affect plant growth and overall health.

### 🔍 Symptoms
- Yellowing or discoloration of leaves
- Spots or lesions on leaves
- Reduced plant growth
- Wilting in severe cases

### 🦠 Causes
The disease may be caused by fungal, bacterial, viral or environmental factors depending on the plant and disease.

### 🌿 Organic Treatment
- Remove affected leaves
- Maintain proper air circulation
- Avoid excessive watering
- Use appropriate organic/fungicidal treatment

### 🧪 Chemical Treatment
Use a suitable agricultural fungicide or pesticide recommended for the identified disease and follow the product label carefully.

### 🛡️ Prevention Tips
- Keep plants healthy and well-spaced
- Avoid overwatering
- Remove infected plant material
- Maintain good garden hygiene
"""


def get_treatment_recommendation(disease_name):

    # Try Hugging Face AI first
    try:
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

    # If HF credits/API unavailable, keep application working
    except Exception:
        return get_local_recommendation(disease_name)