from dotenv import load_dotenv
import os
from transformers import pipeline
env_path = os.path.join(os.path.dirname(__file__), "./trainingllm/.env")

load_dotenv(dotenv_path=env_path)

hf_token=os.getenv("huggingfacetoken")
print(hf_token)

from transformers import AutoModelForSequenceClassification, AutoTokenizer

token = hf_token
model_id = "UmarKhattab09/llmfinetuning"

model = AutoModelForSequenceClassification.from_pretrained(model_id, use_auth_token=token)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
model_name = "UmarKhattab09/llmfinetuning"
classifier = pipeline(
    "text-classification",
    model=model_name,
    tokenizer=model_name,
    )   

