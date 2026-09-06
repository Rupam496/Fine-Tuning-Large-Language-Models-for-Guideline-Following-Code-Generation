import re

from fastapi import FastAPI
from pydantic import BaseModel

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from transformers import BitsAndBytesConfig
import torch

# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "/home/rupamdas/Desktop/Rupam/ET project/llm/Qwen2.5-7B-Instruct"

MAX_NEW_TOKENS = 50


# ============================================================
# Load Qwen model
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)



model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto"
)

model.eval()

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# Create Hugging Face generation pipeline
# ============================================================

generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,
    return_full_text=False
)


# ============================================================
# Create LangChain LLM
# ============================================================

llm = HuggingFacePipeline(
    pipeline=generation_pipeline
)


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI()


# ============================================================
# Request model
# ============================================================

class JudgeRequest(BaseModel):
    prompt: str


# ============================================================
# Generate scores using Qwen
# ============================================================

def generate_scores(prompt):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict Python code evaluation judge. "
                "You evaluate Python code according to coding guidelines."
            )
        },
        {
            "role": "user",
            "content": f"""
Evaluate Code 1 and Code 2 using the provided coding guidelines.

Give each code a score between 0 and 1.

Return ONLY two numbers in this exact format:

0.85, 0.65

The first number is the score for Code 1.
The second number is the score for Code 2.

Do not provide explanations.
Do not provide markdown.
Do not provide guidelines.
Do not provide code.
Do not provide any other text.

{prompt}
"""
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    response = llm.invoke(formatted_prompt)

    print("response by raw llm:", repr(response))

    return response.strip()


# ============================================================
# Extract two scores
# ============================================================

def extract_scores(response):

    numbers = re.findall(
        r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b",
        response
    )

    if len(numbers) < 2:
        raise ValueError(
            f"Judge did not return two valid scores: {response}"
        )

    score_1 = float(numbers[0])
    score_2 = float(numbers[1])
    print(f"scores are {score_1},{score_2}")
    return score_1, score_2


# ============================================================
# POST /judge
# ============================================================

@app.post("/judge")
def judge(request: JudgeRequest):

    response = generate_scores(
        request.prompt
    )

    score_1, score_2 = extract_scores(
        response
    )
    print("response sent")
    return {
        "score_1": score_1,
        "score_2": score_2
    }