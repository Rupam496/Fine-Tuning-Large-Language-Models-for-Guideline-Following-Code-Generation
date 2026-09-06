from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from retriever import get_judge_prompt
from sender import send_to_judge
from peft import LoraConfig
from transformers import AutoTokenizer


# ============================================================
# Hyperparameters
# ============================================================

NUM_GENERATIONS = 2
# MAX_PROMPT_LENGTH = 256
MAX_COMPLETION_LENGTH = 680

LEARNING_RATE = 4e-5
NUM_EPOCHS = 3
BATCH_SIZE = 2


# ============================================================
# Paths
# ============================================================

MODEL_PATH = "/home/rupamdas/Rupam/ET project/Code generation following guidelines/llm"

DATASET_PATH = "/home/rupamdas/Rupam/ET project/Code generation following guidelines/prompts.txt"


# ============================================================
# Load questions
# ============================================================

questions = []

with open(DATASET_PATH, "r", encoding="utf-8") as file:

    for question in file:

        if question.strip():

            questions.append(question.strip())


SYSTEM_PROMPT = """You are an expert Python programmer.

Solve the programming task given by the user.

Rules:
1. Return only the Python code that solves the requested task.
2. Do not provide explanations outside the code.
3. Do not generate an answer for another question.
4. Do not continue with another question.
5. Stop immediately after completing the requested task.
"""


dataset = Dataset.from_dict({
    "prompt": [
        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ]
        for question in questions
    ],
    "question": questions
})


# ============================================================
# Reward function
# ============================================================

def reward_function(completions, question, **kwargs):
    rewards = []

    for i in range(0, len(completions), NUM_GENERATIONS):
        codes = completions[i:i + NUM_GENERATIONS]
        current_question = question[i]

        print("\n" + "=" * 80)
        print("QUESTION:")
        print(current_question)

        for j, code in enumerate(codes):
            print(f"\n--- COMPLETION {j + 1} ---")
            print(code)
            print(f"Length: {len(code)}")

        print("=" * 80)

        judge_prompt = get_judge_prompt(current_question, codes)

        print("request sent")
        score_1, score_2 = send_to_judge(judge_prompt)
        print("response came")

        rewards.append(score_1)
        rewards.append(score_2)

    return rewards

# ============================================================
# PEFT configuration
# ============================================================

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
)
# ============================================================
# GRPO configuration
# ============================================================

training_args = GRPOConfig(
    output_dir="./grpo_output",

    learning_rate=LEARNING_RATE,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,

    num_generations=NUM_GENERATIONS,

    max_completion_length=MAX_COMPLETION_LENGTH,

    gradient_checkpointing=True,

    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },

    remove_unused_columns=False,

    logging_steps=1,

    save_steps=100,
)


# ============================================================
# GRPO trainer
# ============================================================

trainer = GRPOTrainer(
    model=MODEL_PATH,
    reward_funcs=reward_function,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
)


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("EOS token:", tokenizer.eos_token)
print("EOS token ID:", tokenizer.eos_token_id)

print("PAD token:", tokenizer.pad_token)
print("PAD token ID:", tokenizer.pad_token_id)
# ============================================================
# Start training
# ============================================================

trainer.train()


# ============================================================
# Save trained model
# ============================================================

trainer.save_model("./grpo_output")