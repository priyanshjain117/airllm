from mlx_lm import load, generate
from dotenv import load_dotenv
import dotenv
import os

load_dotenv()

model_path = os.path.join(dotenv.get_key(".env", "MODEL_PATH"), "llama3-mlx-4bit")

model, tokenizer = load(model_path)

messages = [
    {"role": "system", "content": "Answer in exactly four words."},
    {"role": "user", "content": "Tell a movie name."}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=20,
    verbose=True
)

print(response)