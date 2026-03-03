from airllm import AutoModel
from transformers import AutoTokenizer
import dotenv
import mlx.core as mx
from dotenv import load_dotenv

load_dotenv()

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModel.from_pretrained(
    model_id,
    layer_shards_saving_path=dotenv.get_key(".env", "MODEL_PATH"),
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "Answer in exactly four words."},
    {"role": "user", "content": "Tell a movie name."}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

ids = tokenizer(prompt)["input_ids"]  # plain list, no mx.array

inputs = mx.array([ids], dtype=mx.int32)  # 1D, no extra brackets

output = model.generate(
    inputs,
    max_new_tokens=20,
    temperature=0
)

print(tokenizer.decode(output.tolist(), skip_special_tokens=True))