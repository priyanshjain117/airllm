from airllm import AutoModel
from transformers import AutoTokenizer
import mlx.core as mx
import dotenv
from dotenv import load_dotenv

load_dotenv()

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModel.from_pretrained(
    model_id,
    layer_shards_saving_path=dotenv.get_key(".env", "MODEL_PATH")
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Keep prompt as short as possible to avoid GPU timeout
prompt = "Name one movie:"

ids = tokenizer(prompt)["input_ids"]
inputs = mx.array([ids], dtype=mx.int32)

output = model.generate(
    inputs,
    max_new_tokens=5,   # as low as possible
    temperature=0
)

print(output)