from airllm import AutoModel
from transformers import AutoTokenizer
import dotenv
from dotenv import load_dotenv

load_dotenv()

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModel.from_pretrained(
    model_id,
    layer_shards_saving_path=dotenv.get_key(".env", "MODEL_PATH"),
    use_mlx=False
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "Answer in exactly four words."},
    {"role": "user", "content": "Tell a movie name."}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

output = model.generate(ids, max_new_tokens=20, temperature=0)
print(tokenizer.decode(output[0], skip_special_tokens=True))