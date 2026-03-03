from airllm import AutoModel
from transformers import AutoTokenizer
import dotenv

load_dotenv()

model_path = dotenv.get_key(".env", "MODEL_PATH") + "/Llama3-8B"
model = AutoModel.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    model_local_path=model_path
)

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct"
)

prompt = "Explain transformers in simple terms."

inputs = tokenizer(prompt, return_tensors="np")["input_ids"]

output = model.generate(inputs, max_new_tokens=100)

print(tokenizer.decode(output[0]))