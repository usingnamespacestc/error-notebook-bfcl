from vllm import LLM, SamplingParams
import json

MODEL_PATH = "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"

# Initialize LLM
llm = LLM(model=MODEL_PATH, quantization="awq", trust_remote_code=True, gpu_memory_utilization=0.8)

data_1 = ["What is the capital of France?"]
data_2 = ["The capital of France is Paris."]

outputs = llm.score(data_1, data_2)

for output in outputs:
    print(f"Score: {output.logprobs}")
