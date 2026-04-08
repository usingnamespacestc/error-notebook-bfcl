import json
import os
import sys
from typing import List, Dict, Any
from tqdm import tqdm

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scoring.teacher_forcing import TeacherForcingScorer
from scoring.format_utils import convert_bfcl_to_ollama_tc, build_fewshot_messages, merge_tools

class MutualPredictabilityScorer:
    def __init__(self, teacher_forcing_scorer: TeacherForcingScorer):
        self.tf_scorer = teacher_forcing_scorer

    def _get_question_content(self, entry: Dict) -> str:
        if isinstance(entry["question"][0], list):
            return entry["question"][0][0]["content"]
        return entry["question"][0]["content"]

    def _build_ollama_tools(self, tool_defs: List[Dict]) -> List[Dict]:
        """Convert BFCL function defs to Ollama tools format for chat template."""
        tools = []
        for func in tool_defs:
            f = dict(func)
            if f.get("parameters", {}).get("type") == "dict":
                f["parameters"] = dict(f["parameters"])
                f["parameters"]["type"] = "object"
            tools.append({"type": "function", "function": f})
        return tools

    def score_subset(self, subset: List[Dict]) -> float:
        """
        P_θ(S) = Σ_{i=1}^{k} log P_θ(y_i | tool_defs, S\{i}, x_i)

        Uses the tokenizer's chat template to render tool calls in Gemma 4's
        native format (e.g. <|tool_call>call:func{args}<tool_call|>).
        """
        k = len(subset)
        if k == 0:
            return 0.0

        tokenizer = self.tf_scorer.tokenizer
        batch_prefixes = []
        batch_full_prompts = []

        for i in range(k):
            target = subset[i]
            others = subset[:i] + subset[i+1:]

            # Merge tool definitions from all examples
            all_tool_defs = merge_tools(target["function"], others)
            ollama_tools = self._build_ollama_tools(all_tool_defs)

            # Build messages: few-shot examples (with tool_calls) + target question
            messages = build_fewshot_messages(others)
            messages.append({"role": "user", "content": self._get_question_content(target)})

            # Full prompt: messages + gold tool call (rendered by chat template)
            gold_tc = convert_bfcl_to_ollama_tc(target["ground_truth"])
            messages_with_gold = list(messages) + [{
                "role": "assistant",
                "content": "",
                "tool_calls": gold_tc
            }]
            full_prompt = tokenizer.apply_chat_template(
                messages_with_gold, tokenize=False, add_generation_prompt=False, tools=ollama_tools
            )

            # Prefix: everything up to the last model turn's content
            # full_prompt ends with: ...<|turn>model\n<|tool_call>...<tool_call|><turn|>\n
            # prefix = everything up to and including <|turn>model\n
            last_model_turn = full_prompt.rfind("<|turn>model\n")
            prefix = full_prompt[:last_model_turn + len("<|turn>model\n")]

            batch_prefixes.append(prefix)
            batch_full_prompts.append(full_prompt)

        # Use teacher forcing to compute logprobs of gold answer tokens
        results = self.tf_scorer.score_batch_raw(batch_prefixes, batch_full_prompts)
        total_log_prob = sum(res["log_prob_sum"] for res in results)

        return total_log_prob


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    with open(os.path.join(BASE_DIR, "data", "pool.jsonl"), "r") as f:
        pool = [json.loads(line) for line in f]

    tf_scorer = TeacherForcingScorer()
    mp_scorer = MutualPredictabilityScorer(tf_scorer)

    subset = pool[:3]
    print(f"Scoring subset of size {len(subset)}...")
    score = mp_scorer.score_subset(subset)
    print(f"Mutual Predictability Score: {score}")
