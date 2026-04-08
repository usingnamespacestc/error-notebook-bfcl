import os
import json
import torch
from vllm import LLM, SamplingParams
from typing import List, Dict, Tuple, Optional
import numpy as np

class TeacherForcingScorer:
    def __init__(self, model_path: str = "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit", gpu_memory_utilization: float = 0.8):
        self.llm = LLM(
            model=model_path,
            quantization="compressed-tensors",
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            max_model_len=4096,
            enforce_eager=True # Recommended for quantization if memory is tight
        )
        self.tokenizer = self.llm.get_tokenizer()
        
    def score_batch_raw(self, batch_prefixes: List[str], batch_full_prompts: List[str]) -> List[Dict]:
        """
        Single-pass teacher forcing on pre-rendered prompt strings.
        Computes log P(gold_tokens | prefix) where gold_tokens = full_prompt[len(prefix):].

        No thinking pass — the prefix ends right before the gold answer tokens,
        and we measure the raw logprob of the gold tool call.
        """
        n = len(batch_prefixes)
        if n == 0:
            return []

        max_len = self.llm.llm_engine.model_config.max_model_len

        # Tokenize and separate into scorable vs too-long
        prefix_ids_list = []
        full_ids_list = []
        valid_indices = []
        results = [None] * n

        for i in range(n):
            p_ids = self.tokenizer.encode(batch_prefixes[i])
            f_ids = self.tokenizer.encode(batch_full_prompts[i])
            prefix_ids_list.append(p_ids)
            full_ids_list.append(f_ids)

            if len(f_ids) > max_len:
                # Penalty for prompts that exceed context window
                gold_len = len(f_ids) - len(p_ids)
                results[i] = {
                    "log_prob_sum": -20.0 * gold_len,
                    "mean_log_prob": -20.0,
                    "gold_len_tokens": gold_len,
                }
            else:
                valid_indices.append(i)

        if valid_indices:
            valid_prompts = [batch_full_prompts[i] for i in valid_indices]
            sampling_params = SamplingParams(
                max_tokens=1,
                prompt_logprobs=1,
            )
            score_outputs = self.llm.generate(valid_prompts, sampling_params, use_tqdm=False)

            for out_idx, i in enumerate(valid_indices):
                prompt_logprobs = score_outputs[out_idx].prompt_logprobs
                full_ids = full_ids_list[i]
                prefix_len = len(prefix_ids_list[i])

                gold_ids = full_ids[prefix_len:]
                log_prob_sum = 0.0

                for j in range(prefix_len, len(full_ids)):
                    token_id = full_ids[j]
                    if j < len(prompt_logprobs) and prompt_logprobs[j] and token_id in prompt_logprobs[j]:
                        log_prob_sum += prompt_logprobs[j][token_id].logprob
                    else:
                        log_prob_sum -= 20.0

                results[i] = {
                    "log_prob_sum": log_prob_sum,
                    "mean_log_prob": log_prob_sum / len(gold_ids) if gold_ids else 0,
                    "gold_len_tokens": len(gold_ids),
                }

        return results

    def score_batch(self, batch_contexts: List[List[Dict]], batch_gold_answers: List[str]) -> List[Dict]:
        """
        Calculates log P(gold_answer | context_messages) for a batch of inputs.
        """
        n = len(batch_contexts)
        if n == 0:
            return []
            
        # Pass 1: Generate Thinking for the whole batch
        sampling_params_think = SamplingParams(
            temperature=0,
            max_tokens=1024,
            stop=["</think>"]
        )
        
        prompts = []
        for ctx in batch_contexts:
            prompt = self.tokenizer.apply_chat_template(ctx, tokenize=False, add_generation_prompt=True)
            if not prompt.endswith("<think>"):
                prompt += "<think>"
            prompts.append(prompt)
            
        outputs = self.llm.generate(prompts, sampling_params_think, use_tqdm=False)
        
        full_prompts = []
        prefix_ids_list = []
        thinking_contents = []
        
        for i in range(n):
            thinking_content = outputs[i].outputs[0].text
            thinking_contents.append(thinking_content)
            
            prefix_text = prompts[i] + thinking_content + "</think>"
            full_text = prefix_text + batch_gold_answers[i]
            
            full_prompts.append(full_text)
            prefix_ids_list.append(self.tokenizer.encode(prefix_text))
            
        # Pass 2: Batch Teacher Forcing
        sampling_params_score = SamplingParams(
            max_tokens=1,
            prompt_logprobs=1,
        )
        
        score_outputs = self.llm.generate(full_prompts, sampling_params_score, use_tqdm=False)
        
        results = []
        for i in range(n):
            prompt_logprobs = score_outputs[i].prompt_logprobs
            full_ids = self.tokenizer.encode(full_prompts[i])
            prefix_ids = prefix_ids_list[i]
            
            gold_ids = full_ids[len(prefix_ids):]
            log_prob_sum = 0.0
            
            for j in range(len(prefix_ids), len(full_ids)):
                token_id = full_ids[j]
                if j < len(prompt_logprobs) and prompt_logprobs[j] and token_id in prompt_logprobs[j]:
                    log_prob_sum += prompt_logprobs[j][token_id].logprob
                else:
                    log_prob_sum -= 20.0
                    
            results.append({
                "log_prob_sum": log_prob_sum,
                "mean_log_prob": log_prob_sum / len(gold_ids) if gold_ids else 0,
                "gold_len_tokens": len(gold_ids),
                "thinking": thinking_contents[i]
            })
            
        return results

if __name__ == "__main__":
    # Quick test
    scorer = TeacherForcingScorer()
    ctx = [{"role": "user", "content": "1+1=?"}]
    gold = "2"
    res = scorer.score(ctx, gold)
    print(json.dumps(res, indent=2))
