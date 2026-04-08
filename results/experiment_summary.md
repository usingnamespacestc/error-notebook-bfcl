# Error Notebook Experiment Summary

## Key Finding: Checker Bug Fix
A critical bug was found in `bfcl_eval/constants/model_config.py`: `underscore_to_dot` defaulted to `True` for unknown models, causing function names with dots (e.g., `uber.eat.order`) to be incorrectly converted to underscores (`uber_eat_order`) during validation. This inflated `wrong_func_name` errors by ~25pp across all evaluations.

**All accuracy numbers below are CORRECTED.**

## Results Overview

| Condition | Accuracy | Total | Notes |
|-----------|----------|-------|-------|
| Gemma4 26B zero-shot | **75.19%** | 782 | Local Ollama |
| Gemma4 selected 5-shot | 75.11% | 466 | Incomplete, SA-selected |
| Volcengine Auto zero-shot | 66.88% | 782 | ark-code-latest |
| **Volcengine Auto error-notebook** | **74.76%** | 737 | **+7.88pp vs zero-shot** |
| Doubao-Seed-2.0-Code (think) | 72.63% | 782 | |
| Doubao-Seed-2.0-Code (nothink) | 66.62% | 782 | |

## Error Notebook: Per-Category Breakdown (Volcengine Auto)

| Category | Zero-shot | Error Notebook | Delta |
|----------|-----------|---------------|-------|
| parallel | 0.0% | **80.4%** | **+80.4pp** |
| live_parallel | 0.0% | **80.0%** | **+80.0pp** |
| parallel_multiple | 18.3% | **68.4%** | **+50.1pp** |
| live_parallel_multiple | 37.5% | **62.5%** | **+25.0pp** |
| javascript | 33.3% | 35.7% | +2.4pp |
| sql | 13.3% | 13.8% | +0.5pp |
| java | 50.0% | 50.0% | 0.0pp |
| multiple | 95.0% | 94.4% | -0.6pp |
| simple | 95.0% | 92.8% | -2.2pp |
| live_simple | 83.3% | 78.7% | -4.7pp |
| live_multiple | 78.8% | 74.2% | -4.6pp |

## Method: Error Notebook Pipeline

1. **Pool Inference**: Run model on pool.jsonl (training data) to get predictions
2. **Error Classification**: Use AST checker to classify each error type
3. **K-Medoids Selection**: Select k=5 correct + k=5 error examples
   - Correct: diverse representatives via KMeans clustering on embeddings
   - Errors: proportional budget per error type, clustered within type
4. **Prompt Construction**:
   - System message: "Pay attention to parameter types and values"
   - k correct examples (standard few-shot)
   - k error-correction examples (wrong call -> user feedback -> correct call)
   - Target question with only its own tool definitions

## Error Distribution (Volcengine Auto, corrected checker)

| Error Type | Count | % of Errors |
|-----------|-------|------------|
| parallel wrong_count | 122 | 49.6% |
| value_error:string | 38 | 15.4% |
| value_error:list/tuple | 19 | 7.7% |
| type_error:java | 18 | 7.3% |
| value_error:others | 14 | 5.7% |

## Analysis

The error notebook approach achieved a **+7.88pp** improvement over zero-shot on the Volcengine Auto model. The improvement is concentrated in **parallel function calling** categories, where the model went from near-zero to 60-80% accuracy. This makes sense because:

1. The dominant error type (49.6%) was `parallel wrong_count` - the model wasn't calling enough functions
2. The error notebook included 2 error-correction examples showing this exact mistake pattern
3. The model learned from seeing "you called 1 function but should have called 2-3"

Categories with slight regressions (-2 to -5pp) are likely due to the longer context from 10 few-shot messages slightly degrading attention on the target question.

## Files Created

- `eval/run_pool_inference.py` - Local Ollama pool inference
- `eval/volcengine_pool_inference.py` - Volcengine API pool inference
- `eval/classify_pool_errors.py` - Error classification with AST checker
- `search/error_notebook_selection.py` - K-Medoids representative selection
- `eval/error_notebook_eval.py` - Error notebook evaluation (Ollama)
- `eval/volcengine_error_notebook_eval.py` - Error notebook evaluation (Volcengine)
- `eval/bfcl_eval/constants/model_config.py` - Fixed underscore_to_dot default

## Next Steps

- [ ] Complete Gemma4 error notebook eval (running on local Ollama)
- [ ] Try k=3 or k=8 to find optimal error notebook size
- [ ] Try error notebook on Doubao-Seed-2.0-Code
- [ ] Focus on improving sql and java categories (still weak)
- [ ] Consider adding sql-specific and java-specific error examples
