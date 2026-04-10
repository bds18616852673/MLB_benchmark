# MLB_benchmark

Official resources for **MLB: A Scenario-Driven Benchmark for Evaluating Large Language Models in Clinical Applications**.

This repository currently releases the benchmark metadata, representative examples, inference prompts, evaluation prompts, and training / inference / evaluation scripts for SFT-based judge model used in our experiments.

## Released Files

### Benchmark specification

- `MLB_benchmark_introduction_metadata_exampleQAdataset_prompt_evaluation_inference.json`
  - Main released benchmark description file.
  - Each dataset entry contains:
    - `name`
    - `intent`
    - `introduction`
    - `metadata`
    - `prompt_evaluation`
    - `prompt_inference`
    - `exampleQAdata`

- `prompt_inference_22datasets.json`
  - Inference prompts for the 22 benchmark datasets.
  - Used to construct task-specific model input templates during benchmark inference.

### Example pipeline scripts

- `step1_judge_model_sft_by_qwen3_14B_train_inference.sh`
  - Example end-to-end script for:
    - supervised fine-tuning
    - checkpoint selection
    - inference
    - evaluation
  - The script uses `swift sft` for LoRA training and `swift infer` for decoding with the vLLM backend.
  - Before running, update placeholder paths such as:
    - `root_path`
    - `QWEN3_14B_model_path`
    - `train_file_path.jsonl`
    - `test_file_path.jsonl`

- `step2_judge_model_evaluation.py`
  - Example binary-evaluation script.
  - Reads prediction results in JSONL format, extracts model answers from `response`, and computes metrics such as:
    - accuracy
    - precision
    - recall
    - F1
    - confusion-matrix-based statistics
  - Exports the evaluation table to Excel.

## Benchmark File Format

The merged benchmark file `MLB_benchmark_introduction_metadata_exampleQAdataset_prompt_evaluation_inference.json` is organized as:

```json
{
  "DatasetName": {
    "name": "...",
    "type": "...",
    "intent": "...",
    "quantity": 300,
    "introduction": "...",
    "metadata": "...",
    "prompt_evaluation": "...",
    "prompt_inference": "...",
    "exampleQAdata": {
      "question": "...",
      "answer": "..."
    }
  }
}
```

Notes:

- `prompt_inference` is the task prompt used to query the model being evaluated.
- `prompt_evaluation` is the evaluator / judge prompt. For some tasks it may be `/`, indicating direct metric-based evaluation instead of judge-model scoring.
- `exampleQAdata` provides one representative example from the dataset and is intended only for illustration.

## Notes

- The released prompt and benchmark JSON files are intended to make the benchmark setup transparent and reproducible for reviewers and researchers.
