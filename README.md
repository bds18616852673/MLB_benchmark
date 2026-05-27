# MLB_benchmark

Official resources for **MLB: A Scenario-Driven Benchmark for Evaluating Large Language Models in Clinical Applications**.

This repository releases the benchmark metadata, representative examples, inference / evaluation prompts, and the example training, inference, and evaluation scripts of the SFT-based judge model used in our experiments.

---

## Paper

- **Title**: MLB: A Scenario-Driven Benchmark for Evaluating Large Language Models in Clinical Applications
- **arXiv**: [https://arxiv.org/abs/2601.06193](https://arxiv.org/abs/2601.06193)
- **Conference**: 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD 2026), Jeju Island, Republic of Korea.
- **Authors**: Qing He*, Dongsheng Bi*, Jianrong Lu*, Minghui Yang, Zixiao Chen, Jiacheng Lu, Jing Chen, Nannan Du, Xiao Cui, Sijing Wu, Peng Xiang, Yingying Hu, Yi Guo, Shaoyang Li, Zhuo Dong, Ming Jiang, Shuai Guo, Liyun Feng, Jin Peng, Zhou Yang, Han Ying, Jie Zheng, Yujie Yang, Jian Wang, Jinjie Gu, Junwei Liu†
  - `*` Equal contribution (co-first authors).
  - `†` Corresponding author: Junwei Liu.

---

## Dataset Access

Since MLB contains proprietary clinical data curated under hospital confidentiality agreements, the **full datasets will be released via a controlled application process**. Please fill out the application form to request the access of downloading the MLB benchmark datasets:

- **Application Portal**: [http://221.12.19.52:8089](http://221.12.19.52:8089)

### Fields required on the application form


| Field                        | Description                                                                                                                                                                                                      |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **机构名称 (Institution)**       | The institution / affiliation of the applicant.                                                                                                                                                                  |
| **联系人 (Contact Person)**     | Full name of the requester.                                                                                                                                                                                      |
| **联系邮箱 (Contact Email)**     | Official institutional email (preferred) for delivery of access credentials.                                                                                                                                     |
| **论文信息 (Paper Information)** | He Q, Bi D, Lu J, et al. *MLB: A Scenario-Driven Benchmark for Evaluating Large Language Models in Clinical Applications*. arXiv preprint, [https://arxiv.org/abs/2601.06193](https://arxiv.org/abs/2601.06193). |


After clicking **提交申请 (Submit)**, the team will review the request and respond with the datasets and the data-use agreement.

---

## Released Files

### Benchmark specification

`MLB_benchmark_introduction_metadata_exampleQAdataset_prompt_evaluation_inference.json`


| Field               | Description                                                                                                                                                       |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`              | Dataset display name.                                                                                                                                             |
| `intent`            | Capability being evaluated (e.g., medical knowledge, diagnostic assistance).                                                                                      |
| `quantity`          | Official number of samples in the dataset after stratified sampling (100–300).                                                                                    |
| `introduction`      | Short description of the dataset.                                                                                                                                 |
| `metadata`          | Schema description of one record.                                                                                                                                 |
| `prompt_inference`  | Task prompt used to query the model being evaluated.                                                                                                              |
| `prompt_evaluation` | Evaluator / judge prompt. For datasets with a verifiable ground truth, this may be `/`, indicating direct metric-based evaluation instead of judge-model scoring. |
| `exampleQAdata`     | One representative `{question, answer}` example, intended only for illustration.                                                                                  |


`prompt_inference_22datasets.json` — inference prompts for all 22 benchmark datasets, ready to use as task-specific model input templates during benchmark inference.

### Example pipeline scripts

- `step1_judge_model_sft_by_qwen3_14B_train_inference.sh`
  - End-to-end example script for:
    - supervised fine-tuning
    - checkpoint selection
    - inference
    - evaluation
  - Uses `swift sft` for LoRA training and `swift infer` for decoding with the vLLM backend.
  - Before running, update placeholder paths such as:
    - `root_path`
    - `QWEN3_14B_model_path`
    - `train_file_path.jsonl`
    - `test_file_path.jsonl`
- `step2_judge_model_evaluation.py`
  - Binary-evaluation example script.
  - Reads prediction results in JSONL format, extracts model answers from `response`, and computes metrics such as:
    - accuracy
    - precision
    - recall
    - F1
    - confusion-matrix-based statistics
  - Exports the evaluation table to Excel.

### Per-dataset schema files

The subfolder `[MLB_benchmark_dataset/](MLB_benchmark_dataset/)` ships **one JSON file per dataset** (`<DatasetName>.json`, 22 files in total). The full N=100–300 samples per dataset are released via the [Dataset Access](#dataset-access) application portal.

---

## Citation

If you find MLB useful in your research, please cite:

```bibtex
@article{he2026mlb,
  title   = {MLB: A Scenario-Driven Benchmark for Evaluating Large Language Models in Clinical Applications},
  author  = {He, Qing and Bi, Dongsheng and Lu, Jianrong and Yang, Minghui and Chen, Zixiao and Lu, Jiacheng and Chen, Jing and Du, Nannan and Cui, Xiao and Wu, Sijing and Xiang, Peng and Hu, Yingying and Guo, Yi and Li, Shaoyang and Dong, Zhuo and Jiang, Ming and Guo, Shuai and Feng, Liyun and Peng, Jin and Yang, Zhou and Ying, Han and Zheng, Jie and Yang, Yujie and Wang, Jian and Gu, Jinjie and Liu, Junwei},
  journal = {arXiv preprint arXiv:2601.06193},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.06193}
}
```

