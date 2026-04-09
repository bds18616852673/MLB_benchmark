
set -euo pipefail

# 设置NCCL环境变量，减少日志输出
export NCCL_DEBUG=WARN  # 只显示警告和错误，不显示INFO
export NCCL_DEBUG_SUBSYS=ALL  # 可以进一步限制子系统

# 设置CUDA设备和内存管理
nproc_per_node=4
CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 根目录配置
root_path="your_root_path"
QWEN3_14B=Qwen3_14B_model_path
QWEN3_14B_MODEL=${QWEN3_14B}  

for i in 14B; do  #8B 4B
    for j in v1; do #dataset
        experiment_name="qwen3-${i}-sft-${j}"
        output_dir="${root_path}/${experiment_name}"
        train_file="${root_path}/train_file_path.jsonl"
        eval_file="${root_path}/test_file_path.jsonl"
        mkdir -p "${output_dir}"
        logs_dir="${output_dir}/logs"
        mkdir -p "${logs_dir}"
        train_log_file="${logs_dir}/train_${j}.log"
  
        # 生成10000-65535之间的随机端口（避免固定端口冲突）
        PORT=$((10000 + RANDOM % 55535))
        echo "生成随机端口: ${PORT}"
        # 强制设置分布式环境变量（优先级高于命令行参数）
        export MASTER_ADDR=localhost
        export MASTER_PORT=${PORT}
        export WORLD_SIZE=1  # 与nproc_per_node一致
        export RANK=0        # 单节点训练时设为0
        echo "端口 ${PORT} 可用"
        # TensorBoard配置
        tensorboard_dir="${output_dir}/tensorboard"
        mkdir -p "${tensorboard_dir}"
        
        echo "===== 训练参数 ====="
        echo "数据集路径: ${train_file}"
        echo "输出目录: ${output_dir}"
        echo "TensorBoard目录: ${tensorboard_dir}"
        echo "模型版本: qwen3_${i}_sft_${j}"
        echo "===================="

        NPROC_PER_NODE=$nproc_per_node \
        swift sft \
            --model ${QWEN3_14B_MODEL} \
            --model_type qwen3 \
            --train_type lora \
            --dataset "${train_file}" \
            --torch_dtype bfloat16 \
            --num_train_epochs 3 \
            --per_device_train_batch_size 2 \
            --per_device_eval_batch_size 2 \
            --learning_rate 2e-5 \
            --lora_rank 8 \
            --lora_alpha 32 \
            --target_modules all-linear \
            --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
            --eval_steps 50 \
            --save_steps 50 \
            --save_total_limit 5 \
            --logging_steps 5 \
            --max_length 8192 \
            --output_dir "${output_dir}" \
            --warmup_ratio 0.05 \
            --dataloader_num_workers 4 \
            --split_dataset_ratio 0.05 \
            --model_author wanyi \
            --ddp_find_unused_parameters False \
            --loss_scale ignore_empty_think \
            --deepspeed zero3 \
            --report_to tensorboard \
            --logging_dir "${tensorboard_dir}" \
            2>&1 | tee -a "${train_log_file}"

        # TensorBoard配置（不启动服务）
        echo "[INFO] TensorBoard日志目录: ${tensorboard_dir}"
        # 开始推理阶段
        echo "[INFO] 训练已完成，开始推理阶段..."
        # 推理
        echo "[INFO] 开始推理阶段..."
        # 清理分布式训练环境变量，避免推理时的冲突
        unset MASTER_ADDR
        unset MASTER_PORT
        unset WORLD_SIZE
        unset RANK
        unset LOCAL_RANK
        # 优先依据 trainer_state.json 或训练日志中的最优/最后 checkpoint
        # 现在无需手动指定 checkpoint；脚本会优先使用 best，其次 last，再次最新时间戳下的最大 checkpoint。
        state_file="${output_dir}/trainer_state.json"
        best_ckpt=""
        last_ckpt=""
        if [ -f "${state_file}" ]; then
            echo "[INFO] 从 trainer_state.json 读取 checkpoint 信息"
            best_ckpt=$(grep -o '"best_model_checkpoint"[^,}]*' "${state_file}" | sed 's/.*:"\\?\([^"\\]*\)".*/\1/' | head -n 1)
            last_ckpt=$(grep -o '"last_model_checkpoint"[^,}]*' "${state_file}" | sed 's/.*:"\\?\([^"\\]*\)".*/\1/' | head -n 1)
        else
            echo "[INFO] trainer_state.json 不存在，从训练日志提取 checkpoint 信息"
            # 从训练日志提取（示例行：[INFO:swift] best_model_checkpoint: /path/to/checkpoint-xxx）
            if [ -f "${train_log_file}" ]; then
                # 使用 -a 参数处理二进制文件，避免 "binary file matches" 错误
                best_ckpt=$(grep -a -E '\[INFO:swift\] best_model_checkpoint:' "${train_log_file}" | tail -n 1 | sed 's/^.*best_model_checkpoint:\s*//')
                last_ckpt=$(grep -a -E '\[INFO:swift\] last_model_checkpoint:' "${train_log_file}" | tail -n 1 | sed 's/^.*last_model_checkpoint:\s*//')
            fi
        fi
        if [ -n "${best_ckpt}" ] && [ -d "${best_ckpt}" ]; then
            ADAPTERS_PATH="${best_ckpt}"
            echo "[INFO] 使用 best_model_checkpoint: ${ADAPTERS_PATH}"
        elif [ -n "${last_ckpt}" ] && [ -d "${last_ckpt}" ]; then
            ADAPTERS_PATH="${last_ckpt}"
            echo "[INFO] 使用 last_model_checkpoint: ${ADAPTERS_PATH}"
        else
            echo "[INFO] 从日志中未找到有效 checkpoint，尝试查找最新 checkpoint"
            # 两级选择：先选最新时间戳目录(v*-YYYYmmdd-HHMMSS)，再选该目录下最大 checkpoint
            latest_stage_dir=$(ls -1d ${output_dir}/v*-????????-?????? 2>/dev/null | sort | tail -n 1)
            if [ -n "$latest_stage_dir" ]; then
                latest_ckpt_dir=$(ls -1d ${latest_stage_dir}/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
            fi
            # 若时间戳目录未找到或其中无 checkpoint，则在 output_dir 下全局回退查找最大 checkpoint
            if [ -z "$latest_ckpt_dir" ]; then
                latest_ckpt_dir=$(find "${output_dir}" -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1)
            fi
            # 若仍未找到，则回退为 output_dir
            if [ -n "$latest_ckpt_dir" ]; then
                ADAPTERS_PATH="$latest_ckpt_dir"
                echo "[INFO] 使用最新 checkpoint: ${ADAPTERS_PATH}"
            else
                ADAPTERS_PATH="${output_dir}"
                echo "[WARN] 未找到 checkpoint，使用 output_dir: ${ADAPTERS_PATH}"
            fi
        fi
        infer_log_file="${logs_dir}/infer_${j}.log"
        echo "[INFO] 开始推理，使用模型: ${QWEN3_14B_MODEL}"
        echo "[INFO] 推理结果将保存到: ${ADAPTERS_PATH}/${j}.jsonl" # 确保CUDA环境变量正确设置
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        echo "[INFO] 已清理分布式训练环境变量，设置CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        swift infer \
          --model ${QWEN3_14B_MODEL} \
          --adapters ${ADAPTERS_PATH} \
          --stream false \
          --infer_backend vllm \
          --tensor_parallel_size 4 \
          --gpu_memory_utilization 0.9 \
          --max_model_len 8192 \
          --temperature 0 \
          --val_dataset "${eval_file}" \
          --result_path ${ADAPTERS_PATH}/${j}.jsonl \
          2>&1 | tee -a "${infer_log_file}"
          

        # pt framework
        # --infer_backend pt \
        #   --max_batch_size 16 \
        #   --device_map auto \
        #   --torch_dtype bfloat16 \

        # sglang framework
        #   --infer_backend sglang \
        #   --sglang_context_length 8192 \
        #   --sglang_tp_size 4 \
        # --response_prefix '<think>\n\n</think>\n\n' \
        #   --model_type qwen \
        #   --template qwen \
        #   --template_backend jinja \
        # --max_batch_size 16 \
        #   --infer_backend pt \
        #   --max_new_tokens 4096 \
        #   --write_batch_size 4 \
        #   --device_map auto \
        #   --torch_dtype bfloat16 \

        # evaluation
        result_file="${ADAPTERS_PATH}/${j}.jsonl"
        score_file="${ADAPTERS_PATH}/${j}_score.xlsx"
        score_log_file="${logs_dir}/score_${j}.log"
        echo "[INFO] Starting evaluation..."
        python ${root_path}/step2_evaluation.py \
          --pred_file "${result_file}" \
          --output_file "${score_file}" \
          2>&1 | tee -a "${score_log_file}"

    done
done


# sh step1_qwen3_14B_train_inference_evaluation.sh
