#!/bin/bash

# 定义要评估的数据集和模型
DATASETS=("finance_SP" "LargeAQ" "power" "traffic")
MODELS=("gpt-4o-mini" "gemini-2.0-flash" "grok-3-mini")

# 定义基础文件名，假设您的 RAG 预测结果都以此为基础
BASE_FILENAME="predictions.csv"

MODEL_TYPES=("llm" "vlm")

MECHANISMS=("direct" "cot" "fewshot")

# 定义最终结果汇总文件名
SUMMARY_FILE="evaluation_summary.csv"

# 初始化汇总文件并写入表头
echo "dataset,file_name,f1_micro,auroc_micro" > "$SUMMARY_FILE"
echo "Created summary file: $SUMMARY_FILE"


echo "Starting evaluation for all datasets and models..."
echo "=================================================="

# 遍历所有数据集
for dataset in "${DATASETS[@]}"
do
  echo ""
  echo "--- Processing Dataset: ${dataset} ---"
  
  # 遍历所有模型
  for model in "${MODELS[@]}"
  do
    for model_type in "${MODEL_TYPES[@]}"
    do
      for mechanism in "${MECHANISMS[@]}"
      do
      # 构造预测结果文件的路径
        FILE_PATH="${model}_${model_type}_${mechanism}_${BASE_FILENAME}"

        # 检查文件是否存在
        echo ""
        echo "--- Evaluating Model: ${model} on Dataset: ${dataset} ---"
#        python evaluate_rag_results.py --dataset_name "$dataset" --file_name "$FILE_PATH"


        # 运行评估脚本并捕获其输出 (包括 stderr, 因为 logging 输出到 stderr)
        output=$(python evaluate_rag_results.py --dataset_name "$dataset" --file_name "$FILE_PATH" 2>&1)

        # 从输出中提取 f1-micro 和 auroc-micro
        f1_micro=$(echo "$output" | grep "F1 Score (Micro):" | awk '{print $NF}')
        auroc_micro=$(echo "$output" | grep "AUROC (Micro, OvR, from discrete predictions):" | awk '{print $NF}')

        # 如果没有提取到值，则设置为 N/A
        [ -z "$f1_micro" ] && f1_micro="N/A"
        [ -z "$auroc_micro" ] && auroc_micro="N/A"

        echo "  -> F1-Micro: $f1_micro, AUROC-Micro: $auroc_micro"

         # 将结果追加到汇总CSV文件中
        echo "$dataset,$FILE_PATH,$f1_micro,$auroc_micro" >> "$SUMMARY_FILE"
        echo "------------------------------------------"


#        if [ -f "$FILE_PATH" ]; then
#          echo ""
#          echo "--- Evaluating Model: ${model} on Dataset: ${dataset} ---"
#          python evaluate_rag_results.py --dataset_name "$dataset" --file_path "$FILE_PATH"
#        else
#          echo ""
#          echo "--- Skipping Model: ${model} on Dataset: ${dataset} (File not found: ${FILE_PATH}) ---"
#        fi
      done
    done
  done
  echo "=================================================="
done

echo "All evaluations complete."