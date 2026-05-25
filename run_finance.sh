#python run_rag_prediction.py --dataset_name finance_SP --hybrid_alpha 0.9 --output_file rag_predictions_9.csv > run_finance_SP4.log 2>&1
#
#python run_rag_prediction.py --dataset_name finance_SP --hybrid_alpha 0.8 --output_file rag_predictions_8.csv > run_finance_SP5.log 2>&1
#
#python run_rag_prediction.py --dataset_name finance_SP --hybrid_alpha 0.7 --output_file rag_predictions_7.csv > run_finance_SP6.log 2>&1
#
#python run_rag_prediction.py --dataset_name finance_SP --hybrid_alpha 0.6 --output_file rag_predictions_6.csv > run_finance_SP7.log 2>&1
#
#python run_rag_prediction.py --dataset_name finance_SP --hybrid_alpha 0.4 --output_file rag_predictions_4.csv > run_finance_SP8.log 2>&1
#
#python run_rag_prediction.py --dataset_name finance_SP --hybrid_alpha 0.3 --output_file rag_predictions_3.csv > run_finance_SP9.log 2>&1
#
#python run_rag_prediction.py --dataset_name finance_SP --hybrid_alpha 0.2 --output_file rag_predictions_2.csv > run_finance_SP10.log 2>&1
#
#python run_rag_prediction.py --dataset_name finance_SP --hybrid_alpha 0.1 --output_file rag_predictions_1.csv > run_finance_SP11.log 2>&1

#python run_rag_prediction.py --dataset_name finance_SP --hybrid_alpha 0.2 --top_k 1  --output_file TopK1_rag_predictions.csv > run_finance_SP_ToPK1.log 2>&1
#
#python run_rag_prediction.py --dataset_name finance_SP --hybrid_alpha 0.2 --top_k 3  --output_file TopK3_rag_predictions.csv > run_finance_SP_ToPK3.log 2>&1
#
#python run_rag_prediction.py --dataset_name finance_SP --hybrid_alpha 0.2 --top_k 7  --output_file TopK7_rag_predictions.csv > run_finance_SP_ToPK7.log 2 >&1
#
#python run_rag_prediction.py --dataset_name finance_SP --hybrid_alpha 0.2 --top_k 9  --output_file TopK9_rag_predictions.csv > run_finance_SP_ToPK9.log 2 >&1

python run_rag_random_prediction.py --dataset_name finance_SP --hybrid_alpha 0.2  --output_file random_rag_predictions.csv