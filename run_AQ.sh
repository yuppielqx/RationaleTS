#python run_rag_prediction.py --dataset_name LargeAQ --hybrid_alpha 0.9 --output_file rag_predictions_9.csv > run_LargeAQ4.log 2>&1
#
#python run_rag_prediction.py --dataset_name LargeAQ --hybrid_alpha 0.8 --output_file rag_predictions_8.csv > run_LargeAQ5.log 2>&1
#
#python run_rag_prediction.py --dataset_name LargeAQ --hybrid_alpha 0.7 --output_file rag_predictions_7.csv > run_LargeAQ6.log 2>&1
#
#python run_rag_prediction.py --dataset_name LargeAQ --hybrid_alpha 0.6 --output_file rag_predictions_6.csv > run_LargeAQ7.log 2>&1
#
#python run_rag_prediction.py --dataset_name LargeAQ --hybrid_alpha 0.4 --output_file rag_predictions_4.csv > run_LargeAQ8.log 2>&1
#
#python run_rag_prediction.py --dataset_name LargeAQ --hybrid_alpha 0.3 --output_file rag_predictions_3.csv > run_LargeAQ9.log 2>&1
#
#python run_rag_prediction.py --dataset_name LargeAQ --hybrid_alpha 0.2 --output_file rag_predictions_2.csv > run_LargeAQ10.log 2>&1
#
#python run_rag_prediction.py --dataset_name LargeAQ --hybrid_alpha 0.1 --output_file rag_predictions_1.csv > run_LargeAQ11.log 2>&1

#python run_rag_prediction.py --dataset_name LargeAQ --hybrid_alpha 0.8 --analysis_model gemini-2.0-flash --prediction_model gemini-2.0-flash --output_file gemini-2.0-flash_rag_predictions.csv

python run_rag_prediction.py --dataset_name finance_SP --hybrid_alpha 0.2 --top_k 1  --output_file TopK1_rag_predictions.csv > run_finance_SP_ToPK1.log 2>&1

python run_rag_prediction.py --dataset_name finance_SP --hybrid_alpha 0.2 --top_k 3  --output_file TopK3_rag_predictions.csv > run_finance_SP_ToPK3.log 2>&1

python run_rag_prediction.py --dataset_name finance_SP --hybrid_alpha 0.2 --top_k 7  --output_file TopK7_rag_predictions.csv > run_finance_SP_ToPK7.log 2>&1

python run_rag_prediction.py --dataset_name finance_SP --hybrid_alpha 0.2 --top_k 9  --output_file TopK9_rag_predictions.csv > run_finance_SP_ToPK9.log 2>&1