#
#python run_rag_prediction.py --dataset_name traffic --hybrid_alpha 0.9 --output_file rag_predictions_9.csv > run_traffic4.log 2>&1
#
#python run_rag_prediction.py --dataset_name traffic --hybrid_alpha 0.8 --output_file rag_predictions_8.csv > run_traffic5.log 2>&1
#
#python run_rag_prediction.py --dataset_name traffic --hybrid_alpha 0.7 --output_file rag_predictions_7.csv > run_traffic6.log 2>&1
#
#python run_rag_prediction.py --dataset_name traffic --hybrid_alpha 0.6 --output_file rag_predictions_6.csv > run_traffic7.log 2>&1
#
#python run_rag_prediction.py --dataset_name traffic --hybrid_alpha 0.4 --output_file rag_predictions_4.csv > run_traffic8.log 2>&1
#
#python run_rag_prediction.py --dataset_name traffic --hybrid_alpha 0.3 --output_file rag_predictions_3.csv > run_traffic9.log 2>&1
#
#python run_rag_prediction.py --dataset_name traffic --hybrid_alpha 0.2 --output_file rag_predictions_2.csv > run_traffic10.log 2>&1
#
#python run_rag_prediction.py --dataset_name traffic --hybrid_alpha 0.1 --output_file rag_predictions_1.csv > run_traffic11.log 2>&1

#python run_rag_prediction.py --dataset_name traffic --hybrid_alpha 0.8 --analysis_model gemini-2.0-flash --prediction_model gemini-2.0-flash --output_file gemini-2.0-flash_rag_predictions.csv

#python run_rag_prediction.py --dataset_name traffic --hybrid_alpha 0.8 --top_k 1  --output_file TopK1_rag_predictions.csv > run_traffic_ToPK1.log 2>&1
#
#python run_rag_prediction.py --dataset_name traffic --hybrid_alpha 0.8 --top_k 3  --output_file TopK3_rag_predictions.csv > run_traffic_ToPK3.log 2>&1
#
#python run_rag_prediction.py --dataset_name traffic --hybrid_alpha 0.8 --top_k 7  --output_file TopK7_rag_predictions.csv > run_traffic_ToPK7.log 2>&1
#
#python run_rag_prediction.py --dataset_name traffic --hybrid_alpha 0.8 --top_k 9  --output_file TopK9_rag_predictions.csv > run_traffic_ToPK9.log 2>&1

python run_rag_random_prediction.py --dataset_name traffic --hybrid_alpha 0.8  --output_file random_rag_predictions.csv