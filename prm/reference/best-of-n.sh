RM=Path_of_PRM_Model
NAME=MetaMath-Mistral-7B
LANG=zh

python eval_prm.py --rm_model $RM --reference_data data/mgsm_$LANG.json --sft_output_files /data/mgsm_gen-mathmistral7b/$LANG --model_name $NAME --lang $LANG --seed 64 --mode prm --testset_name mgsm --prm_score_folder ./prm_score/ --res_folder ./res/