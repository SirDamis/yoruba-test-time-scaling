for i in {1..64}
do
    python infer-mgsm.py --base_model meta-math/MetaMath-Mistral-7B --lang zh --seed $i --out_path  data/mgsm_gen-mathmistral7b/zh/
    
done

