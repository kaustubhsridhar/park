mkdir ../../data/logs

env_name=load_balance
algo=ppo

gpu=0
for job_distribution in Pareto #Saw Uniform CyclicPos CyclicNeg DriftPos DriftNeg Constant
do
    CUDA_VISIBLE_DEVICES=${gpu} nohup python -u run_ppo_sb3.py --env_name=${env_name} --algo=${algo} --job_distribution=${job_distribution} > ../../data/logs/${algo}_sb3_${env_name}_${job_distribution}.log &
    gpu=$((gpu+1))
done 

