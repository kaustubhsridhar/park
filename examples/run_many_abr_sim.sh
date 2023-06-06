mkdir ../../data/logs

env_name=abr_sim
algo=a2c
gpu=3
n_steps=5
total_timesteps=10000000
for distribution in default
do
    CUDA_VISIBLE_DEVICES=${gpu} nohup python -u run_agent_sb3.py --env_name=${env_name} --algo=${algo} --distribution=${distribution} --n_steps=${n_steps} --total_timesteps=${total_timesteps} > ../../data/logs/${algo}_sb3_${env_name}_${distribution}_${n_steps}.log &
    gpu=$((gpu+1))
done 