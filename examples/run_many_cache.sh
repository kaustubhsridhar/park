mkdir ../../data/logs

env_name=cache
algo=a2c
gpu=1
for n_steps in 50 1000 10000 100000
do
    total_timesteps=50000000
    for distribution in default
    do
        CUDA_VISIBLE_DEVICES=${gpu} nohup python -u run_agent_sb3.py --env_name=${env_name} --algo=${algo} --distribution=${distribution} --n_steps=${n_steps} --total_timesteps=${total_timesteps} > ../../data/logs/${algo}_sb3_${env_name}_${distribution}_${n_steps}.log &
        gpu=$((gpu+1))
    done 
done 