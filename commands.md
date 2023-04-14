# Commands 

Create conda environment from YAML file and activate it:
`conda env create -f conda_env.yml`
`conda activate curla`

Save minimal conda environment to YAML file:
`conda env export --from-history > conda_env.yml`

Install PyTorch 1.13.1 with CUDA 11.7:
`conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia`
`pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117`

Activate tensorboard logging
`tensorboard --logdir tmp --port 6006`

Test command:
```
python train.py ^
    --carla_town Town04 ^
    --max_npc_vehicles 75 ^
    --num_eval_episodes 1 ^
    --init_steps 1 ^
    --encoder_type pixel ^
    --save_tb ^
    --save_video ^
    --pre_transform_image_height 90 ^
    --pre_transform_image_width 160 ^
    --image_height 76 ^
    --image_width 135 ^
    --replay_buffer_capacity 100000 ^
    --work_dir .\tmp ^
    --agent curl_sac ^
    --frame_stack 3 ^
    --seed -1 ^
    --eval_freq 100 ^
    --batch_size 128 ^
    --log_interval 10 ^
    --num_train_steps 200
```   

Test command 2
```
python train.py ^
    --num_eval_episodes 5 ^
    --init_steps 100 ^
    --eval_freq 1000 ^
    --num_train_steps 5000 ^
    --load_model True
```

Train command
``` 
python train.py ^
    --carla_town Town04 ^
    --max_npc_vehicles 75 ^
    --num_eval_episodes 10 ^
    --init_steps 1000 ^
    --encoder_type pixel ^
    --save_tb ^
    --save_video ^
    --pre_transform_image_height 90 ^
    --pre_transform_image_width 160 ^
    --image_height 76 ^
    --image_width 135 ^
    --replay_buffer_capacity 100_000 ^
    --work_dir .\tmp ^
    --agent curl_sac ^
    --frame_stack 3 ^
    --seed -1 ^
    --eval_freq 2_000 ^
    --batch_size 128 ^
    --num_train_steps 10_000
``` 