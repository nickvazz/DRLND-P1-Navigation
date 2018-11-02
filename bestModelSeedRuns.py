import os
import numpy as np
import glob
import json

best_models = [119, 118, 95, 138, 134]

for model in best_models:
    params = glob.glob(f'results-noPixels/model-{model}/*.json')[0]
    with open(params) as f:
        data = json.load(f)
    
        for key, value in data.items():
            exec(f'{key} = {value}')
    
        for idx, seed in enumerate([10,20,30,40,50]):
            model_num = f'{model}{seed}'
            if idx == 4:
                os.system(f'python3 ./training.py \
                         --n_episodes 2000 \
                         --max_t 2000 \
                         --model_num {model_num} \
                         --num_units {num_units} \
                         --GAMMA {GAMMA} \
                         --UPDATE_EVERY {UPDATE_EVERY} \
                         --seed {seed}')
            else:
                os.system(f'nohup python3 ./training.py \
                         --n_episodes 2000 \
                         --max_t 2000 \
                         --model_num {model_num} \
                         --num_units {num_units} \
                         --GAMMA {GAMMA} \
                         --UPDATE_EVERY {UPDATE_EVERY} \
                         --seed {seed} &')