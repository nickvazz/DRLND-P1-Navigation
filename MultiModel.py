import os
import numpy as np


counter = 1
for gamma in np.arange(0.96, 1.00, 0.01):
	for update_every in np.arange(6, 0, -1):
		for num_units in reversed([8, 16, 32, 64]):
			
			os.system(f'python3 ./training.py \
				--n_episodes 2000 \
				--max_t 2000 \
				--model_num {counter} \
				--num_units {num_units} \
				--GAMMA {gamma} \
				--UPDATE_EVERY {update_every}')
			counter += 1
			