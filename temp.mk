# https://github.com/Unity-Technologies/ml-agents/issues/305
# https://support.unity3d.com/hc/en-us/articles/115003118426-Running-multiple-instances-of-Unity-referencing-the-same-project

SELF := $(abspath $(lastword $(MAKEFILE_LIST)))
HERE := $(dir $(SELF))

model_nums := 200 201 202

all: \
    $(model_nums)


$(model_nums):
		python $(HERE)training.py --n_episodes 3 --max_t 20 --model_num $@
