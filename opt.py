import nni
from nni.experiment import Experiment
import os
cuda_id = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
print('cuda_id:', cuda_id)
search_space = {
    "base_weight": {"_type": "uniform", "_value": [0.0, 1.0]},  # 0.6
    "un_weight": {"_type": "uniform", "_value": [0.0, 1.0]},  # 0.5
    "norm_weight": {"_type": "uniform", "_value": [0.0, 1.0]},  # 0.5
    "temperature": {"_type": "uniform", "_value": [0.0, 1.0]},
    "con_weight": {"_type": "uniform", "_value": [0.0, 0.1]},
    "margin": {"_type": "uniform", "_value": [0.0, 1.0]},
}

# 下面是一些NNI的设置
experiment = Experiment('local')
# 这里把之前的训练命令行写过来，同时可以把一些需要的但不是超参的argument加上，如数据集
mode = 'similarity'
norm = 'kl'
experiment.config.trial_command = f'python temp.py --sup_mode {mode} --nni --norm_term {norm}'

# 选择代码的目录，这里同目录就是一个.
experiment.config.trial_code_directory = '.'
# nni工作时log放哪里
experiment.config.experiment_working_directory = './experiments'
# 使用刚刚的搜索空间
experiment.config.search_space = search_space
# 搜索模式
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
# 做几次实验？
experiment.config.max_trial_number = 100
# 并行数
experiment.config.trial_concurrency = 1
# 一次最多跑多久？
experiment.config.max_trial_duration = '48h'
# 把刚刚的port拿来启动NNI
experiment.run(8089)