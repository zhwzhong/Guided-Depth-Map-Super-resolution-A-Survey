# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   get_gpu_info.py
@Time    :   2020/1/14 16:08
@Desc    :
"""
import os
import numpy as np
def parse(line,qargs):
    '''
    line:
        a line of text
    qargs:
        query arguments
    return:
        a dict of gpu infos
    Pasing a line of csv format text returned by nvidia-smi
    解析一行nvidia-smi返回的csv格式文本
    '''
    numberic_args = ['memory.free', 'memory.total', 'power.draw', 'power.limit']#可计数的参数
    power_manage_enable=lambda v:(not 'Not Support' in v)#lambda表达式，显卡是否滋瓷power management（笔记本可能不滋瓷）
    to_numberic=lambda v:float(v.upper().strip().replace('MIB','').replace('W',''))#带单位字符串去掉单位
    process = lambda k,v:((int(to_numberic(v)) if power_manage_enable(v) else 1) if k in numberic_args else v.strip())
    return {k:process(k,v) for k,v in zip(qargs,line.strip().split(','))}

def query_gpu(qargs=[]):
    '''
    qargs:
        query arguments
    return:
        a list of dict
    Querying GPUs infos
    查询GPU信息
    '''
    qargs = ['index', 'gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit'] + qargs
    cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
    results = os.popen(cmd).readlines()
    return [parse(line, qargs) for line in results]


def get_memory(num_gpu):
    gpu_memory = query_gpu()
    use_memory = []
    sum_gpus = 0
    for gpu in gpu_memory:
        used_memory = gpu['memory.total'] - gpu['memory.free']
        if used_memory < 20000:
            sum_gpus += 1
        use_memory.append(used_memory)
        # print(used_memory)
        # if used_memory > 1000:
        #     while True:
        #         i = 10
    # print('Num Of GPUs {}, Min Used: {}, Cuda: {}'.format(len(gpu_memory), np.min(use_memory), np.argmin(use_memory)))
    np.argsort(use_memory)
    if sum_gpus < num_gpu: return False
    return map(lambda x: str(x), np.argsort(use_memory)[: num_gpu])
    # return len(gpu_memory), np.argmin(use_memory), np.min(use_memory)

# print(get_memory(num_gpu=1))

# a = [3,4,1,7,2]
#
#
# print(np.argsort(a))