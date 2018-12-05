import random
import numpy as np
from tensorflow.python.client import device_lib
from word sequence import WordSequence
'''获取当前GPU信息'''
def __get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type =='GPU']

if __name__ == '_main_':
    print(__get_available_gpus())