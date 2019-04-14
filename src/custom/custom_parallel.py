""" Add custom module functions
"""
import torch.nn as nn

class CustomDataParallel(nn.DataParallel):  
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super().__init__(module, device_ids=None, output_device=None, dim=0)

    def del_param_in_flat_arch(self, rm_name):
        # We remove an entire layer holding the delete target parameters
        rm_module = rm_name.split('.')
        module  = self._modules[rm_module[0]]
        if module._modules[rm_module[1]] != None:
            print("[INFO] Removing parameters/buffers in module [{}]".format(rm_module[0]+'.'+rm_module[1]))
            del module._modules[rm_module[1]]