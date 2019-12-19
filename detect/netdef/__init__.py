from importlib import import_module


def get_model(model_name):
    return import_module('%s.%s' % (__package__, model_name))



'''检查模型
import torch
from detect.netdef import dpn3d26
from detect.netdef import res18

model = dpn3d26.DPN92_3D()

print('我们定义的dpn模型')
print(model.state_dict().keys())
print(len(model.state_dict().keys()))  # 278


torch.save({'epoch': epoch,'save_dir': save_dir,'state_dict': state_dict,'args': args},get_save_file_name(save_dir, epoch))


path = '/Users/christyluo/Desktop/fd0066.ckpt'  #原论文模型

ckpt = torch.load(path)

print('论文跑出来的模型')
print(list(ckpt['state_dict'].keys()))
print(len(ckpt['state_dict'])) # 244个keys

sd = model.load_state_dict(ckpt['state_dict'])
print("loaded!")

torch.save(ckpt.keys,'/Users/christyluo/Desktop/log')   #重新存
'''




