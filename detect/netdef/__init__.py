from importlib import import_module


def get_model(model_name):
    return import_module('%s.%s' % (__package__, model_name))


def get_common_config():
    config = {}
    config['anchors'] = [5., 10., 20.]  # [ 10.0, 30.0, 60.]
    config['chanel'] = 1
    config['crop_size'] = [96, 96, 96]
    config['stride'] = 4
    config['max_stride'] = 12
    config['num_neg'] = 800
    config['num_pos'] = 800
    config['th_neg'] = 0.02
    config['th_pos_train'] = 0.5
    config['th_pos_val'] = 1
    config['num_hard'] = 2
    config['bound_size'] = 12
    config['reso'] = 1
    config['sizelim'] = 2.5  # 3 #6. #mm
    config['sizelim2'] = 10  # 30
    config['sizelim3'] = 20  # 40
    config['aug_scale'] = True
    config['r_rand_crop'] = 0.3
    config['pad_value'] = 170
    config['augtype'] = {'flip': True, 'swap': False, 'scale': True, 'rotate': False}
    config['side_len'] = 72
    config['margin'] = 12
    config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38',
                           '990fbe3f0a1b53878669967b9afd1441',
                           'adc3bbc63d40f8761c59be10f1e504c3']
    return config


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
