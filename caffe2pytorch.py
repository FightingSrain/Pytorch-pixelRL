from chainer.links.caffe import CaffeFunction
from FCN import PPO
import torch
import numpy as np
net = CaffeFunction('./initial_weight/zhang_cvpr17_denoise_50_gray.caffemodel')
print(net.layer1.W.data.shape)
model = PPO(9, 1)
model_dict = model.state_dict()
print(model_dict['conv.0.weight'].size())
print(model_dict.keys())
model_dict['conv.0.weight'] = torch.FloatTensor(net.layer1.W.data)
model_dict['conv.0.bias'] = torch.FloatTensor(net.layer1.b.data)
model_dict['conv.2.weight'] = torch.FloatTensor(net.layer3.W.data)
model_dict['conv.2.bias'] = torch.FloatTensor(net.layer3.b.data)
model_dict['conv.4.weight'] = torch.FloatTensor(net.layer6.W.data)
model_dict['conv.4.bias'] = torch.FloatTensor(net.layer6.b.data)
model_dict['conv.6.weight'] = torch.FloatTensor(net.layer9.W.data)
model_dict['conv.6.bias'] = torch.FloatTensor(net.layer9.b.data)

model_dict['diconv1_p.weight'] = torch.FloatTensor(net.layer12.W.data)
model_dict['diconv1_p.bias'] = torch.FloatTensor(net.layer12.b.data)
model_dict['diconv2_p.weight'] = torch.FloatTensor(net.layer15.W.data)
model_dict['diconv2_p.bias'] = torch.FloatTensor(net.layer15.b.data)

model_dict['diconv1_v.weight'] = torch.FloatTensor(net.layer12.W.data)
model_dict['diconv1_v.bias'] = torch.FloatTensor(net.layer12.b.data)
model_dict['diconv2_v.weight'] = torch.FloatTensor(net.layer15.W.data)
model_dict['diconv2_v.bias'] = torch.FloatTensor(net.layer15.b.data)
model.load_state_dict(model_dict)
torch.save(model.state_dict(), "./torch_initweight/sig50_gray.pth")



