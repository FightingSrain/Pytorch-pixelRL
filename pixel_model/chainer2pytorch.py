
import numpy as np
from RL_model.Attention_FCN import PPO
import torch


d = np.load('./pretrained_50.npz')
print(d.files)
print(d['diconv6_pi/diconv/W'].shape)
model = PPO(9, 1)
model_dict = model.state_dict()

model_dict['conv.0.weight'] = torch.FloatTensor(d['conv1/W'])
model_dict['conv.0.bias'] = torch.FloatTensor(d['conv1/b'])
model_dict['conv.2.weight'] = torch.FloatTensor(d['diconv2/diconv/W'])
model_dict['conv.2.bias'] = torch.FloatTensor(d['diconv2/diconv/b'])
model_dict['conv.4.weight'] = torch.FloatTensor(d['diconv3/diconv/W'])
model_dict['conv.4.bias'] = torch.FloatTensor(d['diconv3/diconv/b'])
model_dict['conv.6.weight'] = torch.FloatTensor(d['diconv4/diconv/W'])
model_dict['conv.6.bias'] = torch.FloatTensor(d['diconv4/diconv/b'])

model_dict['diconv1_p.weight'] = torch.FloatTensor(d['diconv5_pi/diconv/W'])
model_dict['diconv1_p.bias'] = torch.FloatTensor(d['diconv5_pi/diconv/b'])
model_dict['diconv2_p.weight'] = torch.FloatTensor(d['diconv6_pi/diconv/W'])
model_dict['diconv2_p.bias'] = torch.FloatTensor(d['diconv6_pi/diconv/b'])
model_dict['policy.weight'] = torch.FloatTensor(d['conv8_pi/model/W'])
model_dict['policy.bias'] = torch.FloatTensor(d['conv8_pi/model/b'])

model_dict['diconv1_v.weight'] = torch.FloatTensor(d['diconv5_V/diconv/W'])
model_dict['diconv1_v.bias'] = torch.FloatTensor(d['diconv5_V/diconv/b'])
model_dict['diconv2_v.weight'] = torch.FloatTensor(d['diconv6_V/diconv/W'])
model_dict['diconv2_v.bias'] = torch.FloatTensor(d['diconv6_V/diconv/b'])
model_dict['value.weight'] = torch.FloatTensor(d['conv7_V/W'])
model_dict['value.bias'] = torch.FloatTensor(d['conv7_V/b'])

model_dict['conv7_Wz.weight'] = torch.FloatTensor(d['conv7_Wz/W'])
model_dict['conv7_Uz.weight'] = torch.FloatTensor(d['conv7_Uz/W'])
model_dict['conv7_Wr.weight'] = torch.FloatTensor(d['conv7_Wr/W'])
model_dict['conv7_Ur.weight'] = torch.FloatTensor(d['conv7_Ur/W'])
model_dict['conv7_W.weight'] = torch.FloatTensor(d['conv7_W/W'])
model_dict['conv7_U.weight'] = torch.FloatTensor(d['conv7_U/W'])

model.load_state_dict(model_dict)
torch.save(model.state_dict(), "pixel_sig50_gray.pth")







