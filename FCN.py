
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import torch.optim as optim
torch.manual_seed(1)

class PPO(nn.Module):
    def __init__(self, Action_N):
        super(PPO, self).__init__()
        self.action_n = Action_N
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(2, 2), dilation=2, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(3, 3), dilation=3, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(4, 4), dilation=4, bias=True),
            nn.ReLU(),
        )
        self.diconv1_p = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(3, 3), dilation=3,
                                   bias=True)
        self.diconv2_p = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(2, 2), dilation=2,
                                   bias=True)
        self.policy = nn.Conv2d(in_channels=64, out_channels=self.action_n, kernel_size=3, stride=1, padding=(1, 1), bias=True)

        self.diconv1_v = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(3, 3), dilation=3,
                                   bias=True)
        self.diconv2_v = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(2, 2), dilation=2,
                                   bias=True)
        self.value = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=(1, 1), bias=True)

        self.conv7_Wz = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.conv7_Uz = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.conv7_Wr = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.conv7_Ur = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.conv7_W = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.conv7_U = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=False)

        self.conv.apply(self.weight_init)
        self.diconv1_p.apply(self.weight_init)
        self.diconv2_p.apply(self.weight_init)
        self.policy.apply(self.weight_init)
        self.diconv1_v.apply(self.weight_init)
        self.diconv2_v.apply(self.weight_init)
        self.value.apply(self.weight_init)
        self.conv7_Wz.apply(self.weight_init)
        self.conv7_Uz.apply(self.weight_init)
        self.conv7_Wr.apply(self.weight_init)
        self.conv7_Ur.apply(self.weight_init)
        self.conv7_W.apply(self.weight_init)
        self.conv7_U.apply(self.weight_init)

    def weight_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0, 0.01)
            m.bias.data = torch.ones(m.bias.data.size())
    
    def pi_and_v(self, x):
        conv = self.conv(x[:,0:1,:,:])
        p = self.diconv1_p(conv)
        p = F.relu(p)
        p = self.diconv2_p(p)
        p = F.relu(p)
        GRU_in = p
        ht = x[:, -64:, :, :]
        z_t = torch.sigmoid(self.conv7_Wz(GRU_in) + self.conv7_Uz(ht))
        r_t = torch.sigmoid(self.conv7_Wr(GRU_in) + self.conv7_Ur(ht))
        h_title_t = torch.tanh(self.conv7_W(GRU_in) + self.conv7_U(r_t * ht))
        h_t = (1 - z_t) * ht + z_t * h_title_t
        policy = F.softmax(self.policy(h_t), dim=1)

        v = self.diconv1_v(conv)
        v = F.relu(v)
        v = self.diconv2_v(v)
        v = F.relu(v)
        value = self.value(v)
        return policy, value, h_t
