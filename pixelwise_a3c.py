
import copy
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch import autograd
from torch.distributions import Categorical
from RL_model.reward_dis import *
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

class PixelWiseA3C_InnerState():

    def __init__(self, model, optimizer, optimizerD, LAMBDA, G, D, batch_size, t_max, gamma, beta=1e-2,
                 phi=lambda x: x,
                 pi_loss_coef=1.0, v_loss_coef=0.5,
                 average_reward_tau=1e-2,
                 act_deterministically=False,
                 average_entropy_decay=0.999,
                 average_value_decay=0.999):

        self.shared_model = model
        self.model = copy.deepcopy(self.shared_model)

        self.optimizer = optimizer
        self.optimizerD = optimizerD
        self.lambdas = LAMBDA
        self.dis_reward = G
        self.target_netD = D
        self.batch_size = batch_size

        self.t_max = t_max
        self.gamma = gamma
        self.beta = beta
        self.phi = phi
        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef
        self.average_reward_tau = average_reward_tau
        self.act_deterministically = act_deterministically
        self.average_value_decay = average_value_decay
        self.average_entropy_decay = average_entropy_decay
        # self.batch_states = batch_states

        self.t = 0
        self.t_start = 0
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}
        self.average_reward = 0

        self.explorer = None

        # Stats
        self.average_value = 0
        self.average_entropy = 0

    """
    异步更新参数
    """
    def sync_parameters(self):
        for m1, m2 in zip(self.model.modules(), self.shared_model.modules()):
            m1._buffers = m2._buffers.copy()
        for target_param, param in zip(self.model.parameters(), self.shared_model.parameters()):
            target_param.data.copy_(param.data)
    """
    异步更新梯度
    """
    def update_grad(self, target, source):
        target_params = dict(target.named_parameters())
        # print(target_params)
        for param_name, param in source.named_parameters():
            if target_params[param_name].grad is None:
                if param.grad is None:
                    pass
                else:
                    target_params[param_name].grad = param.grad
            else:
                if param.grad is None:
                    target_params[param_name].grad = None
                else:
                    target_params[param_name].grad[...] = param.grad

    def update(self, statevar):
        assert self.t_start < self.t
        # 如果为最开始，那么状态为none，奖励为0
        if statevar is None:
            R = torch.zeros(22, 1, 63, 63).cuda()
        else:
            # 否则用价值来计算累计奖励和
            _, vout, _ = self.model.pi_and_v(statevar)
            R = vout.data
        pi_loss = 0
        v_loss = 0
        # 倒序遍历时间步
        for i in reversed(range(self.t_start, self.t)):
            R *= self.gamma
            R += self.past_rewards[i]
            v = self.past_values[i]
            advantage = R - v.detach() # (32, 3, 63, 63)
            log_prob = self.past_action_log_prob[i]
            entropy = self.past_action_entropy[i]

            pi_loss -= log_prob * advantage.data
            # Entropy is maximized
            pi_loss -= self.beta * entropy
            # Accumulate gradients of value function
            v_loss += (v - R) ** 2 / 2

        if self.pi_loss_coef != 1.0:
            pi_loss *= self.pi_loss_coef

        if self.v_loss_coef != 1.0:
            v_loss *= self.v_loss_coef
        print(pi_loss.mean())
        print(v_loss.mean())
        print("==========")
        total_loss = (pi_loss + v_loss).mean()

        print("loss:", total_loss) # 5
        # self.update_dis(s_next[:, 0:3, :, :].cuda(), real.cuda(), rotlabel)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # self.shared_model.zero_grad()
        self.update_grad(self.shared_model, self.model)



        self.sync_parameters()

        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}

        self.t_start = self.t

    def act_and_train(self, state, reward):
        statevar = torch.Tensor(state).cuda()
        self.past_rewards[self.t - 1] = torch.Tensor(reward).cuda()
##########################
        # t是不断增长得，其中self.t_max = 5
        # 如果够5步或定义得步数后执行更新
        if self.t - self.t_start == self.t_max:
            self.update(statevar)

        self.past_states[self.t] = statevar
        # label
        # with torch.no_grad():
        pout, vout, inner_state = self.model.pi_and_v(statevar)
        p_trans = pout.permute([0,2,3,1])
        dist = Categorical(p_trans)
        action = dist.sample().data # 动作
        log_p = torch.log(pout) # 对数概率
        action_prob = pout.gather(1, action.unsqueeze(1))
            # with torch.no_grad():
        entropy = torch.stack([- torch.sum(log_p * pout, dim=1)]).permute([1,0,2,3])

        self.past_action_log_prob[self.t] = torch.log(action_prob).cuda() # 对数概率
        # F.stack([- F.sum(self.all_prob * self.all_log_prob, axis=1)], axis=1)
        self.past_action_entropy[self.t] = entropy.cuda()# 动作熵
        self.past_values[self.t] = vout
        self.t += 1

        return action.squeeze(1).data.cpu(), inner_state.data.cpu(), action_prob.squeeze(1).data.cpu()


    def stop_episode_and_train(self, state, reward, done=False):
        self.past_rewards[self.t - 1] = torch.Tensor(reward).cuda()
        if done:
            self.update(None)
        else:
            #statevar = self.batch_states([state], np, self.phi)
            statevar = state
            self.update(statevar)
        # torch.cuda.empty_cache()

    # WGAN Loss
    def cal_gradient_penalty(self, netD, real_data, fake_data, batch_size):
        alpha = torch.rand(batch_size, 1)
        # int(real_data.nelement() / batch_size) real_data 中tensor元素得个数
        # print(real_data.size()) # 80, 6, 63, 63
        # print(real_data.nelement()) # 1905120=80*6*63*63
        # print(int(real_data.nelement() / batch_size)) # 119070=63*63* 6*5
        # print(batch_size) # 16
        # print("********")
        alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
        # print(alpha.size())
        # print("$$$$$$$$")
        alpha = alpha.view(batch_size, 6, 63, 63)  # (16, 1, 63, 63)
        alpha = alpha.to(device)

        fake_data = fake_data.view(batch_size, 6, 63, 63)  # (16, 1, 63, 63)
        interpolates = Variable(alpha * real_data.data + ((1 - alpha) * fake_data.data), requires_grad=True)
        disc_interpolates, _ = netD(interpolates)
        gradients = autograd.grad(disc_interpolates, interpolates,
                                      grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                      create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambdas
        return gradient_penalty

    def update_dis(self, fake_data, real_data, Rlabel):
        # 下一状态状态
        fake_data = fake_data.detach()
        # 无噪声图片
        real_data = real_data.detach()

        fake = torch.cat([real_data, fake_data], 1)
        real = torch.cat([real_data, real_data], 1)

        D_real, rot_real = self.dis_reward(real)
        D_fake, rot_fake = self.dis_reward(fake)

        rot_labels = torch.LongTensor(Rlabel.to(torch.int64).unsqueeze(1))
        Rot_labels = torch.zeros(self.t_max * self.batch_size, 4).scatter_(1, rot_labels, 1).cuda()
        d_real_class_loss = torch.sum(F.binary_cross_entropy_with_logits(
                input=rot_real,
                target=Rot_labels))
        gradient_penalty = self.cal_gradient_penalty(self.dis_reward, real, fake, real.size(0))
        self.optimizerD.zero_grad()
        D_cost = D_fake.mean() - D_real.mean() + gradient_penalty + d_real_class_loss
        print("Cost:", D_cost)
        print("classLoss:", d_real_class_loss)
        D_cost.backward()
        # nn.utils.clip_grad_norm_(dis_reward.parameters(), 50)
        self.optimizerD.step()
        soft_update(self.target_netD, self.dis_reward, 0.001)

