
import copy
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch import autograd
from torch.distributions import Categorical
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

class PixelWiseA3C_InnerState():

    def __init__(self, model, optimizer, batch_size, t_max, gamma, beta=1e-2,
                 phi=lambda x: x,
                 pi_loss_coef=1.0, v_loss_coef=0.5,
                 average_reward_tau=1e-2,
                 act_deterministically=False,
                 average_entropy_decay=0.999,
                 average_value_decay=0.999):

        self.shared_model = model
        self.model = copy.deepcopy(self.shared_model)

        self.optimizer = optimizer
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
            target_param.detach().copy_(param.detach())
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
        if statevar is None:
            R = torch.zeros(self.batch_size, 1, 63, 63).cuda()
        else:
            _, vout, _ = self.model.pi_and_v(statevar)
            R = vout.detach()
        pi_loss = 0
        v_loss = 0

        for i in reversed(range(self.t_start, self.t)):
            R *= self.gamma
            R += self.past_rewards[i]
            v = self.past_values[i]
            advantage = R - v.detach() # (32, 3, 63, 63)
            log_prob = self.past_action_log_prob[i]
            entropy = self.past_action_entropy[i]

            pi_loss -= log_prob * advantage.detach()
            pi_loss -= self.beta * entropy
            v_loss += (v - R) ** 2 / 2.

        if self.pi_loss_coef != 1.0:
            pi_loss *= self.pi_loss_coef

        if self.v_loss_coef != 1.0:
            v_loss *= self.v_loss_coef

        print(pi_loss.mean())
        print(v_loss.mean())
        print("==========")
        total_loss = (pi_loss + v_loss).mean()

        print("loss:", total_loss) 

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
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

        if self.t - self.t_start == self.t_max:
            self.update(statevar)

        self.past_states[self.t] = statevar
        pout, vout, inner_state = self.model.pi_and_v(statevar)
        n, num_actions, h, w = pout.shape

        p_trans = pout.permute([0, 2, 3, 1]).contiguous().view(-1, pout.shape[1])
        dist = Categorical(p_trans)
        action = dist.sample()
        log_p = torch.log(torch.clamp(p_trans, min=1e-9, max=1-1e-9))
        log_action_prob = torch.gather(log_p, 1, Variable(action.unsqueeze(-1))).view(n, 1, h, w)
        entropy = -torch.sum(p_trans * log_p, dim=-1).view(n, 1, h, w)


        self.past_action_log_prob[self.t] = log_action_prob.cuda()
        self.past_action_entropy[self.t] = entropy.cuda()
        self.past_values[self.t] = vout
        self.t += 1

        return action.squeeze(1).detach().cpu(), inner_state.detach().cpu(), torch.exp(log_action_prob).squeeze(1).detach().cpu()


    def stop_episode_and_train(self, state, reward, done=False):
        self.past_rewards[self.t - 1] = torch.Tensor(reward).cuda()
        if done:
            self.update(None)
        else:
            statevar = state
            self.update(statevar)




