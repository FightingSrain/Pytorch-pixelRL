
import torch
import numpy as np
import cv2
from FCN import PPO
import State as State
import matplotlib.pyplot as plt
from torch.distributions import Categorical
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = PPO(9).to(device)
model.load_state_dict(torch.load('./torch_pixel_model/pixel_sig25_gray.pth'))
simga = 25
def tst(model):
    model.eval()
    img_path = "./test002.png"
    raw_x = cv2.imread(img_path).astype(np.float32)
    raw_x = cv2.cvtColor(raw_x, cv2.COLOR_RGB2GRAY)
    raw_n = np.random.normal(0, simga, raw_x.shape).astype(raw_x.dtype)/255
    raw_x = np.expand_dims(raw_x, 0)
    raw_x = np.array([raw_x]) / 255
    step_test = State((raw_x.shape[0], 1, raw_x.shape[2], raw_x.shape[3]), move_range=3)
    s = np.clip(raw_x + raw_n, a_max=1., a_min=0.)
    ht = np.zeros([s.shape[0], 64, s.shape[2], s.shape[3]], dtype=np.float32)
    st = np.concatenate([s, ht], axis=1)

    image = np.asanyarray(raw_x[0, 0:1, :, :].transpose(1, 2, 0) * 255, dtype=np.uint8)
    image = np.squeeze(image)
    cv2.imshow("rerr", image)
    cv2.waitKey(0)

    image = np.asanyarray(st[0, 0:1, :, :].transpose(1, 2, 0) * 255, dtype=np.uint8)
    image = np.squeeze(image)
    cv2.imshow("rerr", image)
    cv2.waitKey(0)
    for t in range(5):
        action_map, action_map_prob, ht_ = select_action(torch.FloatTensor(st).to(device), test=True)  # 1, 1, 63, 63
        step_test.set(st, 1)
        paint_amap(action_map[0])
        print(action_map[0])
        print(action_map_prob[0])
        st = step_test.steps(action_map, ht_, st, 1)
        image = np.asanyarray(st[0, 0:1, :, :].transpose(1, 2, 0) * 255, dtype=np.uint8)
        image = np.squeeze(image)
        cv2.imshow("rerr", image)
        cv2.waitKey(0)

def select_action(state, test=False):
    with torch.no_grad():
        pout, _, ht_, a = model(state, 1)
    pout = torch.clamp(pout, min=0, max=1)
    p_trans = pout.permute([0, 2, 3, 1])
    dist = Categorical(p_trans)
    if test:
        _, action = torch.max(pout, dim=1)
    else:
        action = dist.sample().detach()  # 动作

    action_prob = pout.gather(1, action.unsqueeze(1))
    return action.unsqueeze(1).detach().cpu().numpy(), action_prob.detach().cpu().numpy(), ht_.detach().cpu().numpy()
def paint_amap(acmap):
    image = np.asanyarray(acmap.squeeze(), dtype=np.uint8)
    plt.imshow(image,  vmin=1, vmax=9)
    plt.colorbar()
    plt.show()
tst(model)













