import numpy as np
import sys
import cv2

class State():
    def __init__(self, size, move_range):
        self.image = np.zeros(size, dtype=np.float32)
        self.move_range = move_range
    
    def reset(self, x, n):
        self.image = np.clip(x + n, a_min=0., a_max=1.)
        size = self.image.shape
        prev_state = np.zeros((size[0], 64, size[2], size[3]),dtype=np.float32)
        self.tensor = np.concatenate([self.image, prev_state], axis=1)

    def set(self, x):
        self.image = x
        self.tensor[:, :self.image.shape[1], :, :] = self.image

    def step(self, act, inner_state):
        act = act.numpy()
        neutral = (self.move_range - 1)/2
        move = act.astype(np.float32)
        move = (move - neutral)/255
        moved_image = self.image + move[:,np.newaxis,:,:]

        gaussian = np.zeros(self.image.shape, self.image.dtype)
        gaussian2 = np.zeros(self.image.shape, self.image.dtype)
        bilateral = np.zeros(self.image.shape, self.image.dtype)
        bilateral2 = np.zeros(self.image.shape, self.image.dtype)
        median = np.zeros(self.image.shape, self.image.dtype)
        box = np.zeros(self.image.shape, self.image.dtype)

        b, c, h, w = self.image.shape
        for i in range(0, b): 
            if np.sum(act[i] == self.move_range) > 0:
                gaussian[i] = np.expand_dims(cv2.GaussianBlur(self.image[i].squeeze().astype(np.float32), ksize=(5, 5),
                                                              sigmaX=0.5), 0)
            if np.sum(act[i] == self.move_range + 1) > 0:
                bilateral[i] = np.expand_dims(cv2.bilateralFilter(self.image[i].squeeze().astype(np.float32), d=5,
                                                                  sigmaColor=0.1, sigmaSpace=5), 0)
            if np.sum(act[i] == self.move_range + 2) > 0:
                median[i] = np.expand_dims(cv2.medianBlur(self.image[i].squeeze().astype(np.float32), ksize=5), 0)  # 5

            if np.sum(act[i] == self.move_range + 3) > 0:
                gaussian2[i] = np.expand_dims(cv2.GaussianBlur(self.image[i].squeeze().astype(np.float32), ksize=(5, 5),
                                                               sigmaX=1.5), 0)
            if np.sum(act[i] == self.move_range + 4) > 0:
                bilateral2[i] = np.expand_dims(cv2.bilateralFilter(self.image[i].squeeze().astype(np.float32), d=5,
                                                                   sigmaColor=1.0, sigmaSpace=5), 0)
            if np.sum(act[i] == self.move_range + 5) > 0:  # 7
                box[i] = np.expand_dims(
                    cv2.boxFilter(self.image[i].squeeze().astype(np.float32), ddepth=-1, ksize=(5, 5)), 0)
        self.image = moved_image
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range, gaussian, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+1, bilateral, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+2, median, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+3, gaussian2, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+4, bilateral2, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+5, box, self.image)

        self.image = np.clip(self.image, a_min=0., a_max=1.)
        self.tensor[:,:self.image.shape[1],:,:] = self.image
        self.tensor[:,-64:,:,:] = inner_state
