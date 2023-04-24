# This piece of code was copied/modified from the following source:
#
#    Title: CURL: Contrastive Unsupervised Representation Learning for Sample-Efficient Reinforcement Learning
#    Author: Laskin, Michael and Srinivas, Aravind and Abbeel, Pieter
#    Date: 2020
#    Availability: https://github.com/MishaLaskin/curl

import imageio
import os
import numpy as np


class VideoRecorder(object):
    def __init__(self, dir_name, fps=20):
        self.dir_name = dir_name
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env):
        if self.enabled:
            try:
                frame = env.render()
            except:
                print('Warning: failed to render environment, setting frame to None')
                frame = None
    
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            kargs = {'macro_block_size': None}
            try:
                imageio.mimsave(path, self.frames, fps=self.fps, **kargs)
            except:
                print('Warning: failed to save video')
