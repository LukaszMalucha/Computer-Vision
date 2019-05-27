# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:02:40 2019

@author: MaximusMinimus
"""

import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio