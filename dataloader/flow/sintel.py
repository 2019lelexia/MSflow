import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
import h5py
from tqdm import tqdm
from glob import glob
import os.path as osp

from utils import frame_utils
from dataloader.template import FlowDataset

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/home/ubuntu/dataset/zml/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

    def read_flow(self, index):
        flow = frame_utils.read_gen(self.flow_list[index])
        valid = (np.abs(flow[..., 0]) < 1000) & (np.abs(flow[..., 1]) < 1000)
        return flow, valid

# import numpy as np
# import torch
# import torch.utils.data as data
# import torch.nn.functional as F

# import os
# import math
# import random
# import h5py
# from tqdm import tqdm
# from glob import glob
# import os.path as osp
# import pickle

# from utils import frame_utils
# from dataloader.template import FlowDataset

# class MpiSintel(FlowDataset):
#     def __init__(self, aug_params=None, split='training', input_frames=2, root='/home/ubuntu/dataset/zml/Sintel', dstype='clean'):
#         super(MpiSintel, self).__init__(aug_params)
#         flow_root = osp.join(root, split, 'flow')
#         image_root = osp.join(root, split, dstype)

#         if split == 'test':
#             self.is_test = True

#         with open("./flow_dataset_mf/sintel_training_" + dstype + "_png.pkl", "rb") as f:
#             _image_list = pickle.load(f)
#         with open("./flow_dataset_mf/sintel_training_" + dstype + "_flo.pkl", "rb") as f:
#             _future_flow_list = pickle.load(f)
#         with open("./flow_dataset_mf/sintel_training_scene.pkl", "rb") as f:
#             _extra_info_list = pickle.load(f)
#         len_list = len(_image_list)
        

#         for index in range(len_list):
#             _images = _image_list[index]
#             _future_flows = _future_flow_list[index]
#             len_image = len(_images)
#             for idx in range(len_image):
#                 _images[idx] = root + _images[idx].strip()
#             for idx in range(len_image - 1):
#                 _future_flows[idx] = root + _future_flows[idx].strip()
#             for idx_image in range(0, len_image - input_frames + 1):
#                 self.image_list.append(_images[idx_image: idx_image + input_frames])
#                 self.extra_info.append(_extra_info_list[index])
#                 self.flow_list.append(_future_flows[idx_image])
#         # for scene in os.listdir(image_root):
#         #     image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
#         #     for i in range(len(image_list)-1):
#         #         self.image_list += [ [image_list[i], image_list[i+1]] ]
#         #         self.extra_info += [ (scene, i) ] # scene and frame_id

#         #     if split != 'test':
#         #         self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

#     def read_flow(self, index):
#         flow = frame_utils.read_gen(self.flow_list[index])
#         valid = (np.abs(flow[..., 0]) < 1000) & (np.abs(flow[..., 1]) < 1000)
#         return flow, valid

# if __name__ == '__main__':
#     dataset = MpiSintel()
#     print(len(dataset))