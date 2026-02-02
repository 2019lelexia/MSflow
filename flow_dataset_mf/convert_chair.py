import re
import os.path as osp
from glob import glob
import os

import pickle

root = "/home/ubuntu/dataset/zml/Hd1k/"

image_list = []
flow_list = []

seq_ix = 0
while 1:
    flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
    images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))
    if len(flows) == 0:
        break
    print(seq_ix, len(flows), images[0], images[-1], '!!!!!')
    for index in range(len(images)):
        images[index] = images[index].replace('/media/zml/Elements/Hd1k', 'Hd1k')
    for index in range(len(flows)):
        flows[index] = flows[index].replace('/media/zml/Elements/Hd1k', 'Hd1k')
    seq_ix += 1
    image_list.append(images)
    flow_list.append(flows)

with open('hd1k_png.pkl', 'wb') as f:
    pickle.dump(image_list, f)
with open('hd1k_flo.pkl', 'wb') as f:
    pickle.dump(flow_list, f)