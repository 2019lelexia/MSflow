import re
import os.path as osp
from glob import glob
import pickle

root = "/media/zml/Elements/FlyingThings3D/"

for dstype in ['frames_cleanpass', 'frames_finalpass']:
    image_list = []
    fflow_list = []
    pflow_list = []

    for cam in ['left']:
        image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
        image_dirs = sorted([osp.join(f, cam) for f in image_dirs])
    
    flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
    flow_future_dirs = sorted([osp.join(f, 'into_future', cam) for f in flow_dirs])
    flow_past_dirs = sorted([osp.join(f, 'into_past', cam) for f in flow_dirs])

    for idir, fdir, pdir in zip(image_dirs, flow_future_dirs, flow_past_dirs):
        images = sorted(glob(osp.join(idir, "*.png")))
        fflows = sorted(glob(osp.join(fdir, "*.pfm")))
        pflows = sorted(glob(osp.join(pdir, "*.pfm")))

        for index in range(len(images)):
            images[index] = images[index].replace(root, "") + "\n"
        for index in range(len(fflows)):
            fflows[index] = fflows[index].replace(root, "") + "\n"
        for index in range(len(pflows)):
            pflows[index] = pflows[index].replace(root, "") + "\n"
        
        image_list.append(images)
        fflow_list.append(fflows)
        pflow_list.append(pflows)

    
    with open("FlyingThings_" + dstype + "_png.pkl", "wb") as f:
        pickle.dump(image_list, f)
    with open("FlyingThings_" + dstype + "_future_pfm.pkl", "wb") as f:
        pickle.dump(fflow_list, f)
    with open("FlyingThings_" + dstype + "_past_pfm.pkl", "wb") as f:
        pickle.dump(pflow_list, f)

