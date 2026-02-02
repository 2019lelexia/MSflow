import os
import sys
from model.vitwarp_v8_dino_multilevel import ViTWarpV8

def fetch_model(args):
    if args.algorithm == 'vitwarp':
        model = ViTWarpV8(args)
    else:
        raise ValueError("Unknown algorithm: {}".format(args.algorithm))
    return model