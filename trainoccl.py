import sys
import os
import argparse
import numpy as np
import torch
import time
import torch.optim as optim

from config.parser import parse_args
from model import fetch_model
from dataloader.loader import fetch_dataloader
from utils.utils import load_ckpt
from utils.ddp_utils import *
from criterion.loss import sequence_loss_with_occl
from utils.logger import Logger
import validate
# import validate
# import wandb
# os.environ["WANDB_MODE"] = "offline"

os.system("export KMP_INIT_AT_FORK=FALSE")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def train(args, rank=0, world_size=1, use_ddp=False):
    """ Full training loop """
    device_id = rank
    model = fetch_model(args).to(device_id)
    # if rank == 0:
    #     avg_loss = AverageMeter()
    #     avg_epe = AverageMeter()
        # wandb.init(
        #     project=args.name
        # )
    if args.restore_ckpt is not None:
        load_ckpt(model, args.restore_ckpt)
        print(f"restore ckpt from {args.restore_ckpt}")

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], static_graph=True)
    model.train()
    train_loader = fetch_dataloader(args, rank=rank, world_size=world_size, use_ddp=use_ddp)
    optimizer, scheduler = fetch_optimizer(args, model)
    logger = Logger(model, scheduler, args)
    total_steps = 0
    epoch = 0
    cnt_overheat = 0
    should_keep_training = True
    while should_keep_training:
        # shuffle sampler
        train_loader.sampler.set_epoch(epoch)
        epoch += 1
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda(non_blocking=True) for x in data_blob]
            occl = 1 - valid
            output = model(image1, image2, flow_gt=flow, occl_gt=occl)
            loss, flow_loss, occl_loss = sequence_loss_with_occl(output, flow, valid, args.gamma)
            # logger
            metrics = {}
            if rank == 0:
                if valid.sum() > 0:
                    # avg_loss.update(loss.item())
                    epe = (((flow - output['flow'][-1])**2).sum(dim=1)).sqrt()
                    epe = (epe * valid).sum() / valid.sum()
                    # avg_epe.update(epe.item())
                    metrics['loss'] = loss.item()
                    metrics['loss_flow'] = flow_loss.item()
                    metrics['loss_occl'] = occl_loss.item()
                    metrics['epe'] = epe.item()
                logger.push(metrics)
                    
                # if total_steps % 100 == 0:
                    # print({"loss": avg_loss.avg, "epe": avg_epe.avg})
                    # avg_loss.reset()
                    # avg_epe.reset()
                    # cnt_overheat = 0


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) 
            optimizer.step()
            scheduler.step()
            if total_steps % args.val_freq == args.val_freq - 1 and rank == 0:
                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'sintel_train':
                        results.update(validate.validate_sintel(model, args, rank))
                    elif val_dataset == 'kitti_train':
                        results.update(validate.validate_kitti(model, args, rank))
                    else:
                        print(val_dataset)
                        raise NotImplementedError
                model.train()
                logger.write_dict(results)
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                os.makedirs(os.path.dirname(PATH), exist_ok=True)
                torch.save(model.module.state_dict(), PATH)
            
            if total_steps > args.num_steps:
                should_keep_training = False
                break
            
            total_steps += 1

    PATH = 'checkpoints/%s.pth' % args.name
    if rank == 0:
        torch.save(model.module.state_dict(), PATH)
        # wandb.finish()
    logger.close()
        
    return PATH

def main(rank, world_size, args, use_ddp):
    if use_ddp:
        print(f"Using DDP [{rank=} {world_size=}]")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        setup_ddp(rank, world_size)

    train(args, rank=rank, world_size=world_size, use_ddp=use_ddp)

def process_logdir(cfg):
    log_dir = 'logs/'
    now = time.localtime()
    now_time = '{:02d}_{:02d}_{:02d}_{:02d}'.format(now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
    log_dir += cfg.name + now_time
    cfg.log_dir = log_dir
    if not cfg.eval_only:
        os.makedirs(log_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--seed', help='seed', default=42, type=int)
    parser.add_argument('--restore_ckpt', help='restore checkpoint', default=None, type=str)

    parser.add_argument('--validation', default=['sintel_train'], type=str, nargs='+')
    parser.add_argument('--nosave', action='store_true', help='no logdir')
    parser.add_argument('--eval_only', action='store_true', default=False, help='eval only')
    args = parse_args(parser)
    process_logdir(args)
    smp, world_size = init_ddp()
    if world_size > 1:
        spwn_ctx = mp.spawn(main, nprocs=world_size, args=(world_size, args, True), join=False)
        spwn_ctx.join()
    else:
        main(0, 1, args, False)
    print("Done!")