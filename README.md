## MSFlow

### 0. Data preparation and val split
In 'Spring' data folder, there will be 2 sub-folders, **train/** and **test/** after *unzip*.

Then copy 2 sub-folders, **0045/** and **0047/** from **train/** to a new sub-folder **val/** created by yourself.

```
[In Spring folder]
mkdir val
cd train
cp -r 0045/ ../val/
cp -r 0047/ ../val/
```
Now there will be 3 sub-folders in Spring, **train/**, **test/** and **val/**.

### 1. Environment
- Python 3.11
```
conda create -n msflow python=3.11
```
- Pytorch 2.5.1 w/ cuda 11.8
```
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
```
- Other packages 1 (with conda better)
```
conda install yacs loguru imageio matplotlib tensorboard scipy h5py tqdm
```
- Other packages 2 (with pip better)
```
pip install einops timm opencv-python
```
- Xformers (only conda)
```
conda install xformers::xformers
```
- Flash_attn (pytorch 2.5.1 + cuda 11 + abi False + cp 311)
```
pip install ./flash_attn-2.7.3+cu11torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

### 2. Path for yourself
Change the right path to Spring data folder.

*dataloader/flow/spring.py: Line 27*
```
root='/home/ubuntu/dataset/zml/Spring'
```

### 3. Test the checkpoint
```
python evaluate.py --cfg config/tar-c-t-spring-1080p.json --ckpt ./tar-c-t_dino_swin_multilevel.pth --dataset spring
```
If this can be run appropriately, then all is ok.

### 4. Ready for training
First, you can ensure the entire process is ok by a toy example. Open the **config/tar-c-t-spring-1080p.json**, and change *Line 25-26* as follows:
```
    "sum_freq": 10,
    "val_freq": 20,
```
Here, **sum_freq** means the frequency of logging, **val_freq** means the frequency of validation on Spring val dataset.

Therefore, if the training process can be kept to at least 30 steps, when you get the logging [000030] on the screen, all is ok. Then just crlt+C to interrupt it. If there is some storage on the GPU caused by interruption, just kill these process as follows:
```
pkill -f multi
```
Then go into the *logs/* and inspect the train process.
```
tensorboard --logdir=./ --port=6006 --host=127.0.0.1
```
After these, remember to change the **sum_freq** and **val_freq** to 100 and 5000.
### 5. Train
```
python train.py --cfg config/tar-c-t-spring-1080p.json --validation spring_train --restore_ckpt ./tar-c-t_dino_multilevel_things30k.pth
```
If you may exit the current terminal, please use this ensure the training will not be interrupted.
```
nohup python train.py --cfg config/tar-c-t-spring-1080p.json --restore_ckpt ./tar-c-t_dino_multilevel_things30k.pth > nohup1.out &
```