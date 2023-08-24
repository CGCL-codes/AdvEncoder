import os
import torch
from pathlib import Path
import torch.nn as nn
from model.resnet import resnet18

# victim model source zoo: https://github.com/vturrisi/solo-learn

def load_victim(args):

    if args.pre_dataset == 'cifar10':
        num_classes = 10
        victim_path = os.path.join('victims', 'cifar10', str(args.victim))
        encoder_path = [Path(victim_path) / ckpt for ckpt in os.listdir(Path(victim_path)) if ckpt.endswith(".ckpt")][0]
        model = resnet18(num_classes=num_classes)
        checkpoint = torch.load(encoder_path)
        state_dict = checkpoint['state_dict']

        new_ckpt = dict()
        for k, value in state_dict.items():
            if k.startswith('backbone'):
                new_ckpt[k.replace('backbone.', '')] = value
            elif k.startswith('classifier'):
                new_ckpt[k.replace('classifier', 'fc')] = value
        if True:
            model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                          bias=False)

        model.load_state_dict(new_ckpt)
        model.fc = nn.Identity()
        model.maxpool = nn.Identity()

    return model

