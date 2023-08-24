import os
import torch
import argparse
import time
import random
import json
import numpy as np
from utils.load_model import load_victim
from utils.load_data import load_data, normalzie
from utils.predict import accuracy, test, make_print_to_file
from model.linear import NonLinearClassifier


def arg_parse():
    parser = argparse.ArgumentParser(description='Train downstream models using of the pre-trained encoder')
    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--gpu', default='1', type=str, help='which gpu the code runs on')
    parser.add_argument('--dataset', default='stl10', choices=['cifar10', 'stl10', 'gtsrb', 'imagenet'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--pre_dataset', default='cifar10', choices=['cifar10', 'imagenet'])
    parser.add_argument('--victim', default='simclr', choices=['simclr', 'byol', 'barlow_twins', 'deepclusterv2', 'dino', 'mocov3', 'mocov2plus', 'nnclr', 'ressl', 'simsiam', 'supcon', 'swav', 'bivcreg', 'vicreg', 'wmse'])
    args = parser.parse_args()
    return args

def classify(args, encoder):
    data = args.dataset
    train_loader, test_loader = load_data(data, args.batch_size)

    # save uap result
    uap_save_path = os.path.join('victims', str(args.pre_dataset), str(args.victim), 'clean_model', str(args.dataset))

    if not os.path.exists(uap_save_path):
        os.makedirs(uap_save_path)

    # downstream task
    if args.dataset == 'imagenet':
        num_classes = 100
        args.epochs = 50
    elif args.dataset == 'gtsrb':
        num_classes = 43
    else:
        num_classes = 10

    F = NonLinearClassifier(feat_dim=512, num_classes=num_classes)
    F.cuda()
    encoder.cuda()

    # classifier
    my_optimizer = torch.optim.Adam(F.parameters(), lr=0.005, weight_decay=0.0008)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optimizer, gamma=0.96)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    F.train()
    encoder.eval()

    for epoch in range(args.epochs):
        start = time.time()
        top1_train_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            my_optimizer.zero_grad()
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            h = encoder(normalzie(args, x_batch))
            downstream_input = h.view(h.size(0), -1)
            logits = F(downstream_input)
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]
            loss.backward()
            my_optimizer.step()

        end = time.time()
        F.train()
        clean_acc_t1, clean_acc_t5 = test(args, encoder, F, test_loader, data)
        torch.save(F, '{}/{}'.format(uap_save_path,  str(args.victim) + '_' +str(args.pre_dataset) + '_' + str(args.dataset) + '_' + str(
            round(clean_acc_t1, 4)) + '_' + str(epoch + 1) + '.pth'))

        my_lr_scheduler.step()
        top1_train_accuracy /= (counter + 1)
        print('Epoch [%d/%d], Top1 train acc: %.4f, Top1 test acc: %.4f, Time: %.4f'
            % (epoch + 1, args.epochs, top1_train_accuracy.item(), clean_acc_t1,(end - start)))


    return clean_acc_t1, clean_acc_t5

def main():
    args = arg_parse()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.set_printoptions(profile="full")
    torch.cuda.synchronize()

    # Logging
    log_save_path = os.path.join('output', str(args.pre_dataset), 'log', 'down_test', "clean_model", str(args.victim), str(args.dataset))
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)

    now_time = make_print_to_file(path=log_save_path)

    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)

    # Dump args
    with open(log_save_path + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)

    model = load_victim(args)

    print('Day: %s, Target encoder:%s, Downstream task:%s'% (now_time, args.victim,  args.dataset))
    print("######################################  Test Attack! ######################################")

    clean_acc_t1, clean_acc_t5 = classify(args, model)
    print('Clean downstream accuracy: %.4f%%'% (clean_acc_t1))


if __name__ == "__main__":
    main()
    