import os
import time
import torch
import random
import json
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import csv
from torch.autograd import Variable
from model.adv_gan import Generator
from utils.nce import InfoNCE
from utils.load_model import load_victim
from utils.load_data import load_data, normalzie
from utils.predict import knn_per_fr, make_print_to_file
from utils.fr_util import generate_high
import warnings
warnings.filterwarnings("ignore")

def arg_parse():
    parser = argparse.ArgumentParser(description="AdvEncoder-PER")
    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--gpu', default='0, 1', type=str, help='which gpu the code runs on')
    parser.add_argument('--pre_dataset', default='cifar10',choices=['cifar10', 'imagenet'])
    parser.add_argument('--dataset', default='imagenet',choices=['cifar10', 'stl10', 'gtsrb', 'imagenet', 'minst', 'fashion-mnist'])
    parser.add_argument('--victim', default='simclr', choices=['simclr', 'byol', 'barlow_twins', 'deepclusterv2', 'dino', 'mocov3', 'mocov2plus', 'nnclr', 'ressl', 'supcon', 'swav', 'vibcreg', 'vicreg', 'wmse'])
    parser.add_argument('--criterion', default='nce', choices=['cos', 'nt', 'nce'])
    parser.add_argument('--alpha', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eps',  type=int, default= 10)
    parser.add_argument('--temperature', default=0.5, type=float, help='temperature')
    parser.add_argument('--radius', type=int, default=8, help='radius of low freq images')
    args = parser.parse_args()
    return args

def uap_dcgan_attack(args, train_loader, test_loader, model):
    # init the GAN
    G_input_dim = 100
    G_output_dim = 3
    num_filters = [1024, 512, 256, 128]
    learning_rate = 0.0002
    betas = (0.5, 0.999)

    # save results
    results = {'clean_acc_t1': [], 'adv_acc_t1': [], 'decline_t1': [], 'clean_acc_t5': [], 'adv_acc_t5': [], 'decline_t5': [], 'fooling_rate': [], 'time': []}
    epoch_start = 0

    G = Generator(G_input_dim, num_filters, G_output_dim, args.batch_size)

    model.eval()
    G.cuda()

    # criterion_l2
    criterion_l2 = torch.nn.MSELoss()

    # criterion_contrastive
    if args.criterion == 'cos':
        criterion_contrastive = nn.CosineSimilarity(dim=0, eps=1e-6)
    elif args.criterion == 'nce':
        criterion_contrastive = InfoNCE()

    # Optimizers
    G_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=betas)
    
    # Training GAN
    # define a global fix noise z
    z = torch.randn(args.batch_size, G_input_dim).view(-1, G_input_dim, 1, 1)
    z = Variable(z.cuda())

    for epoch in range(args.epochs):

        # init the start time
        start = time.time()

        for i, (images, _) in enumerate(train_loader):

            x = Variable(images.cuda())
            new_shape = x.shape

            uap_noise = G(z).squeeze()
            uap_noise = torch.clamp(uap_noise, -args.eps, args.eps)
            uap_noise.cuda()
            
            # fake image
            f_x = x + uap_noise.expand(new_shape)

            x.cuda()
            f_x.cuda()
            
            # l_{2} loss
            reconstruction_loss = criterion_l2(f_x, x)

            clean_output = model(normalzie(args, x))
            per_output = model(normalzie(args, f_x).cuda())
            
            
            # adv loss
            if args.criterion == 'cos':
                adv_loss = criterion_contrastive(clean_output, per_output).mean()
            else:
                adv_loss_pos = criterion_contrastive(clean_output, per_output).mean()
                adv_loss = -adv_loss_pos

            # HFC loss
            clean_hfc = generate_high(x, r=args.radius)
            per_hfc = generate_high(f_x, r=args.radius)
            HFC_loss = criterion_l2(clean_hfc, per_hfc)
            HFC_loss = - HFC_loss
            # lack D
            G_loss = args.alpha * adv_loss + reconstruction_loss + HFC_loss

            # Back propagation
            G.zero_grad()
            G_loss.backward()
            G_optimizer.step()


            if i % 1 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Adv_loss: %.4f, L2_loss: %.4f, HFC_loss: %.4f, G_loss: %.4f'
                    % (epoch + 1, args.epochs, i + 1, len(train_loader), adv_loss.item(), reconstruction_loss.item(), HFC_loss.item(), G_loss.item()))
        end = time.time()
        run_time = end - start


        # caculate the acc decline
        clean_acc_t1, clean_acc_t5, adv_acc_t1, adv_acc_t5, fooling_rate = knn_per_fr(args, train_loader, test_loader,
                                                                                      model, uap_noise.cuda())

        decline_t1 = ((clean_acc_t1 - adv_acc_t1) / clean_acc_t1) * 100
        decline_t5 = ((clean_acc_t5 - adv_acc_t5) / clean_acc_t5) * 100

        print('##############################  Epoch [%d/%d], Clean_acc: %.4f%%, Per_acc: %.4f%%, Decline Rate: %.4f%%, Fooling Rate: %.4f%%! ##############################' % (
                epoch + 1, args.epochs, clean_acc_t1, adv_acc_t1, decline_t1, fooling_rate), end='\n')

        results['clean_acc_t1'].append(clean_acc_t1)
        results['clean_acc_t5'].append(clean_acc_t5)
        results['decline_t1'].append(decline_t1)
        results['adv_acc_t1'].append(adv_acc_t1)
        results['adv_acc_t5'].append(adv_acc_t5)
        results['decline_t5'].append(decline_t5)
        results['fooling_rate'].append(fooling_rate)
        results['time'].append(run_time)

        # Save statistics
        data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
        data_frame.to_csv(log_save_path + '/log.csv', indexlabel='epoch')

        # save uap result
        uap_save_path = os.path.join('output', str(args.pre_dataset), 'uap_results', 'gan_per', str(args.victim), str(args.dataset),
                                     str(args.criterion), str(args.eps))
        
        if not os.path.exists(uap_save_path):
            os.makedirs(uap_save_path)
        torch.save(uap_noise.cpu().data, '{}/{}'.format(uap_save_path, 'uap_gan_' + str(args.dataset) + '_' + str(
            round(decline_t1, 4)) + '_' + str(round(fooling_rate, 4)) + '_' + str(epoch + 1) + '.pt'))

    return clean_acc_t1, clean_acc_t5, adv_acc_t1, adv_acc_t5, fooling_rate, decline_t1, decline_t5



if __name__ == '__main__':

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
    args.eps = args.eps / 255

    # Logging
    log_save_path = os.path.join('output', str(args.pre_dataset), 'log', 'gan_per', str(args.victim), str(args.dataset), str(args.criterion), str(args.eps))
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)

    # Dump args
    with open(log_save_path + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)

    now_time = make_print_to_file(path = log_save_path)

    print("#########################################################  Attack Setting  ######################################################### ")
    print('Day: %s, Pre-trained Encoder: %s, Dataset: %s, Adv_criterion: %s, Epsilon: %.2f' % (now_time, args.victim, args.dataset, args.criterion, args.eps))

    # load the pre-trained model
    model = torch.nn.DataParallel(load_victim(args))


    # load the data
    train_loader, test_loader = load_data(args.dataset, args.batch_size)

    model.cuda()
    for param in model.parameters():
        param.requires_grad = False

    # output the basic information
    print("##########################################################  Attack Start! ########################################################## ")

    clean_acc_t1, clean_acc_t5, adv_acc_t1, adv_acc_t5, fooling_rate, decline_t1, decline_t5  = uap_dcgan_attack(args, train_loader, test_loader, model)


    final_log_save_path = os.path.join('output', str(args.pre_dataset), 'log', 'gan_per')
    if not os.path.exists(final_log_save_path):
        os.makedirs(final_log_save_path)

    final_result = []
    final_result_ = {"now_time": now_time, "victim": str(args.victim), "pre_dataset": str(args.pre_dataset), "sup_dataset": str(args.dataset),
                     "criterion": str(args.criterion), "eps": str(round(args.eps, 4)), "clean_acc_t1": round(clean_acc_t1, 4),
                     "clean_acc_t5": round(clean_acc_t5, 4), "decline_t1": round(decline_t1, 4), "adv_acc_t1": round(adv_acc_t1, 4),
                     "adv_acc_t5": round(adv_acc_t5, 4), "decline_t5": round(decline_t5, 4), "fooling_rate": round(fooling_rate, 4)}
    final_result.append(final_result_)

    header = ["now_time", "victim", "pre_dataset", "sup_dataset", "criterion", "eps", "clean_acc_t1", "clean_acc_t5", "decline_t1",
              "adv_acc_t1", "adv_acc_t5", "decline_t5", "fooling_rate"]

    with open(final_log_save_path + '/all_final_results.csv', 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(final_result)