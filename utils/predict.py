import os
import sys
import datetime
import torch
from tqdm import tqdm
import torch.nn.functional as F
from utils.load_data import normalzie
from utils.patch_utils import patch_initialization, mask_generation

def make_print_to_file(path='./'):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "w", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime('%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)
    return fileName


def knn_per_fr(args, train_loader, test_loader, model, p, k=20, T=0.07):
    # extract train features
    train_features, train_targets = extract_features(args, train_loader, model)

    # extract test features
    test_features, test_targets = extract_features(args, test_loader, model)

    # extract per_test features
    p_test_features, p_test_targets = extract_per_features(args, test_loader, model, p)

    max_distance_matrix_size = int(5e6)
    train_features = F.normalize(train_features)
    test_features = F.normalize(test_features)
    p_test_features = F.normalize(p_test_features)

    num_classes = torch.unique(test_targets).numel()
    num_test_images = test_targets.size(0)
    num_p_test_images = p_test_targets.size(0)
    num_train_images = train_targets.size(0)
    chunk_size = min(
        max(1, max_distance_matrix_size // num_train_images),
        num_test_images,
    )

    k = min(k, num_train_images)

    # test clean
    top1, top5, total = 0.0, 0.0, 0
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    p_top1, p_top5, p_total = 0.0, 0.0, 0
    p_retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    fr = 0.0

    for idx in range(0, num_test_images, chunk_size):
        # get the features for test images
        features = test_features[idx: min((idx + chunk_size), num_test_images), :]
        targets = test_targets[idx: min((idx + chunk_size), num_test_images)]
        batch_size = targets.size(0)

        # calculate the dot product and compute top-k neighbors
        similarities = torch.mm(features, train_features.t())

        similarities, indices = similarities.topk(k, largest=True, sorted=True)
        candidates = train_targets.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)

        similarities = similarities.clone().div_(T).exp_()

        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                similarities.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # get the features for test images
        p_features = p_test_features[idx: min((idx + chunk_size), num_p_test_images), :]
        p_targets = p_test_targets[idx: min((idx + chunk_size), num_p_test_images)]
        p_batch_size = p_targets.size(0)

        # calculate the dot product and compute top-k neighbors
        p_similarities = torch.mm(p_features, train_features.t())

        p_similarities, p_indices = p_similarities.topk(k, largest=True, sorted=True)
        p_candidates = train_targets.view(1, -1).expand(p_batch_size, -1)
        p_retrieved_neighbors = torch.gather(p_candidates, 1, p_indices)

        p_retrieval_one_hot.resize_(p_batch_size * k, num_classes).zero_()
        p_retrieval_one_hot.scatter_(1, p_retrieved_neighbors.view(-1, 1), 1)
        p_similarities = p_similarities.clone().div_(T).exp_()

        p_probs = torch.sum(
            torch.mul(
                p_retrieval_one_hot.view(p_batch_size, -1, num_classes),
                p_similarities.view(p_batch_size, -1, 1),
            ),
            1,
        )
        _, p_predictions = p_probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = (top5 + correct.narrow(1, 0,
                                      min(5, k, correct.size(-1))).sum().item())  # top5 does not make sense if k < 5
        total += targets.size(0)

        # find the predictions that match the target
        p_correct = p_predictions.eq(p_targets.data.view(-1, 1))
        p_top1 = p_top1 + p_correct.narrow(1, 0, 1).sum().item()
        p_top5 = (p_top5 + p_correct.narrow(1, 0, min(5, k, p_correct.size(
            -1))).sum().item())  # top5 does not make sense if k < 5

        p_total += p_targets.size(0)

        fr += predictions.eq(p_predictions).narrow(1, 0, 1).sum().item()

    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total

    p_top1 = p_top1 * 100.0 / p_total
    p_top5 = p_top5 * 100.0 / p_total

    attack_success_rate = (total - fr)  * 100.0 / float(total)

    return top1, top5, p_top1, p_top5, attack_success_rate


def knn_patch_fr(args, train_loader, test_loader, model, p, mask, k=20, T=0.07):
    # extract train features
    train_features, train_targets = extract_features(args, train_loader, model)

    # extract test features
    test_features, test_targets = extract_features(args, test_loader, model)

    # extract per_test features
    p_test_features, p_test_targets = extract_patch_features(args, test_loader, model, p, mask)

    max_distance_matrix_size = int(5e6)
    train_features = F.normalize(train_features)
    test_features = F.normalize(test_features)
    p_test_features = F.normalize(p_test_features)

    num_classes = torch.unique(test_targets).numel()
    num_test_images = test_targets.size(0)
    num_p_test_images = p_test_targets.size(0)
    num_train_images = train_targets.size(0)
    chunk_size = min(
        max(1, max_distance_matrix_size // num_train_images),
        num_test_images,
    )

    k = min(k, num_train_images)

    # test clean
    top1, top5, total = 0.0, 0.0, 0
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    p_top1, p_top5, p_total = 0.0, 0.0, 0
    p_retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    fr = 0.0

    for idx in range(0, num_test_images, chunk_size):
        # get the features for test images
        features = test_features[idx: min((idx + chunk_size), num_test_images), :]
        targets = test_targets[idx: min((idx + chunk_size), num_test_images)]
        batch_size = targets.size(0)

        # calculate the dot product and compute top-k neighbors
        similarities = torch.mm(features, train_features.t())

        similarities, indices = similarities.topk(k, largest=True, sorted=True)
        candidates = train_targets.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)

        similarities = similarities.clone().div_(T).exp_()

        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                similarities.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # get the features for test images
        p_features = p_test_features[idx: min((idx + chunk_size), num_p_test_images), :]
        p_targets = p_test_targets[idx: min((idx + chunk_size), num_p_test_images)]
        p_batch_size = p_targets.size(0)

        # calculate the dot product and compute top-k neighbors
        p_similarities = torch.mm(p_features, train_features.t())

        p_similarities, p_indices = p_similarities.topk(k, largest=True, sorted=True)
        p_candidates = train_targets.view(1, -1).expand(p_batch_size, -1)
        p_retrieved_neighbors = torch.gather(p_candidates, 1, p_indices)

        p_retrieval_one_hot.resize_(p_batch_size * k, num_classes).zero_()
        p_retrieval_one_hot.scatter_(1, p_retrieved_neighbors.view(-1, 1), 1)
        p_similarities = p_similarities.clone().div_(T).exp_()

        p_probs = torch.sum(
            torch.mul(
                p_retrieval_one_hot.view(p_batch_size, -1, num_classes),
                p_similarities.view(p_batch_size, -1, 1),
            ),
            1,
        )
        _, p_predictions = p_probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = (top5 + correct.narrow(1, 0,
                                      min(5, k, correct.size(-1))).sum().item())  # top5 does not make sense if k < 5
        total += targets.size(0)

        # find the predictions that match the target
        p_correct = p_predictions.eq(p_targets.data.view(-1, 1))
        p_top1 = p_top1 + p_correct.narrow(1, 0, 1).sum().item()
        p_top5 = (p_top5 + p_correct.narrow(1, 0, min(5, k, p_correct.size(
            -1))).sum().item())  # top5 does not make sense if k < 5

        p_total += p_targets.size(0)

        fr += predictions.eq(p_predictions).narrow(1, 0, 1).sum().item()

    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total

    p_top1 = p_top1 * 100.0 / p_total
    p_top5 = p_top5 * 100.0 / p_total

    attack_success_rate = (total - fr)  * 100.0 / float(total)
    return top1, top5, p_top1, p_top5, attack_success_rate


@torch.no_grad()
def extract_features(args, loader, model):
    model.eval()
    backbone_features, labels = [], []
    for im, lab in tqdm(loader):
        im = im.cuda(non_blocking=True)
        lab = lab.cuda(non_blocking=True)
        outs = model(normalzie(args,im))
        backbone_features.append(outs)
        labels.append(lab)
    model.train()
    backbone_features = torch.cat(backbone_features)
    labels = torch.cat(labels)
    return backbone_features,  labels

@torch.no_grad()
def extract_patch_features(args, loader, model, p, mask):
    model.eval()
    backbone_features, labels = [], []
    for im, lab in tqdm(loader):
        new_shape = im.shape
        im = torch.mul(mask.type(torch.FloatTensor), p.type(torch.FloatTensor)) + torch.mul(
            1 - mask.expand(new_shape).type(torch.FloatTensor), im.type(torch.FloatTensor))

        im = im.cuda(non_blocking=True)
        lab = lab.cuda(non_blocking=True)
        outs = model(normalzie(args, im))
        backbone_features.append(outs)
        labels.append(lab)
    model.train()
    backbone_features = torch.cat(backbone_features)
    labels = torch.cat(labels)
    return backbone_features,  labels

@torch.no_grad()
def extract_per_features(args, loader, model, p):
    model.eval()
    backbone_features, labels = [], []
    for im, lab in tqdm(loader):
        new_shape = im.shape
        im = im.cuda(non_blocking=True)
        lab = lab.cuda(non_blocking=True)
        im = im + p.expand(new_shape)
        outs = model(normalzie(args, im))
        backbone_features.append(outs)
        labels.append(lab)
    model.train()
    backbone_features = torch.cat(backbone_features)
    labels = torch.cat(labels)
    return backbone_features,  labels

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # correct = pred.eq(target.expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def attack_success_rate(clean_output, per_output, target):
    with torch.no_grad():
        # maxk = max(topk)
        batch_size = target.size(0)

        _, c_pred = clean_output.max(1)
        c_pred = c_pred.t()

        _, p_pred = per_output.max(1)
        p_pred = p_pred.t()

        attack_success_rate = float(torch.sum(c_pred != p_pred)) / float(batch_size)

        return attack_success_rate * 100

def test(args, encoder, classifier, test_loader, data):
    top1_accuracy = 0
    top5_accuracy = 0

    classifier.eval()

    with torch.no_grad():
        for counter, (x_batch, y_batch) in enumerate(tqdm(test_loader)):
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            h = encoder(normalzie(args, x_batch))
            x_in = h.view(h.size(0), -1)
            logits = classifier(x_in)
            top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)

    return top1_accuracy.item(), top5_accuracy.item()


def adv_test(args, encoder, classifier, test_loader, uap, data):
    top1_accuracy = 0
    top5_accuracy = 0

    classifier.eval()

    with torch.no_grad():
        for counter, (x_batch, y_batch) in enumerate(tqdm(test_loader)):
            new_shape = x_batch.shape
            if args.type == 'gan_patch':
                patch = patch_initialization(args)
                mask, applied_patch, x, y = mask_generation(args, patch)
                mask = torch.from_numpy(mask)
                x_batch = torch.mul(mask.type(torch.FloatTensor), uap.type(torch.FloatTensor)) + torch.mul(1 - mask.expand(new_shape).type(torch.FloatTensor), x_batch.type(torch.FloatTensor))

            else:
                x_batch = x_batch.cuda() + uap.expand(new_shape).cuda()

            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

            h = encoder(normalzie(args, x_batch))
            x_in = h.view(h.size(0), -1)
            logits = classifier(x_in)
            # print(counter)
            top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)

    return top1_accuracy.item(), top5_accuracy.item()

def fr_test(args, encoder, classifier, test_loader, uap, data):
    fr = 0
    classifier.eval()

    with torch.no_grad():
        for counter, (x_batch, y_batch) in enumerate(tqdm(test_loader)):

            new_shape = x_batch.shape

            clean_x_batch = x_batch.clone()
            clean_x_batch = clean_x_batch.cuda()
            y_batch = y_batch.cuda()

            if args.type == 'gan_patch':
                patch = patch_initialization(args)
                mask, applied_patch, x, y = mask_generation(args, patch)
                # applied_patch = torch.from_numpy(applied_patch)
                mask = torch.from_numpy(mask)
                per_x_batch = torch.mul(mask.type(torch.FloatTensor), uap.type(torch.FloatTensor)) + torch.mul(
                    1 - mask.expand(new_shape).type(torch.FloatTensor), x_batch.type(torch.FloatTensor))
            else:
                per_x_batch = x_batch.cuda() + uap.expand(new_shape).cuda()

            per_x_batch = per_x_batch.cuda()
            c_h = encoder(normalzie(args, clean_x_batch))
            p_h = encoder(normalzie(args, per_x_batch))

            c_x_in = c_h.view(c_h.size(0), -1)
            p_x_in = p_h.view(p_h.size(0), -1)

            c_logits = classifier(c_x_in)
            p_logits = classifier(p_x_in)

            fooling_ra = attack_success_rate(c_logits, p_logits, y_batch)
            fr += fooling_ra

        fr /= (counter + 1)

    return fr
