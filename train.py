#  Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# Python imports
import tqdm
from tqdm import tqdm
import os
from os.path import join as ospj
import csv
import datetime as dt
#Local imports
from data import dataset as dset
from models.common import Evaluator
from utils.utils import save_args, load_args
from utils.config_model import configure_model,optimizer_builder
from flags import parser, DATA_FOLDER
import random
import numpy as np
from dataset import CompositionDataset
best_auc = 0
best_hm = 0
best_attr = 0
best_obj = 0
best_seen = 0
best_unseen = 0
latest_changes = 0
compose_switch = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random_seed = random.randint(0, 10000)
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
def main():
    # Get arguments and start logging
    now_time = dt.datetime.now().strftime('%F %T')
    print('new time is' + now_time)
    args = parser.parse_args()
    load_args(args.config, args)
    logpath = os.path.join(args.cv_dir, args.name+'_'+now_time)
    os.makedirs(logpath, exist_ok=True)
    save_args(args, logpath, args.config)

    best_val_auc = 0.0
    patience = 10
    counter = 0

    # Get dataset
    trainset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='train',
        split=args.splitname,
        model =args.image_extractor,
        num_negs=args.num_negs,
        pair_dropout=args.pair_dropout,
        update_features = args.update_features,
        train_only= args.train_only,
        open_world=args.open_world
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)
    testset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase=args.test_set,
        split=args.splitname,
        model =args.image_extractor,
        subset=args.subset,
        update_features = args.update_features,
        open_world=args.open_world
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers)


    # Get model and optimizer
    image_extractor, model, optimizer, image_decoupler = configure_model(args, trainset)
    args.extractor = image_extractor
    train = train_normal
    evaluator_val =  Evaluator(testset, model)
    start_epoch = 0

    p_log = dict()
    p_log['attr'] = torch.log(torch.ones((1,int(model.train_pairs.shape[0]))))
    p_log['objs'] = torch.log(torch.ones((1,int(model.train_pairs.shape[0]))))

    p_log['test_a'] = torch.log(torch.ones((1,int(model.val_pairs.shape[0]))))
    p_log['test_o'] = torch.log(torch.ones((1,int(model.val_pairs.shape[0]))))

    for epoch in tqdm(range(start_epoch, args.max_epochs + 1), desc = 'Current epoch'):
        p_log['attr'] = p_log['attr'].to(device)
        p_log['objs'] = p_log['objs'].to(device)
        model.p_log = p_log

        model.args.if_ds = False
        model.freeze_model(model.C_y ,False)
        optimizer = optimizer_builder(args,model,image_extractor,image_decoupler)
        p_log = train(epoch, image_extractor, model, trainloader, optimizer,p_log,image_decoupler)
        if epoch % args.eval_val_every == 0:
            with torch.no_grad(): # todo: might not be needed
                auc = test(epoch, image_extractor, model, testloader, evaluator_val, args, logpath,image_decoupler)

        if auc > best_val_auc:
            best_val_auc = auc
            counter = 0
            print('new val auc is',"{:.2f}%".format(auc*100))
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    counter = 0
    best_val_auc = 0
    patience = patience*10
    for epoch in tqdm(range(start_epoch, args.max_epochs + 1), desc = 'Current epoch'):
        p_log['attr'] = p_log['attr'].to(device)
        p_log['objs'] = p_log['objs'].to(device)
        model.p_log = p_log
        model.args.if_ds = True
        model.freeze_model(model.C_y, True)
        optimizer = optimizer_builder(args, model, image_extractor, image_decoupler)
        p_log = train(epoch, image_extractor, model, trainloader, optimizer, p_log, image_decoupler)
        if epoch % args.eval_val_every == 0:
            with torch.no_grad():  # todo: might not be needed
                auc = test(epoch, image_extractor, model, testloader, evaluator_val, args, logpath, image_decoupler)

        if auc > best_val_auc:
            best_val_auc = auc
            counter = 0
            print('new val auc is', "{:.2f}%".format(auc * 100))
            embedding_save_path = os.path.join(logpath, 'Best_AUC_Embedding.pth')
            torch.save(model.state_dict(), embedding_save_path)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

def train_normal(epoch, image_extractor, model, trainloader, optimizer,p_log,img_decoupler):
    '''
    Runs training for an epoch
    '''

    if image_extractor:
        image_extractor.train()
    model.train() # Let's switch to training

    train_loss = 0.0
    for idx, data in tqdm(enumerate(trainloader), total=len(trainloader), desc = 'Training'):
        data  = [d.to(device) for d in data]

        if image_extractor:
            img = data[0]
            data[0] = image_extractor(data[0])
            data.append(img_decoupler(img))
        loss, pred = model(data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx == 0:
            pred_attr = pred[0].detach().cpu().numpy()
            pred_objs = pred[1].detach().cpu().numpy()

        else:
            pred_attr = np.concatenate((pred_attr, pred[0].detach().cpu().numpy()))
            pred_objs = np.vstack([pred_objs, pred[1].detach().cpu().numpy()])

        train_loss += loss.item()
    train_loss = train_loss/len(trainloader)
    print('Epoch: {}| Loss: {}'.format(epoch, round(train_loss, 2)))

    '''k(s,o)'''
    pred_attr = np.mean(pred_attr , axis=0)
    pred_objs = np.mean(pred_objs , axis=0)
    p_log['attr'] = torch.from_numpy(pred_objs)
    p_log['objs'] = torch.from_numpy(pred_attr)
    p_log['attr'] = F.softmax(p_log['attr'],dim=-1)
    p_log['objs'] = F.softmax(p_log['objs'],dim=-1)
    p_log['attr'] = np.log(p_log['attr'])
    p_log['objs'] = np.log(p_log['objs'])

    return p_log

def test(epoch, image_extractor, model, testloader, evaluator, args, logpath,image_decoupler):
    '''
    Runs testing for an epoch
    '''
    global best_auc, best_hm, best_obj,best_attr,best_seen,best_unseen,latest_changes
    if image_extractor:
        image_extractor.eval()
        image_decoupler.eval()

    model.eval()

    accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []

    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
        data = [d.to(device) for d in data]

        if image_extractor:
            img = data[0]
            data[0] = image_extractor(data[0])
            data.append(image_decoupler(img))

        _, predictions = model(data)

        attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]

        all_pred.append(predictions)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)

    if args.cpu_eval:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt), torch.cat(all_obj_gt), torch.cat(all_pair_gt)
    else:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
            'cpu'), torch.cat(all_pair_gt).to('cpu')

    all_pred_dict = {}
    # Gather values as dict of (attr, obj) as key and list of predictions as values
    if args.cpu_eval:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k].to('cpu') for i in range(len(all_pred))])
    else:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k] for i in range(len(all_pred))])

    # Calculate best unseen accuracy
    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=args.bias, topk=args.topk)
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=args.topk)
    return stats['AUC']
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Latest Improved Epoch is', latest_changes)