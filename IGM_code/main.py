# implementation of our prpose IGM

import argparse
import os
import os.path as osp
from copy import deepcopy
from datetime import datetime
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from drugood.datasets import build_dataset
from mmcv import Config
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from sklearn.metrics import matthews_corrcoef
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
from GOOD.data.good_datasets.good_hiv import GOODHIV
#TODO:使用tensorboard
from torch.utils.tensorboard import SummaryWriter

from DataLoading import pyg_molsubdataset
import warnings
warnings.filterwarnings("ignore")
from datasets.drugood_dataset import DrugOOD
from datasets.graphss2_dataset import get_dataloader_per, get_dataset
from datasets.mnistsp_dataset import CMNIST75sp
from datasets.spmotif_dataset import SPMotif
from models.igm import  IGM
from models.losses import get_contrast_loss, get_irm_loss
from utils.logger import Logger
from utils.util import args_print, set_seed

def concrete_sample(att_log_logit, temp, training):
    if training:
        random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
    else:
        att_bern = (att_log_logit).sigmoid()
    return att_bern


def get_loaders_and_test_set(batch_size, dataset=None, split_idx=None, dataset_splits=None):
    if split_idx is not None:
        assert dataset is not None
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)
        test_set = dataset.copy(split_idx["test"])  # For visualization
    else:
        assert dataset_splits is not None
        train_loader = DataLoader(dataset_splits['train'], batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset_splits['valid'], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset_splits['test'], batch_size=batch_size, shuffle=False)
        test_set = dataset_splits['test']  # For visualization
    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}, test_set

def mix_criterion(input, target, size_average=True):
    """Categorical cross-entropy with logits input and one-hot target"""
    l = -(target * torch.log(F.softmax(input, dim=1) + 1e-10)).sum(1)
    if size_average:
        l = l.mean()
    else:
        l = l.sum()
    return l

@torch.no_grad()
def eval_model(model, device, loader, evaluator, eval_metric='acc', save_pred=False,c_pred=False):
    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        batch = batch.to(device)
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                batch.x = batch.x.float()
                batch.y  = batch.y.reshape(-1)
                pred = model(batch)
                # pred = model(batch,c_pred=c_pred)
            is_labeled = batch.y == batch.y
            if eval_metric == 'acc':
                if len(batch.y.size()) == len(batch.y.size()):
                    y_true.append(batch.y.view(-1, 1).detach().cpu())
                    y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu())
                else:
                    y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                    y_pred.append(pred.argmax(-1).unsqueeze(-1).detach().cpu())
            elif eval_metric == 'rocauc':
                pred = F.softmax(pred, dim=-1)[:, 1].unsqueeze(-1)
                if len(batch.y.size()) == len(batch.y.size()):
                    y_true.append(batch.y.view(-1, 1).detach().cpu())
                    y_pred.append(pred.detach().view(-1, 1).cpu())
                else:
                    y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                    y_pred.append(pred.unsqueeze(-1).detach().cpu())
            elif eval_metric == 'mat':
                y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                y_pred.append(pred.argmax(-1).unsqueeze(-1).detach().cpu())
            elif eval_metric == 'ap':
                y_true.append(batch.y.view(pred.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())
            else:
                if is_labeled.size() != pred.size():
                    with torch.no_grad():
                        pred, rep = model(batch, return_data="rep", debug=True)
                        print(rep.size())
                    print(batch)
                    print(global_mean_pool(batch.x, batch.batch).size())
                    print(pred.shape)
                    print(batch.y.size())
                    print(sum(is_labeled))
                    print(batch.y)
                batch.y = batch.y[is_labeled]
                pred = pred[is_labeled]
                y_true.append(batch.y.view(pred.shape).unsqueeze(-1).detach().cpu())
                y_pred.append(pred.detach().unsqueeze(-1).cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    if eval_metric == 'mat':
        res_metric = matthews_corrcoef(y_true, y_pred)
    else:
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        res_metric = evaluator.eval(input_dict)[eval_metric]

    if save_pred:
        return res_metric, y_pred
    else:
        return res_metric


def main():
    parser = argparse.ArgumentParser(description='Causality Inspired Invariant Graph LeArning')
    parser.add_argument('--device', default=0, type=int, help='cuda device')
    parser.add_argument('--root', default='./data', type=str, help='directory for datasets.')
    parser.add_argument('--dataset', default='SPMotif', type=str)
    parser.add_argument('--bias', default='0.5', type=str, help='select bias extend')
    parser.add_argument('--feature', type=str, default="full", help='full feature or simple feature')

    # training config
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--epoch', default=400, type=int, help='training iterations')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate for the predictor')
    parser.add_argument('--seed', nargs='?', default='[1,2,3,4,5,6,7,8,9,10]', help='random seed')
    parser.add_argument('--pretrain', default=20, type=int, help='pretrain epoch before early stopping')

    # model config
    parser.add_argument('--emb_dim', default=32, type=int)
    
    # emb dim of classifier
    parser.add_argument('-c_dim', '--classifier_emb_dim', default=32, type=int)
    # inputs of classifier
    # raw:  raw feat
    # feat: hidden feat from featurizer 
    parser.add_argument('-c_in', '--classifier_input_feat', default='raw', type=str)
    parser.add_argument('--model', default='gin', type=str)
    parser.add_argument('--pooling', default='mean', type=str)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--early_stopping', default=5, type=int)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--virtual_node', action='store_true')
    parser.add_argument('--eval_metric',
                        default='',
                        type=str,
                        help='specify a particular eval metric, e.g., mat for MatthewsCoef')



    # model cofig

    parser.add_argument('--c_pred', default=0.0, type=float, help='use casual part to predict')
    parser.add_argument('--s_pred', default=0.0, type=float, help='use spu part to predict')
    parser.add_argument('--caus', default=0.0, type=float, help='add casual manifold mixup')
    parser.add_argument('--mix', default=0.0, type=float)
    parser.add_argument('--r', default=0.7, type=float, help='selected ratio')
    parser.add_argument('--my_irm', default=0.0, type=float)
    parser.add_argument('--my_vrex', default=0.0, type=float)


    # misc
    parser.add_argument('--no_tqdm', action='store_true')
    parser.add_argument('--commit', default='', type=str, help='experiment name')
    parser.add_argument('--save_model', action='store_true')  # save pred to ./pred if not empty

    args = parser.parse_args()
    erm_model = None  # used to obtain pesudo labels for CNC sampling in contrastive loss
    # log
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    #TODO: debug tesorboard

    args.seed = eval(args.seed)

    if args.device<=7:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(device)
    def ce_loss(a, b, reduction='mean'):
        return F.cross_entropy(a, b, reduction=reduction)

    criterion = ce_loss
    eval_metric = 'acc' if len(args.eval_metric) == 0 else args.eval_metric
    edge_dim = -1.

    ### automatic dataloading and splitting
    if args.dataset.lower().startswith('drugood'):
        # drugood_lbap_core_ic50_assay.json
        config_path = os.path.join("configs", args.dataset + ".py")
        cfg = Config.fromfile(config_path)
        root = os.path.join(args.root, "DrugOOD")
        train_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.train), name=args.dataset, mode="train")
        val_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.ood_val), name=args.dataset, mode="ood_val")
        test_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.ood_test), name=args.dataset, mode="ood_test")
        if args.eval_metric == 'auc':
            evaluator = Evaluator('ogbg-molhiv')
            eval_metric = args.eval_metric = 'rocauc'
        else:
            evaluator = Evaluator('ogbg-ppa')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        input_dim = 39
        edge_dim = 10
        num_classes = 2
    elif args.dataset.lower().startswith('ogbg'):

        _,_,dataset = pyg_molsubdataset(args.dataset, 'recap')

        input_dim = dataset[0].x.shape[1]
        num_classes = dataset.num_classes
        if args.feature == 'full':
            pass
        
        split_idx = dataset.get_idx_split()
        ### automatic evaluator. takes dataset name as input
        evaluator = Evaluator(args.dataset)
        # evaluator = Evaluator('ogbg-ppa')

        train_loader = DataLoader(dataset[split_idx["train"]],
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers)
        valid_loader = DataLoader(dataset[split_idx["valid"]],
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers)
        test_loader = DataLoader(dataset[split_idx["test"]],
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers)

        eval_metric = dataset.eval_metric
    elif args.dataset.lower() in ['spmotif', 'mspmotif']:
        train_dataset = SPMotif(os.path.join(args.root, f'{args.dataset}-{args.bias}/'), mode='train')
        val_dataset = SPMotif(os.path.join(args.root, f'{args.dataset}-{args.bias}/'), mode='val')
        test_dataset = SPMotif(os.path.join(args.root, f'{args.dataset}-{args.bias}/'), mode='test')
        input_dim = 4
        num_classes = 3
        evaluator = Evaluator('ogbg-ppa')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    elif args.dataset.lower() in ['graph-sst5']:
        dataset = get_dataset(dataset_dir=args.root, dataset_name=args.dataset, task=None)
        dataloader = get_dataloader_per(dataset, batch_size=args.batch_size, small_to_large=True, seed=args.seed)
        train_loader = dataloader['train']
        valid_loader = dataloader['eval']
        test_loader = dataloader['test']
        input_dim = 768
        num_classes = int(args.dataset[-1].lower()) if args.dataset[-1].lower() in ['2', '5'] else 3
        evaluator = Evaluator('ogbg-ppa')
    elif args.dataset.lower() in ['graph-twitter']:
        dataset = get_dataset(dataset_dir=args.root, dataset_name=args.dataset, task=None)
        dataloader = get_dataloader_per(dataset, batch_size=args.batch_size, small_to_large=False, seed=args.seed)
        train_loader = dataloader['train']
        valid_loader = dataloader['eval']
        test_loader = dataloader['test']
        input_dim = 768
        num_classes = int(args.dataset[-1].lower()) if args.dataset[-1].lower() in ['2', '5'] else 3
        evaluator = Evaluator('ogbg-ppa')
    elif args.dataset.lower() in ['cmnist']:
        n_val_data = 5000
        train_dataset = CMNIST75sp(os.path.join(args.root, 'CMNISTSP/'), mode='train')
        test_dataset = CMNIST75sp(os.path.join(args.root, 'CMNISTSP/'), mode='test')
        perm_idx = torch.randperm(len(test_dataset), generator=torch.Generator().manual_seed(0))
        test_val = test_dataset[perm_idx]
        val_dataset, test_dataset = test_val[:n_val_data], test_val[n_val_data:]
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        input_dim = 7
        num_classes = 2
        evaluator = Evaluator('ogbg-ppa')
    elif args.dataset.lower() in ['proteins', 'dd', 'nci1', 'nci109']:
        dataset = TUDataset(os.path.join(args.root, "TU"), name=args.dataset.upper())
        train_idx = np.loadtxt(os.path.join(args.root, "TU", args.dataset.upper(), 'train_idx.txt'), dtype=np.int64)
        val_idx = np.loadtxt(os.path.join(args.root, "TU", args.dataset.upper(), 'val_idx.txt'), dtype=np.int64)
        test_idx = np.loadtxt(os.path.join(args.root, "TU", args.dataset.upper(), 'test_idx.txt'), dtype=np.int64)

        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]
        test_dataset = dataset[test_idx]
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)#TODO:
        valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        input_dim = dataset[0].x.size(1)
        num_classes = dataset.num_classes
        evaluator = Evaluator('ogbg-ppa')
    #TODO: ADD molhiv dataset
    elif args.dataset.lower() in ['molhiv']:
        dataset = PygGraphPropPredDataset(root='data/', name='ogbg-molhiv')
        split_idx = dataset.get_idx_split()
        print('[INFO] Using default splits!')
        loaders, test_set = get_loaders_and_test_set(args.batch_size, dataset=dataset, split_idx=split_idx)
        train_loader = loaders['train']
        valid_loader = loaders['valid']
        test_loader = loaders['test']
        input_dim = dataset[0].x.size(1)
        num_classes = dataset.num_classes
        evaluator = Evaluator('ogbg-molhiv')
        eval_metric = args.eval_metric = 'rocauc'
    elif args.dataset.lower() in ['good']:
        dataset, _ = GOODHIV.load('data/', domain='scaffold', shift='covariate', generate=False)
        train_loader=DataLoader(dataset["train"],args.batch_size,shuffle=True)
        valid_loader=DataLoader(dataset["val"],args.batch_size,shuffle=False)
        test_loader=DataLoader(dataset["test"],args.batch_size,shuffle=False)
        input_dim = 9
        num_classes = 2
        evaluator = Evaluator('ogbg-molhiv')
        eval_metric = 'rocauc'

    else:
        raise Exception("Invalid dataset name")

    
    all_info = {
        'test_acc': [],
        'train_acc': [],
        'val_acc': [],
    }
    experiment_name = f'{args.dataset}-{args.bias}_my_{args.my}_erm_{args.erm}_coes{args.contrast}-{args.spu_coe}_seed{args.seed}_envs_{args.num_envs}_{datetime_now}'
    #experiment_name = f'{datetime_now[4::]}'
    exp_dir = os.path.join('./logs/', experiment_name)
    os.mkdir(exp_dir)
    logger = Logger.init_logger(filename=exp_dir + '/log.log')
    args_print(args, logger)
    logger.info(f"Using criterion {criterion}")

    logger.info(f"# Train: {len(train_loader.dataset)}  #Val: {len(valid_loader.dataset)} #Test: {len(test_loader.dataset)} ")
    best_weights = None

    for seed in args.seed:
        set_seed(seed)
        
        model = IGM(ratio=args.r,
                        input_dim=input_dim,
                        edge_dim=edge_dim,
                        out_dim=num_classes,
                        gnn_type=args.model,
                        num_layers=args.num_layers,
                        emb_dim=args.emb_dim,
                        drop_ratio=args.dropout,
                        graph_pooling=args.pooling,
                        virtual_node=args.virtual_node,
                        c_dim=args.classifier_emb_dim,
                        c_in=args.classifier_input_feat,
                        c_rep=args.contrast_rep,
                        c_pool=args.contrast_pooling,
                        s_rep=args.spurious_rep).to(device)
        model_optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
        
        last_train_acc, last_test_acc, last_val_acc = 0, 0, 0
        cnt = 0
        for epoch in range(args.epoch):
            
            #TODO:debug
            batch_ratio_list = []
            batch_weight_list = []
            all_loss, n_bw = 0, 0
            all_losses = {}
            contrast_loss, all_contrast_loss = torch.zeros(1).to(device), 0.
            spu_pred_loss = torch.zeros(1).to(device)
            model.train()
            torch.autograd.set_detect_anomaly(True)
            num_batch = (len(train_loader.dataset) // args.batch_size) + int(
                (len(train_loader.dataset) % args.batch_size) > 0)
            for step, graph in tqdm(enumerate(train_loader), total=num_batch, desc=f"Epoch [{epoch}] >>  ", disable=args.no_tqdm, ncols=60):
                

                n_bw += 1
                graph.to(device)
                graph.x = graph.x.float()
                graph.y = graph.y.reshape(-1)
                # ignore nan targets
                # https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/main_pyg.py
                is_labeled = graph.y == graph.y
                #is_labeled = is_labeled.reshape(-1)

                
                if args.caus:
                    mixup_x,mixup_y,ori_pred,mix_pred,new_y,c_pred,edge_ratio_list,edge_weight_list,c_graph_pred,mix_rep= model(graph,return_data="feat",casual_mix=True,num_label=num_classes)
                    cau_loss = mix_criterion(mixup_x[is_labeled],mixup_y[is_labeled])

                else:
                    ori_pred,mix_pred,new_y,c_pred = model(graph,return_data="feat")
                #TODO: debug add all elements in edge_ratio_list to batch_ratio_list
                # print(edge_ratio_list)
                batch_ratio_list.extend(edge_ratio_list)
                batch_weight_list.extend(edge_weight_list)
    
                dummy_w = torch.tensor(1.).to(device).requires_grad_()
                ori_pred_loss = criterion(ori_pred[is_labeled], graph.y[is_labeled], reduction='none')
                mix_pred_loss = criterion(mix_pred[is_labeled], new_y[is_labeled].long(), reduction='none')
                c_loss = criterion(c_pred[is_labeled], graph.y[is_labeled].long(), reduction='none')
    
                cgraph_loss = criterion(c_graph_pred[is_labeled], graph.y[is_labeled].long(), reduction='none')
                loss0 = criterion(ori_pred[is_labeled]*dummy_w, graph.y[is_labeled].long())
                loss1 = criterion(mix_pred[is_labeled]*dummy_w, new_y[is_labeled].long())
                grad_0 = torch.autograd.grad(loss0, dummy_w, create_graph=True)[0]
                grad_1 = torch.autograd.grad(loss1, dummy_w, create_graph=True)[0]
                irm_loss = torch.sum(grad_0 * grad_1)
                vrex_loss = torch.var(torch.FloatTensor([ori_pred_loss.mean(), mix_pred_loss.mean()]).to(device))
                all_losses['irm'] = (all_losses.get('irm', 0) * (n_bw - 1) + irm_loss.item()) / n_bw
                
                pred_loss =  ori_pred_loss.mean() + args.mix * mix_pred_loss.mean() + args.my_vrex * vrex_loss\
                            + args.my_irm * irm_loss +args.caus * cau_loss.mean()+args.c_pred * (cgraph_loss.mean()) 
                
                   
                # compile losses
                batch_loss = pred_loss 
                model_optimizer.zero_grad()
                batch_loss.backward()
                model_optimizer.step()
                all_loss += batch_loss.item()
            all_contrast_loss /= n_bw
            all_loss /= n_bw


            model.eval()
            train_acc = eval_model(model, device, train_loader, evaluator, eval_metric=eval_metric,c_pred=args.c_pred)
            val_acc = eval_model(model, device, valid_loader, evaluator, eval_metric=eval_metric,c_pred=args.c_pred)
            test_acc = eval_model(model,
                                  device,
                                  test_loader,
                                  evaluator,
                                  eval_metric=eval_metric,c_pred=args.c_pred)
            if val_acc <= last_val_acc:
                # select model according to the validation acc,
                #                  after the pretraining stage
                cnt += epoch >= args.pretrain
            else:
                cnt = (cnt + int(epoch >= args.pretrain)) if last_val_acc == 1.0 else 0
                last_train_acc = train_acc
                last_val_acc = val_acc
                last_test_acc = test_acc

                if args.save_model:
                    best_weights = deepcopy(model.state_dict())
            if epoch >= args.pretrain and cnt >= args.early_stopping:
                logger.info("Early Stopping")
                logger.info("+" * 50)
                logger.info("Last: Test_ACC: {:.3f} Train_ACC:{:.3f} Val_ACC:{:.3f} ".format(
                    last_test_acc, last_train_acc, last_val_acc))
                break

            all_info['test_acc'].append(last_test_acc)
            all_info['train_acc'].append(last_train_acc)
            all_info['val_acc'].append(last_val_acc)

            print("      [{:3d}/{:d}]".format(epoch, args.epoch) +
                        "\n       train_ACC: {:.4f} / {:.4f}"
                        "\n       valid_ACC: {:.4f} / {:.4f}"
                        "\n       tests_ACC: {:.4f} / {:.4f}\n".format(
                            train_acc, torch.tensor(all_info['train_acc']).max(),
                            val_acc, torch.tensor(all_info['test_acc']).max(),
                            test_acc, torch.tensor(all_info['val_acc']).max()))
        logger.info("=" * 50)
   
    if args.save_model:
        print("Saving best weights..")
        # model_path = os.path.join('save_my', 'ciga_'+args.dataset) + str(args.r)+'_' +datetime_now+ ".pt"
        model_path = os.path.join('save_my', args.dataset) + str(args.r)+'_' +datetime_now+ ".pt"
        for k, v in best_weights.items():
            best_weights[k] = v.cpu()
        torch.save(best_weights, model_path)
        print("Done..")

    print("\n\n\n")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()