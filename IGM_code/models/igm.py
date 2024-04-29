import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data.batch as DataBatch
from torch_geometric.nn import (ASAPooling, global_add_pool, global_max_pool,
                                global_mean_pool)
from utils.get_subgraph import relabel, split_batch
from utils.mask import clear_masks, set_masks ,my_clear_masks,my_set_masks

from models.conv import GNN_node, GNN_node_Virtualnode
from models.gnn import GNN, LeGNN
from collections import  defaultdict
import random



class IGM(nn.Module):

    def __init__(self,
                 input_dim,
                 out_dim,
                 edge_dim=-1,
                 emb_dim=300,
                 num_layers=5,
                 ratio=0.25,
                 gnn_type='gin',
                 virtual_node=True,
                 residual=False,
                 drop_ratio=0.5,
                 JK="last",
                 graph_pooling="mean",
                 c_dim=-1,
                 c_in="raw",
                 c_rep="rep",
                 c_pool="add",
                 s_rep="rep",
                 sigma_len=3):
        super(IGM, self).__init__()
        ### GNN to generate node embeddings
        if gnn_type.lower() == "le":
            self.gnn_encoder = LeGNN(in_channels=input_dim,
                                     hid_channels=emb_dim,
                                     num_layer=num_layers,
                                     drop_ratio=drop_ratio,
                                     num_classes=out_dim,
                                     edge_dim=edge_dim)
        else:
            if virtual_node:
                self.gnn_encoder = GNN_node_Virtualnode(num_layers,
                                                        emb_dim,
                                                        input_dim=input_dim,
                                                        JK=JK,
                                                        drop_ratio=drop_ratio,
                                                        residual=residual,
                                                        gnn_type=gnn_type,
                                                        edge_dim=edge_dim)
            else:
                self.gnn_encoder = GNN_node(num_layers,
                                            emb_dim,
                                            input_dim=input_dim,
                                            JK=JK,
                                            drop_ratio=drop_ratio,
                                            residual=residual,
                                            gnn_type=gnn_type,
                                            edge_dim=edge_dim)
        self.ratio = ratio

        self.edge_att = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim * 4), nn.ReLU(), nn.Linear(emb_dim * 4, 1),nn.Sigmoid())
        self.s_rep = s_rep
        self.c_rep = c_rep
        self.c_pool = c_pool

        # predictor based on the decoupled subgraph
        self.pred_head = "spu" if s_rep.lower() == "conv" else "inv"
        self.c_dim = emb_dim if c_dim < 0 else c_dim
        self.c_in = c_in
        self.c_input_dim = input_dim if c_in.lower() == "raw" else emb_dim
        self.classifier = GNN(gnn_type=gnn_type,
                              input_dim=self.c_input_dim,
                              num_class=out_dim,
                              num_layer=num_layers,
                              emb_dim=self.c_dim,
                              drop_ratio=drop_ratio,
                              virtual_node=virtual_node,
                              graph_pooling=graph_pooling,
                              residual=residual,
                              JK=JK,
                              pred_head=self.pred_head,
                              edge_dim=edge_dim)


        self.log_sigmas = nn.Parameter(torch.zeros(sigma_len))
        self.log_sigmas.requires_grad_(True)

    def concrete_sample(self,att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = -torch.log(-torch.log(random_noise)) 
            p1 = (torch.log(att_log_logit+1e-9) + random_noise) / temp
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = -torch.log(-torch.log(random_noise))
            p2 = (torch.log(1-att_log_logit+1e-9) + random_noise) / temp
            att_bern = torch.exp(p1)/(torch.exp(p1)+torch.exp(p2))
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern
    # TODO: implementation of my model
    def forward(self, batch, return_data="pred",casual_mix=False,c_pred=False,num_newedge=2,num_label = 3,adapt_select = True, alpha=0.2):
        # obtain the graph embeddings from the featurizer GNN encoder
        h = self.gnn_encoder(batch)
        device = h.device
        row, col = batch.edge_index
        if batch.edge_attr is None:
            batch.edge_attr = torch.ones(row.size(0)).to(device)
        edge_rep = torch.cat([h[row], h[col]], dim=-1)
        pred_edge_weight = self.edge_att(edge_rep).view(-1)

        #TODO: adaptive edge select
        pred_edge_weight = self.concrete_sample(pred_edge_weight, 1, True)


        edge_indices, num_nodes, cum_nodes, num_edges, cum_edges = split_batch(batch)
        casual_edge_indices =defaultdict(list)
        spu_edge_indices = defaultdict(list)
        causal_edge_attr_dict = defaultdict(list)
        spu_edge_attr_dict = defaultdict(list)
        causal_weight_dict = defaultdict(list)
        spu_weight_dict = defaultdict(list)
        causal_index = torch.LongTensor([[], []]).to(device)
        causal_weight = torch.tensor([]).to(device)
        causal_edge_attr = torch.tensor([]).to(device)
        spu_index = torch.LongTensor([[], []]).to(device)
        spu_weights = torch.tensor([]).to(device)
        spu_edge_attr = torch.tensor([]).to(device)
        #TODO: debug edge select
        edge_ratio_list = []
        edge_weight_list = pred_edge_weight.detach().cpu().numpy()
        for edge_index, N, C,y in zip(edge_indices, num_edges, cum_edges,batch.y.detach().cpu().numpy()):
            
            # if edge_index.shape[1]==0:
            #     edge_index = troch.tensor[[0,1][1,0]].to(device)
            edge_attr = batch.edge_attr[C:C + N]
            single_mask = pred_edge_weight[C:C + N]
            if adapt_select:
            # TODO:  adaptive edge select
                idx_reserve = torch.where(single_mask> 0.5)[0]
                idx_drop = torch.where(single_mask<= 0.5)[0]
                
                if len(idx_reserve)/N>self.ratio:
                    n_reserve = int(self.ratio * N)
                    single_mask_detach = pred_edge_weight[C:C + N].detach().cpu().numpy()
                    rank = np.argpartition(-single_mask_detach, n_reserve)
                    idx_reserve, idx_drop = rank[:n_reserve], rank[n_reserve:]
                if len(idx_reserve)/N<(1-self.ratio):
                    n_reserve = int((1-self.ratio) * N)
                    single_mask_detach = pred_edge_weight[C:C + N].detach().cpu().numpy()
                    rank = np.argpartition(-single_mask_detach, n_reserve)
                    idx_reserve, idx_drop = rank[:n_reserve], rank[n_reserve:]
                if len(idx_reserve)<num_newedge:
                    n_reserve = N
                    single_mask_detach = pred_edge_weight[C:C + N].detach().cpu().numpy()
                    idx_reserve, idx_drop = [0,1], [0,1]

                edge_ratio_list.append(len(idx_reserve)/N.detach().cpu().numpy())
            else:
                n_reserve = int(self.ratio * N)
                single_mask_detach = pred_edge_weight[C:C + N].detach().cpu().numpy()
                rank = np.argpartition(-single_mask_detach, n_reserve)
                idx_reserve, idx_drop = rank[:n_reserve], rank[n_reserve:]
            causal_edge_index =edge_index[:, idx_reserve]
            spu_edge_index = edge_index[:, idx_drop]
            casual_edge_indices[y].append(causal_edge_index)
            spu_edge_indices[y].append(spu_edge_index)

            causal_weight_dict[y].append(single_mask[idx_reserve])
            spu_weight_dict[y].append(1 - single_mask[idx_drop])
            causal_edge_attr_dict[y].append(edge_attr[idx_reserve])
            spu_edge_attr_dict[y].append(edge_attr[idx_drop])

            causal_weight = torch.cat([causal_weight, single_mask[idx_reserve]])  
            causal_index = torch.cat([causal_index, edge_index[:, idx_reserve]], dim=1)                     
            causal_edge_attr = torch.cat([causal_edge_attr, edge_attr[idx_reserve]])


            spu_weights = torch.cat([spu_weights,  single_mask[idx_drop]])
            spu_index = torch.cat([spu_index, edge_index[:, idx_drop]], dim=1)                     
            spu_edge_attr = torch.cat([spu_edge_attr, edge_attr[idx_drop]])

        batch_edge_index = torch.LongTensor([[], []]).to(device)
        batch_edge_weight = torch.tensor([]).to(device)
        batch_edge_attr = torch.tensor([]).to(device)
        batch_graph_label = torch.tensor([]).to(device)
        #num_newedge = 2
        labels = list(casual_edge_indices.keys())
        numlabel_batch = len(labels)
        new_batch_label = []
        for i in range(numlabel_batch):
            y = labels[i]
            for casual_part,causal_attr,casual_weight in zip(casual_edge_indices[y],causal_edge_attr_dict[y],causal_weight_dict[y]):

                if numlabel_batch>1:
                    target_label = labels[(i+1)%numlabel_batch]
                    random_spu = random.randint(0, len(spu_edge_indices[target_label])-1)
                    spu_part = spu_edge_indices[target_label][random_spu]
                    spu_attr = spu_edge_attr_dict[target_label][random_spu]
                    spu_weight = spu_weight_dict[target_label][random_spu]
                
                else:
                    random_spu = random.randint(0, len(spu_edge_indices[y])-1)
                    spu_part = spu_edge_indices[y][random_spu]
                    spu_attr = spu_edge_attr_dict[y][random_spu]
                    spu_weight = spu_weight_dict[y][random_spu]
                
                

               
                spu_part_set = torch.unique(spu_part)
                casual_part_set = torch.unique(casual_part)
                new_spu = random.sample(spu_part_set.detach().cpu().tolist(), num_newedge)
                new_cau = random.sample(casual_part_set.detach().cpu().tolist(), num_newedge)
                new_edges = torch.tensor([new_cau, new_spu]).to(device)
                new_attr = batch_edge_attr[:num_newedge].to(device)
                new_weight = 0.1 * torch.ones(num_newedge).reshape(-1).to(device)
                label = torch.tensor(y).reshape(-1).to(device)
                new_graph = torch.cat([new_edges,spu_part,casual_part],dim=1)
                new_graph_weight = torch.cat([new_weight,casual_weight,spu_weight])
                new_edge_attr = torch.cat([new_attr,causal_attr,spu_attr],dim=0)
                batch_edge_index = torch.cat([batch_edge_index, new_graph], dim=1)
                batch_edge_weight = torch.cat([batch_edge_weight,new_graph_weight])
                batch_edge_attr = torch.cat([batch_edge_attr, new_edge_attr])
                batch_graph_label = torch.cat([batch_graph_label,label])

        if self.c_in.lower() == "raw":
            causal_x, causal_index, causal_batch, _ = relabel(batch.x, causal_index, batch.batch)
            spu_x, spu_index, spu_batch, _ = relabel(batch.x, spu_index, batch.batch)
            new_x, new_edge_index, new_batch, _ = relabel(batch.x, batch_edge_index, batch.batch)
        else:
            causal_x, causal_index, causal_batch, _ = relabel(h, causal_index, batch.batch)
            new_x, new_edge_index, new_batch, _ = relabel(h, batch_edge_index, batch.batch)
            spu_x, spu_index, spu_batch, _ = relabel(h, spu_index, batch.batch)
            batch = DataBatch.Batch(batch=batch.batch,
                                edge_index=batch.edge_index,
                                x=h,
                                edge_attr=batch.edge_attr,
                                y=batch.y
                                )
        mix_batch = DataBatch.Batch(batch=new_batch,
                            edge_index=new_edge_index,
                            x=new_x,
                            edge_attr=batch_edge_attr,
                            y=batch_graph_label
                            )
        
        batch_pred, batch_rep = self.classifier(batch, get_rep=True)
        mix_pred, mix_rep = self.classifier(mix_batch, get_rep=True)

        
        causal_graph = DataBatch.Batch(batch=causal_batch,
                                   edge_index=causal_index,
                                   x=causal_x,
                                   edge_attr=causal_edge_attr,
                                   y  = batch.y)
        spu_graph = DataBatch.Batch(batch=spu_batch,
                                   edge_index=spu_index,
                                   x=spu_x,
                                   edge_attr=spu_edge_attr,
                                   y  = batch.y)

        my_set_masks(causal_weight, self.classifier)
        # obtain predictions with the classifier based on \hat{G_c}
        c_batch_pred = self.classifier(causal_graph)
        my_clear_masks(self.classifier)
    

        c_graph_pred = self.classifier(causal_graph)
        # TODO:casual mixup
        if casual_mix:
            
            my_set_masks(causal_weight, self.classifier)
            mixup_x, mixup_y = self.classifier(causal_graph, if_mix=True,num_classes=num_label,alpha = alpha)
            my_clear_masks(self.classifier)
            
            return mixup_x,mixup_y,batch_pred,mix_pred,mix_batch.y,c_batch_pred,edge_ratio_list,edge_weight_list,c_graph_pred,mix_rep
        
        if c_pred and return_data=="pred":
            # return c_batch_pred
            return c_graph_pred
        if return_data=="pred":
            return batch_pred
        
        return batch_pred,mix_pred,mix_batch.y,c_batch_pred
        