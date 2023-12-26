import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import MLP
import numpy as np
from .Embeddings import down_sample_embedder,img_decoupleing,additional_embbeding

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CZSL(nn.Module):
    def __init__(self, dset, args):
        super(CZSL, self).__init__()
        self.args = args
        self.dset = dset
        def get_all_ids(relevant_pairs):
            # Precompute validation pairs
            attrs, objs = zip(*relevant_pairs)
            attrs = [dset.attr2idx[attr] for attr in attrs]
            objs = [dset.obj2idx[obj] for obj in objs]
            pairs = [a for a in range(len(relevant_pairs))]
            attrs = torch.LongTensor(attrs).to(device)
            objs = torch.LongTensor(objs).to(device)
            pairs = torch.LongTensor(pairs).to(device)
            return attrs, objs, pairs
        # Validation
        self.val_attrs, self.val_objs, self.val_pairs = get_all_ids(self.dset.pairs)

        # for indivual projections
        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(device), \
                                          torch.arange(len(self.dset.objs)).long().to(device)
        self.factor = 2
        self.scale_pairs = self.args.cosine_scale_pairs
        self.scale_c = self.args.cosine_scale_components

        if dset.open_world:
            self.train_forward = self.train_forward_open
            self.known_pairs = dset.train_pairs
            seen_pair_set = set(self.known_pairs)
            mask = [1 if pair in seen_pair_set else 0 for pair in dset.pairs]
            self.seen_mask = torch.BoolTensor(mask).to(device) * 1.
            self.activated = False
            # Init feasibility-related variables
            self.attrs = dset.attrs
            self.objs = dset.objs
            self.possible_pairs = dset.pairs
            self.validation_pairs = dset.val_pairs
            self.feasibility_margin = (1-self.seen_mask).float()
            self.epoch_max_margin = self.args.epoch_max_margin
            self.cosine_margin_factor = -args.margin
            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for (a, o) in self.known_pairs:
                self.obj_by_attrs_train[a].append(o)
            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for (a, o) in self.known_pairs:
                self.attrs_by_obj_train[o].append(a)
        else:
            self.train_forward = self.train_forward_closed

        # Precompute training compositions
        if args.train_only:
            self.train_attrs, self.train_objs, self.train_pairs = get_all_ids(self.dset.train_pairs)
        else:
            self.train_attrs, self.train_objs, self.train_pairs = self.val_attrs, self.val_objs, self.val_pairs
        try:
            self.args.fc_emb = self.args.fc_emb.split(',')
        except:
            self.args.fc_emb = [self.args.fc_emb]

        self.num_attrs, self.num_objs, self.num_pairs = len(dset.attrs), len(dset.objs), len(dset.pairs)
        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(device), \
                                          torch.arange(len(self.dset.objs)).long().to(device)
        self.pairs = dset.pairs

        layers = []
        for a in self.args.fc_emb:
            a = int(a)
            layers.append(a)

        if args.nlayers:
            self.image_embedder = MLP(dset.feat_dim, int(args.emb_dim), relu=args.relu, num_layers=args.nlayers,
                                      dropout=self.args.dropout,
                                      norm=self.args.norm, layers=layers)
        # Fixed
        self.composition = args.composition

        if self.args.train_only:
            train_idx = []
            for current in dset.train_pairs:
                train_idx.append(dset.all_pair2idx[current])
            self.train_idx = torch.LongTensor(train_idx).to(device)

        self.label_smooth = np.zeros((self.num_pairs, self.num_pairs))
        pairs_array = np.array(self.pairs)
        equal_pairs_1 = pairs_array[:, 1].reshape(-1, 1) == pairs_array[:, 1].reshape(1, -1)
        equal_pairs_0 = pairs_array[:, 0].reshape(-1, 1) == pairs_array[:, 0].reshape(1, -1)
        self.label_smooth = np.where(equal_pairs_1, self.label_smooth + 2, self.label_smooth)
        self.label_smooth = np.where(equal_pairs_0, self.label_smooth + 1, self.label_smooth)
        self.label_smooth = torch.from_numpy(self.label_smooth).to(device)

        if args.train_only:
            self.label_smooth = self.label_smooth[:, self.train_idx]
            self.label_smooth = self.label_smooth[self.train_idx, :]

        K_1 = (self.label_smooth == 1).sum(dim=1)
        K_2 = (self.label_smooth == 2).sum(dim=1)
        K = K_1+2*K_2
        self.epi = args.smooth_factor
        template = torch.ones_like(self.label_smooth).to(device)/K
        template = template*self.epi
        self.label_smooth = self.label_smooth*template
        for i in range(self.label_smooth.shape[0]):
            self.label_smooth[i,i] = 1-(self.epi)


        self.C_y = down_sample_embedder(self.dset,self.args)
        self.img_decouple = img_decoupleing(self.dset,self.args)

        if args.IC:
            self.IC_logits = additional_embbeding(self.dset,self.args)

        self.obj2pairs = self.create_obj_pairs()
        self.attr2pairs = self.create_attr_pairs()
        self.obj2pairs_test = self.create_obj_pairs()
        self.attr2pairs_test = self.create_attr_pairs()
        if args.train_only:
            self.obj2pairs = self.obj2pairs[:,self.train_idx].cuda()
            self.attr2pairs = self.attr2pairs[:,self.train_idx].cuda()

            self.obj2pairs_test[:,self.train_idx] = self.obj2pairs_test[:,self.train_idx]*0.2
            self.attr2pairs_test[:,self.train_idx] = self.attr2pairs_test[:,self.train_idx]*0.2
        self.p_log = dict()
    def cross_entropy(self, logits, label):
        logits = F.log_softmax(logits, dim=-1)
        loss = -(logits * label).sum(-1).mean()
        return loss

    def create_obj_pairs(self):
        obj_matrix = torch.zeros(self.num_objs,self.num_pairs)
        for i in range(self.num_objs):
            for j in range(self.num_pairs):
                if self.dset.objs[i] == self.pairs[j][1]:
                    obj_matrix[i,j] = 1
        return obj_matrix
    def create_attr_pairs(self):
        obj_matrix = torch.zeros(self.num_attrs,self.num_pairs)
        for i in range(self.num_attrs):
            for j in range(self.num_pairs):
                if self.dset.attrs[i] == self.pairs[j][0]:
                    obj_matrix[i,j] = 1
        return obj_matrix

    def freeze_representations(self):
        print('Freezing representations')
        for param in self.image_embedder.parameters():
            param.requires_grad = False
        for param in self.attr_embedder.parameters():
            param.requires_grad = False
        for param in self.obj_embedder.parameters():
            param.requires_grad = False

    def C_y_logits(self,img,attr,objs):
        text = self.C_y(attr,objs).permute(1,0)
        logits = torch.mm(img,text)
        return logits

    def val_forward(self, x):
        img,d_img = x[0],x[-1]
        if self.args.update_features == False:
            img = F.avg_pool2d(img, kernel_size=7).view(-1, self.dset.feat_dim)
            d_img = img
        if self.args.nlayers:
            img_feats = self.image_embedder(img)
        else:
            img_feats = img

        if self.args.C_y == 'GCN':
            img_feats_normed = img_feats
        else:
            img_feats_normed = F.normalize(img_feats, dim=1)

        if self.args.C_y:
            logits_ds = self.C_y_logits(img_feats_normed,self.val_attrs,self.val_objs)
        else:
            logits_ds = 0

        #Code about Sec.3.5 will be added later.
        logits_a = self.IC_logits(d_img,self.val_attrs,self.val_objs)

        logits_a[0] = torch.log(F.softmax(logits_a[0], dim=-1))
        logits_a[1] = torch.log(F.softmax(logits_a[1], dim=-1))

        if self.args.if_ds == False:
            score = (logits_a[0] + logits_a[1])
        else:
            score = logits_ds + self.args.eta * (logits_a[0] +  logits_a[1])
        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:, self.dset.all_pair2idx[pair]]

        return None, scores

    def train_forward_closed(self, x):
        img, attrs, objs, pairs, d_img = x[0], x[1], x[2], x[3],x[-1]

        img = F.avg_pool2d(img, kernel_size=7).view(-1, self.dset.feat_dim)
        smoothed_labels = self.label_smooth[pairs]

        if self.args.nlayers:
            img_feats = self.image_embedder(img)
        else:
            img_feats = img

        if self.args.C_y == 'GCN':
            img_feats_normed = img_feats
        else:
            img_feats_normed = F.normalize(img_feats, dim=1)

        '''C_y'''
        if self.args.if_ds:
            logits_adjust = self.args.eta*(self.p_log['attr'] + self.p_log['objs'])
            logits_ds = self.C_y_logits(img_feats_normed,self.train_attrs,self.train_objs)
            loss_ds = self.cross_entropy(self.scale_c*(logits_ds)+logits_adjust.detach(),smoothed_labels)
        else:
            loss_ds = 0
        '''C_s,C_o'''
        if self.args.IC:
            logits_a = self.IC_logits(img,self.uniq_attrs,self.uniq_objs)
            loss_attr = F.cross_entropy(self.scale_c*logits_a[0],attrs)
            loss_objs = F.cross_entropy(self.scale_c*logits_a[1],objs)
        else:
            loss_attr,loss_objs = 0,0
            logits_a = 0

        loss_total = loss_ds+loss_attr+loss_objs


        pred = []
        pred.append(logits_a[0].detach()@self.attr2pairs)
        pred.append(logits_a[1].detach()@self.obj2pairs)

        mask_a = torch.zeros_like(pred[0])
        mask_o = torch.zeros_like(pred[0])
        mask_a[:,pairs] = 1
        mask_o[:,pairs] = 1
        pred[0] = pred[0] * mask_a
        pred[1] = pred[1] * mask_o
        return loss_total, pred

    def freeze_model(self,model,fine_tune):
        for p in model.parameters():
            p.require_grad = fine_tune

    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
        return loss, pred



