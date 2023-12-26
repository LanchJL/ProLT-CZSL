import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .common import MLP
from .gcn import GCN, GCNII
from .word_embedding import load_word_embeddings
import scipy.sparse as sp
from .graph_method import GraphFull

def adj_to_edges(adj):
    # Adj sparse matrix to list of edges
    rows, cols = np.nonzero(adj)
    edges = list(zip(rows.tolist(), cols.tolist()))
    return edges


def edges_to_adj(edges, n):
    # List of edges to Adj sparse matrix
    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype='float32')
    return adj


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GraphFull(nn.Module):
    def __init__(self, dset, args):
        super(GraphFull, self).__init__()
        self.args = args
        self.dset = dset

        self.val_forward = self.val_forward_dotpr
        self.train_forward = self.train_forward_normal
        # Image Embedder
        self.num_attrs, self.num_objs, self.num_pairs = len(dset.attrs), len(dset.objs), len(dset.pairs)
        self.pairs = dset.pairs

        if self.args.train_only:
            train_idx = []
            for current in dset.train_pairs:
                train_idx.append(dset.all_pair2idx[current]+self.num_attrs+self.num_objs)
            self.train_idx = torch.LongTensor(train_idx).to(device)

        all_words = list(self.dset.attrs) + list(self.dset.objs)
        self.displacement = len(all_words)

        self.obj_to_idx = {word: idx for idx, word in enumerate(self.dset.objs)}
        self.attr_to_idx = {word: idx for idx, word in enumerate(self.dset.attrs)}

        if args.graph_init is not None:
            path = args.graph_init
            graph = torch.load(path)
            embeddings = graph['embeddings'].to(device)
            adj = graph['adj']
            self.embeddings = embeddings
        else:
            embeddings = self.init_embeddings(all_words).to(device)
            adj = self.adj_from_pairs()
            self.embeddings = embeddings

        hidden_layers = self.args.gr_emb
        if args.gcn_type == 'gcn':
            self.gcn = GCN(adj, self.embeddings.shape[1], args.emb_dim, hidden_layers)
        else:
            self.gcn = GCNII(adj, self.embeddings.shape[1], args.emb_dim, args.hidden_dim, args.gcn_nlayers, lamda = 0.5, alpha = 0.1, variant = False)

    def init_embeddings(self, all_words):

        def get_compositional_embeddings(embeddings, pairs):
            # Getting compositional embeddings from base embeddings
            composition_embeds = []
            for (attr, obj) in pairs:
                attr_embed = embeddings[self.attr_to_idx[attr]]
                obj_embed = embeddings[self.obj_to_idx[obj]+self.num_attrs]
                composed_embed = (attr_embed + obj_embed) / 2
                composition_embeds.append(composed_embed)
            composition_embeds = torch.stack(composition_embeds)
            print('Compositional Embeddings are ', composition_embeds.shape)
            return composition_embeds

        # init with word embeddings
        embeddings = load_word_embeddings(self.args.emb_init, all_words)

        composition_embeds = get_compositional_embeddings(embeddings, self.pairs)
        full_embeddings = torch.cat([embeddings, composition_embeds], dim=0)

        return full_embeddings


    def update_dict(self, wdict, row,col,data):
        wdict['row'].append(row)
        wdict['col'].append(col)
        wdict['data'].append(data)

    def adj_from_pairs(self):
        def edges_from_pairs(pairs):
            weight_dict = {'data':[],'row':[],'col':[]}


            for i in range(self.displacement):
                self.update_dict(weight_dict,i,i,1.)

            for idx, (attr, obj) in enumerate(pairs):
                attr_idx, obj_idx = self.attr_to_idx[attr], self.obj_to_idx[obj] + self.num_attrs

                self.update_dict(weight_dict, attr_idx, obj_idx, 1.)
                self.update_dict(weight_dict, obj_idx, attr_idx, 1.)

                node_id = idx + self.displacement
                self.update_dict(weight_dict,node_id,node_id,1.)

                self.update_dict(weight_dict, node_id, attr_idx, 1.)
                self.update_dict(weight_dict, node_id, obj_idx, 1.)


                self.update_dict(weight_dict, attr_idx, node_id, 1.)
                self.update_dict(weight_dict, obj_idx, node_id, 1.)

            return weight_dict

        edges = edges_from_pairs(self.pairs)
        adj = sp.csr_matrix((edges['data'], (edges['row'], edges['col'])),
                            shape=(len(self.pairs)+self.displacement, len(self.pairs)+self.displacement))

        return adj



    def train_forward_normal(self, x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]

        if self.args.nlayers:

            img_feats = self.image_embedder(img)
        else:
            img_feats = (img)

        current_embeddings = self.gcn(self.embeddings)
        if self.args.train_only:
            pair_embed = current_embeddings[self.train_idx]
        else:
            pair_embed = current_embeddings[self.num_attrs+self.num_objs:self.num_attrs+self.num_objs+self.num_pairs,:]

        pair_embed = pair_embed.permute(1,0)
        pair_pred = torch.matmul(img_feats, pair_embed)
        loss = F.cross_entropy(pair_pred, pairs)

        return  loss, None

    def val_forward_dotpr(self, x):
        img = x[0]

        if self.args.nlayers:

            img_feats = self.image_embedder(img)
        else:
            img_feats = (img)

        current_embedddings = self.gcn(self.embeddings)

        pair_embeds = current_embedddings[self.num_attrs+self.num_objs:self.num_attrs+self.num_objs+self.num_pairs,:].permute(1,0)

        score = torch.matmul(img_feats, pair_embeds)

        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:,self.dset.all_pair2idx[pair]]

        return None, scores

    def val_forward_distance_fast(self, x):
        img = x[0]

        img_feats = (self.image_embedder(img))
        current_embeddings = self.gcn(self.embeddings)
        pair_embeds = current_embeddings[self.num_attrs+self.num_objs:,:]

        batch_size, pairs, features = img_feats.shape[0], pair_embeds.shape[0], pair_embeds.shape[1]
        img_feats = img_feats[:,None,:].expand(-1, pairs, -1)
        pair_embeds = pair_embeds[None,:,:].expand(batch_size, -1, -1)
        diff = (img_feats - pair_embeds)**2
        score = diff.sum(2) * -1

        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:,self.dset.all_pair2idx[pair]]

        return None, scores

    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
        return loss, pred

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:  # 整除
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        # 参数定义
        self.num_attention_heads = num_attention_heads  # 8
        self.attention_head_size = int(hidden_size / num_attention_heads)  # 16  每个注意力头的维度
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)
        # all_head_size = 128 即等于hidden_size, 一般自注意力输入输出前后维度不变
        # query, key, value 的线性变换（上述公式2）
        self.query = nn.Linear(hidden_size, self.all_head_size)  # 128, 128
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        # dropout
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states):
        # eg: attention_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])  shape=[bs, seqlen]
        # 线性变换
        mixed_query_layer = self.query(hidden_states)  # [bs, hidden_size]
        mixed_key_layer = self.key(hidden_states)  # [bs, hidden_size]
        mixed_value_layer = self.value(hidden_states)  # [bs, hidden_size]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # 计算query与title之间的点积注意力分数，还不是权重（个人认为权重应该是和为1的概率分布）
        attention_scores = torch.matmul(mixed_query_layer, mixed_key_layer.T)        #bs*bs

        # 将注意力转化为概率分布，即注意力权重
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [bs, bs]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # 矩阵相乘，[bs, 8, seqlen, seqlen]*[bs, 8, seqlen, 16] = [bs, 8, seqlen, 16]
        context_layer = torch.matmul(attention_probs, mixed_value_layer)  # [bs, hidden_size]
        return context_layer  # [bs, seqlen, 128] 得到输出

class Complex_embedder_text(nn.Module):
    def __init__(self,dset, args):
        super(Complex_embedder_text, self).__init__()
        '''get word vectors'''
        if args.emb_init == 'word2vec':
            word_vector_dim = 300
        elif args.emb_init == 'glove':
            word_vector_dim = 300
        elif args.emb_init == 'fasttext':
            word_vector_dim = 300
        else:
            word_vector_dim = 600
        self.dset = dset
        self.args = args

        self.attr_embedder = nn.Embedding(len(dset.attrs), word_vector_dim)
        self.obj_embedder = nn.Embedding(len(dset.objs), word_vector_dim)

        pretrained_weight = load_word_embeddings(args.emb_init, dset.attrs)
        self.attr_embedder.weight.data.copy_(pretrained_weight)
        pretrained_weight = load_word_embeddings(args.emb_init, dset.objs)
        self.obj_embedder.weight.data.copy_(pretrained_weight)

        if args.static_inp:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False

        self.attr_projection = nn.Linear(word_vector_dim, args.emb_dim)
        self.obj_projection = nn.Linear(word_vector_dim, args.emb_dim)

    def forward(self,attrs,objs):
        text_complex = torch.zeros((attrs.shape[0], 2, self.args.emb_dim)).cuda()

        attr_ = F.normalize(self.attr_projection(F.leaky_relu(self.attr_embedder(attrs), inplace=True)), dim=1)
        objs_ = F.normalize(self.obj_projection(F.leaky_relu(self.obj_embedder(objs), inplace=True)), dim=1)

        text_complex[:, 0], text_complex[:, 1] = objs_, attr_
        return text_complex

class down_sample_embedder(nn.Module):
    def __init__(self,dset, args):
        super(down_sample_embedder, self).__init__()
        self.dset = dset
        self.args = args
        if self.args.C_y == 'GCN':

            self.num_attrs, self.num_objs, self.num_pairs = len(dset.attrs), len(dset.objs), len(dset.pairs)
            self.pairs = dset.pairs

            graph_method = GraphFull(dset, args)
            self.gcn = graph_method.gcn
            self.embeddings = graph_method.embeddings
            self.train_idx = graph_method.train_idx

        else:
            '''get word vectors'''
            if args.emb_init == 'word2vec':
                word_vector_dim = 300
            elif args.emb_init == 'glove':
                word_vector_dim = 300
            elif args.emb_init == 'fasttext':
                word_vector_dim = 300
            else:
                word_vector_dim = 600

            train_idx = []
            for current in dset.train_pairs:
                train_idx.append(dset.all_pair2idx[current])
            self.train_idx = torch.LongTensor(train_idx).to(device)

            self.attr_embedder = nn.Embedding(len(dset.attrs), word_vector_dim)
            self.obj_embedder = nn.Embedding(len(dset.objs), word_vector_dim)

            pretrained_weight = load_word_embeddings(args.emb_init, dset.attrs)
            self.attr_embedder.weight.data.copy_(pretrained_weight)
            pretrained_weight = load_word_embeddings(args.emb_init, dset.objs)
            self.obj_embedder.weight.data.copy_(pretrained_weight)

            if args.static_inp:
                for param in self.attr_embedder.parameters():
                    param.requires_grad = False
                for param in self.obj_embedder.parameters():
                    param.requires_grad = False

            self.layer_norm = nn.LayerNorm(word_vector_dim*2)
            self.MLP = nn.Sequential(
                nn.Linear(word_vector_dim*2,args.latent_dim),
                nn.LeakyReLU(inplace=True),
                nn.Linear(args.latent_dim,args.emb_dim)
            )
    def forward(self,attrs,objs):
        if self.args.C_y == 'GCN':
            if self.training:
                output = self.gcn(self.embeddings)
                output = output[self.train_idx]
            else:
                output = self.gcn(self.embeddings)
                output = output[self.num_attrs + self.num_objs:self.num_attrs + self.num_objs +
                                                               self.num_pairs, :]
        elif self.args.C_y == 'MLP':
            attrs, objs = self.attr_embedder(attrs), self.obj_embedder(objs)
            inputs = torch.cat([attrs, objs], 1)
            inputs = self.layer_norm(inputs)
            output = self.MLP(inputs)
            output = F.normalize(output, dim=1)
        return output

class img_decoupleing(nn.Module):
    def __init__(self,dset, args):
        super(img_decoupleing, self).__init__()
        layers2 = []
        self.dset = dset
        self.args = args

        self.args.attr_objs_fc_emb = self.args.attr_objs_fc_emb.split(',')
        for dim in self.args.attr_objs_fc_emb:
            dim = int(dim)
            layers2.append(dim)

        self.obj_img_embedder = MLP(dset.feat_dim, int(args.emb_dim), relu=False, num_layers=2,
                                    dropout=self.args.dropout,
                                    norm=self.args.norm, layers=layers2)
        self.attr_img_embedder = MLP(dset.feat_dim, int(args.emb_dim), relu=False, num_layers=2,
                                     dropout=self.args.dropout,
                                     norm=self.args.norm, layers=layers2)
    def forward(self,img):
        attr_img = F.normalize(self.attr_img_embedder(img), dim=1)
        obj_img = F.normalize(self.obj_img_embedder(img), dim=1)
        return attr_img,obj_img

class additional_embbeding(nn.Module):
    def __init__(self,dset, args):
        super(additional_embbeding, self).__init__()
        layers2 = []
        self.dset = dset
        self.args = args

        for dim in self.args.attr_objs_fc_emb:
            dim = int(dim)
            layers2.append(dim)

        self.obj_img_embedder = MLP(dset.feat_dim, int(args.emb_dim), relu=False, num_layers=2,
                                    dropout=self.args.dropout,
                                    norm=self.args.norm, layers=layers2)
        self.attr_img_embedder = MLP(dset.feat_dim, int(args.emb_dim), relu=False, num_layers=2,
                                     dropout=self.args.dropout,
                                     norm=self.args.norm, layers=layers2)
        '''get word vectors'''
        if args.emb_init == 'word2vec':
            word_vector_dim = 300
        elif args.emb_init == 'glove':
            word_vector_dim = 300
        elif args.emb_init == 'fasttext':
            word_vector_dim = 300
        else:
            word_vector_dim = 600
        self.attr_embedder = nn.Embedding(len(dset.attrs), word_vector_dim)
        self.obj_embedder = nn.Embedding(len(dset.objs), word_vector_dim)

        pretrained_weight = load_word_embeddings(args.emb_init, dset.attrs)
        self.attr_embedder.weight.data.copy_(pretrained_weight)
        pretrained_weight = load_word_embeddings(args.emb_init, dset.objs)
        self.obj_embedder.weight.data.copy_(pretrained_weight)

        if args.static_inp:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False

        self.obj_projection = nn.Linear(word_vector_dim, args.emb_dim)
        self.attr_projection = nn.Linear(word_vector_dim, args.emb_dim)

    def forward(self,img,attrs,objs):
        logits = []
        obj_img = F.normalize(self.obj_img_embedder(img), dim=1)
        objs_ = F.normalize(self.obj_projection(F.leaky_relu(self.obj_embedder(objs), inplace=True)), dim=1)

        attr_img = F.normalize(self.attr_img_embedder(img), dim=1)
        attr_ = F.normalize(self.attr_projection(F.leaky_relu(self.attr_embedder(attrs), inplace=True)), dim=1)

        logits.append(torch.matmul(attr_img, attr_.T))
        logits.append(torch.matmul(obj_img, objs_.T))

        return logits