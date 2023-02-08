import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedProd2vec(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super().__init__()
        self.norm_adj = norm_adj
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        self.emb_size = args.embed_size
        self.batch_size = args.batch_size
        self.plot_gradients = args.plot_gradients
        self.cf_pen = args.cf_pen
        self.cf_distance = args.cf_distance
        self.cf_loss = 0
        self.l2_pen = args.l2_pen
        self.embedding_dict, self.weight_dict = self.init_weight()

    def init_weight(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict(
            {'user_emb': nn.Parameter(initializer(torch.empty(self.n_user, self.emb_size))),
             'item_emb': nn.Parameter(initializer(torch.empty(self.n_item, self.emb_size)))
             }
        )
        weight_dict = nn.ParameterDict()
        weight_dict.update({'alpha': nn.Parameter(initializer(torch.empty(1)))})
        weight_dict.update({'global_bias': nn.Parameter(initializer(torch.empty(1)))})
        weight_dict.update({'user_b': nn.Parameter(initializer(torch.zeros([self.num_users])))})
        weight_dict.update({'prod_b': nn.Parameter(initializer(torch.zeros([self.num_products])))})

        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def create_loss(self, users, items, logits, prediction):
        # Sigmoid loss between the logits and labels
        log_loss = torch.mean(
            nn.cross_entropy(logits=logits, labels=self.items))

        # Adding the regularizer term on user vct and prod vct and their bias terms
        reg_term = self.l2_pen * (torch.norm(self.user_embeddings, 2) + torch.norm(self.product_embeddings, 2))
        reg_term_biases = self.l2_pen * (torch.norm(self.weight_dict['prod_b'], 2) + torch.norm(self.weight_dict['user_b'], 2))
        factual_loss = log_loss + reg_term + reg_term_biases

        # Adding the counter-factual loss
        loss = factual_loss + (self.cf_pen * self.cf_loss)  # Imbalance loss
        mse_loss = torch.nn.MSELoss(labels=items, predictions=prediction)
        return loss, mse_loss, factual_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings)

    def forward(self, users, items):
        users_2 = users+users
        user_emb = self.embedding_dict['user_emb'][users_2]
        item_emb = self.embedding_dict['item_emb'][items]

        emb_logits = self.weight_dict['alpha'] * torch.unsqueeze(torch.sum(torch.mul(user_emb, item_emb), 1),
                                                                 dim=1)
        logits = torch.unsqueeze(torch.add(self.weight_dict['user_b'][users_2], self.weight_dict['prod_b'][items]), 1) + \
                 self.weight_dict['global_bias']
        logits = emb_logits + logits

        prediction = nn.sigmoid(logits)
        return prediction, logits
