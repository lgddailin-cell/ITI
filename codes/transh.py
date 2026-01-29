# model_transH.py
from __future__ import absolute_import, division, print_function

import os
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from dataloader import TestDataset
# 注意：为了与原 KGEModel 接口完全一致，我把类命名为 KGEModel，
# 并保留了 __init__ 的参数顺序与默认值（尽量与您原来代码一致）。
# 本文件实现 TransH 与 ITI_TransH 两个行为分支。

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 house_dim, house_num, housd_num, thred,
                 double_entity_embedding=False, double_relation_embedding=False):
        """
        signature kept exactly same as your original KGEModel
        model_name: 'TransH' or 'ITI_TransH'
        """
        super(KGEModel, self).__init__()

        # keep attributes for compatibility
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = int(hidden_dim)   # for TransH we use hidden_dim as embedding size
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)

        # keep redundant params to match signature (not used in TransH core)
        self.house_dim = house_dim
        self.house_num = house_num
        self.housd_num = housd_num
        self.thred = thred

        # embeddings (TransH uses vectors)
        self.entity_embedding = nn.Parameter(torch.zeros(self.nentity, self.hidden_dim))
        self.relation_embedding = nn.Parameter(torch.zeros(self.nrelation, self.hidden_dim))
        # TransH norm vectors (relation-specific normal vectors for hyperplane)
        self.relation_norm = nn.Parameter(torch.zeros(self.nrelation, self.hidden_dim))

        # initialize embeddings
        emb_bound = 6.0 / float(self.hidden_dim) if self.hidden_dim > 0 else 0.1
        nn.init.uniform_(self.entity_embedding, -emb_bound, emb_bound)
        nn.init.uniform_(self.relation_embedding, -emb_bound, emb_bound)
        nn.init.uniform_(self.relation_norm, -emb_bound, emb_bound)

        # ITI-specific components: only created if model_name == 'ITI_TransH'
        self.use_type = (self.model_name == 'ITI_TransH')
        if self.use_type:
            # Keep defaults aligned to your previous code style
            # You can change type_dim to any desired value
            self.ntype = 571   # default number of types (kept from your earlier code)
            self.type_dim = min(64, self.hidden_dim)  # type embedding dim (tunable)
            # type embeddings
            self.type_embedding = nn.Parameter(torch.zeros(self.ntype, self.type_dim))
            nn.init.uniform_(self.type_embedding, -1.0/self.type_dim, 1.0/self.type_dim)
            # gating/projection: map type embedding into hidden_dim space to perturb entity
            self.type_gate = nn.Parameter(torch.zeros(self.type_dim, self.hidden_dim))
            nn.init.xavier_uniform_(self.type_gate)

            # optional small MLP to transform type_repr into additive vector
            self.type_mlp = nn.Sequential(
                nn.Linear(self.type_dim, self.type_dim),
                nn.ReLU(),
                nn.Linear(self.type_dim, self.hidden_dim)
            )

            # load entity->type file if exists (path can be modified by user)
            # file expected format: each line "entity_id type_id"
            dict_path = "D:/codes/data/YAGO3-10"
            path_entity2type = os.path.join(dict_path, 'entity2type_id.txt')
            self.entity_type = torch.zeros(self.nentity, dtype=torch.long)  # default 0
            if os.path.exists(path_entity2type):
                with open(path_entity2type, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            e_id = int(parts[0])
                            t_id = int(parts[1])
                            if 0 <= e_id < self.nentity and 0 <= t_id < self.ntype:
                                self.entity_type[e_id] = t_id
            else:
                # file not found -> keep zeros, user should provide or adjust path
                logging.warning("entity2type_id.txt not found at %s, using default type 0 for all entities." % path_entity2type)

        # sanity check on model_name
        if self.model_name not in ['TransH', 'ITI_TransH']:
            raise ValueError('model %s not supported by model_transH.KGEModel' % self.model_name)

    # ---------------------------
    # helper: project vector onto relation hyperplane (TransH)
    # ---------------------------
    def _project_to_relation(self, e, n):
        """
        e: (..., hidden_dim)
        n: (..., hidden_dim)  -- normal vector(s)
        returns: projected vector of same shape as e
        TransH projection: e_perp = e - n * (n^T e)
        with normalized n
        """
        # normalize n along last dim
        n_norm = F.normalize(n, p=2, dim=-1)
        # inner product: (...,1)
        inner = (e * n_norm).sum(dim=-1, keepdim=True)
        proj = e - inner * n_norm
        return proj

    # ---------------------------
    # TransH score for a batch
    # ---------------------------
    def score_TransH(self, head, relation, tail):
        """
        head, relation, tail: tensors with shape (B, D) or (B,Nneg,D)
        return score: (...,) or (B,Nneg)
        We'll compute negative L2 distance and map to same scoring as other models:
        score = gamma - || h_perp + r - t_perp ||_2
        """
        # relation normals: must have same leading dims as relation
        # ensure shapes compatible
        # project head/tail onto relation specific hyperplane
        h_proj = self._project_to_relation(head, relation)
        t_proj = self._project_to_relation(tail, relation)
        # distance
        diff = h_proj + relation - t_proj
        # L2 norm over last dim
        dist = torch.norm(diff, p=2, dim=-1)
        score = self.gamma.item() - dist
        return score

    # ---------------------------
    # ITI + TransH scoring: incorporate type embeddings
    # ---------------------------
    def score_ITI_TransH(self, head, relation, tail, head_idx=None, tail_idx=None):
        """
        head, relation, tail: tensors (..., D)
        head_idx, tail_idx: index tensors to lookup types (optional).
        Approach:
            - lookup type embedding for head/tail
            - transform type embedding via type_mlp -> delta vector
            - add delta to entity embedding (residual style)
            - then apply TransH projection + scoring
        """
        # lookup types if provided; otherwise, if entity_type defined, use it
        if head_idx is None or tail_idx is None:
            # we assume head/tail are of shape (B, D) and have been selected using indices before
            # cannot map types without indices; fallback: use zeros
            head_type_emb = None
            tail_type_emb = None
        else:
            # head_idx/tail_idx may be flattened indices of shape (B,) or (B,Nneg)
            flat_head_idx = head_idx.view(-1)
            flat_tail_idx = tail_idx.view(-1)
            head_type_emb = self.type_embedding[flat_head_idx].view(*head_idx.shape, -1)
            tail_type_emb = self.type_embedding[flat_tail_idx].view(*tail_idx.shape, -1)

        # compute deltas via mlp
        if head_type_emb is not None:
            delta_h = self.type_mlp(head_type_emb)  # (..., hidden_dim)
            delta_t = self.type_mlp(tail_type_emb)
            head = head + delta_h
            tail = tail + delta_t
        else:
            # no type info: do nothing
            pass

        # proceed with TransH scoring
        return self.score_TransH(head, relation, tail)

    # ---------------------------
    # forward: keep original calling convention: forward(sample, mode)
    # sample in ('head-batch' or 'tail-batch') is (positive_part, negative_part)
    # ---------------------------
    def forward(self, sample, mode='single'):
        """
        This forward function aims to be compatible with your training loop:
        - mode == 'head-batch' : sample = (tail_part, head_part)
        - mode == 'tail-batch' : sample = (head_part, tail_part)
        We will select appropriate embeddings and compute scores.
        """

        if mode == 'head-batch':
            tail_part, head_part = sample  # tail_part: (B,3) pos triples ; head_part: (B, Nneg)
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            # head: select negative candidates flattened then reshape
            head = torch.index_select(self.entity_embedding, dim=0, index=head_part.view(-1)).view(batch_size, negative_sample_size, self.hidden_dim)
            # relation: use relation from tail_part[:,1]
            relation = torch.index_select(self.relation_embedding, dim=0, index=tail_part[:,1]).unsqueeze(1)  # (B,1,D)
            # tail: positive tail from tail_part[:,2]
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part[:,2]).unsqueeze(1)  # (B,1,D)

            # For TransH, relation normals must align: pick corresponding relation normal vector
            relation_norm = torch.index_select(self.relation_norm, dim=0, index=tail_part[:,1]).unsqueeze(1)

            # Expand relation to match shapes
            # relation (B,1,D)  head (B,Nneg,D), tail (B,1,D)
            # We'll broadcast relation to head/tail shapes when necessary.

            # prepare indices for types if using ITI
            if self.use_type:
                # head_part contains negative head ids; flatten to map types
                head_idx = head_part  # (B, Nneg)
                tail_idx = tail_part[:,2].unsqueeze(1).expand(-1, negative_sample_size)  # (B, Nneg)
                # prepare relation vector shaped to (B, Nneg, D)
                rel_expand = relation.expand(-1, negative_sample_size, -1)
                rel_norm_expand = relation_norm.expand(-1, negative_sample_size, -1)
                # compute scores with ITI branch
                score = self.score_ITI_TransH(head, rel_expand, tail.expand(-1, negative_sample_size, -1),
                                              head_idx, tail_idx)
            else:
                # expand relation to match head/tail dims
                rel_expand = relation.expand(-1, negative_sample_size, -1)
                score = self.score_TransH(head, rel_expand, tail.expand(-1, negative_sample_size, -1))

            return score  # shape (B, Nneg)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:,0]).unsqueeze(1)  # (B,1,D)
            relation = torch.index_select(self.relation_embedding, dim=0, index=head_part[:,1]).unsqueeze(1)  # (B,1,D)
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, self.hidden_dim)

            relation_norm = torch.index_select(self.relation_norm, dim=0, index=head_part[:,1]).unsqueeze(1)

            if self.use_type:
                head_idx = head_part[:,0].unsqueeze(1).expand(-1, negative_sample_size)
                tail_idx = tail_part
                rel_expand = relation.expand(-1, negative_sample_size, -1)
                score = self.score_ITI_TransH(head.expand(-1, negative_sample_size, -1), rel_expand, tail, head_idx, tail_idx)
            else:
                rel_expand = relation.expand(-1, negative_sample_size, -1)
                score = self.score_TransH(head.expand(-1, negative_sample_size, -1), rel_expand, tail)

            return score

        else:
            raise ValueError('mode %s not supported' % mode)

    # ---------------------------
    # Training step & testing step: kept API consistent with your original KGEModel
    # ---------------------------
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        """
        A single train step. Apply back-propation and return the loss
        This implementation mirrors the logic in your original file.
        """
        model.train()
        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        if mode == 'head-batch':
            pos_part = positive_sample[:, 0].unsqueeze(dim=1)
        else:
            pos_part = positive_sample[:, 2].unsqueeze(dim=1)

        positive_score = model((positive_sample, pos_part), mode=mode)
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:
            regularization = args.regularization * (
                model.entity_embedding.norm(dim=1, p=2).mean()
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()
        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }
        return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        """
        Evaluate the model on test or valid datasets — kept compatible with your original code.
        """
        model.eval()

        if args.countries:
            sample = list()
            y_true = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)
            auc_pr = average_precision_score(y_true, y_score)
            metrics = {'auc_pr': auc_pr}
            return metrics

        # Otherwise standard filtered metrics
        test_dataloader_head = DataLoader(
            TestDataset(test_triples, all_true_triples, args.nentity, args.nrelation, 'head-batch'),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )
        test_dataloader_tail = DataLoader(
            TestDataset(test_triples, all_true_triples, args.nentity, args.nrelation, 'tail-batch'),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )
        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        logs = []
        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()
                        filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)
                    score = model((positive_sample, negative_sample), mode)
                    score = score + filter_bias

                    # compute ranking (vectorized)
                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    else:
                        positive_arg = positive_sample[:, 2]

                    argsort = torch.argsort(score, dim=1, descending=True)
                    # find ranks
                    # create mask comparing argsort positions with positive_arg
                    # convert to ranks by finding index
                    # This is slightly memory-heavy but keeps interface same
                    batch_indices = torch.arange(batch_size).unsqueeze(1).to(score.device)
                    # find where positive_arg located
                    # build boolean mask: (B, Nneg) where True means match
                    mask = argsort == positive_arg.unsqueeze(1)
                    # get ranks: find index of True
                    nonzero = mask.nonzero(as_tuple=False)
                    # map to rank per sample; careful since nonzero returns flattened indices
                    rank_tensor = torch.zeros(batch_size, dtype=torch.long)
                    for (i, j) in nonzero:
                        rank_tensor[i] = j + 1  # ranks start at 1

                    # compute metrics for batch
                    for i in range(batch_size):
                        rank = max(1, int(rank_tensor[i].item()))
                        logs.append({
                            'MRR': 1.0 / rank,
                            'MR': float(rank),
                            'HITS@1': 1.0 if rank <= 1 else 0.0,
                            'HITS@3': 1.0 if rank <= 3 else 0.0,
                            'HITS@10': 1.0 if rank <= 10 else 0.0,
                        })

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))
                    step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        return metrics
