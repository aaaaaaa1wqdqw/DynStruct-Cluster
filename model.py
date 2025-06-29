from torch.nn.parameter import Parameter

import torch
import torch.nn.functional as F

from torch import nn

from AE import AE
from GNN import GNNLayer

class DyCluster(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1,pretrain_path=None,temperature=0.1):
        super(DyCluster, self).__init__()
        """
               Deep clustering model integrating autoencoder and graph neural network (GNN).

               Args:
                   n_enc_*, n_dec_*: Layer sizes for encoder and decoder in AE.
                   n_input: Input feature dimension.
                   n_z: Dimension of latent space.
                   n_clusters: Number of clusters.
                   v: Degree of freedom parameter for Student's t-distribution.
                   pretrain_path: Path to pretrained AE weights.
                   temperature: Temperature parameter for prototype contrastive loss.
               """
        #Initialize autoencoder and load pretrained weights
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(pretrain_path, map_location='cpu'))

        # Define GNN layers to extract structural features
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)



        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v
        # Temperature parameter for contrastive loss
        self.temperature = nn.Parameter(torch.tensor(temperature))
    def forward(self, x, adj,sigma):
        # Autoencoder reconstruction and latent representation
        x_bar, tra1, tra2, tra3, z = self.ae(x)
        # GNN layers with skip connections weighted by sigma
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1 - sigma) * h + sigma * tra1, adj)
        h = self.gnn_3((1 - sigma) * h + sigma * tra2, adj)
        h = self.gnn_4((1 - sigma) * h + sigma * tra3, adj)
        h = self.gnn_5((1 - sigma) * h + sigma * z, adj, active=False)
        # Softmax for predicted cluster assignments
        predict = F.softmax(h, dim=1)
        h  = h.detach()
        # Compute soft assignments q based on Student's t-distribution with cluster centers
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x_bar, q, predict, z , h

    def initialize_prototypes(self, embeddings, pseudo_labels):
        """
               Initialize cluster prototypes as normalized mean embeddings of pseudo-labeled samples.

               Args:
                   embeddings: Node embeddings tensor.
                   pseudo_labels: Pseudo cluster assignments.
               """
        n_clusters = self.cluster_layer.shape[0]
        n_dim = embeddings.shape[1]

        self.prototypes = torch.zeros(n_clusters, n_dim).to(embeddings.device)

        for k in range(n_clusters):
            indices = (pseudo_labels == k).nonzero(as_tuple=True)[0]
            if indices.shape[0] > 0:
                self.prototypes[k] = embeddings[indices].mean(dim=0)
            else:
                # Random initialization if no samples in cluster
                self.prototypes[k] = torch.randn(n_dim).to(embeddings.device)
        # Normalize prototypes to unit vectors
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

    def update_prototypes(self, embeddings, pseudo_labels):
        """
                Update cluster prototypes during training.

                Args:
                    embeddings: Current embeddings.
                    pseudo_labels: Current pseudo cluster assignments.
                """
        n_clusters = self.cluster_layer.shape[0]
        n_dim = embeddings.shape[1]

        new_prototypes = torch.zeros(n_clusters, n_dim).to(embeddings.device)

        for k in range(n_clusters):
            indices = (pseudo_labels == k).nonzero(as_tuple=True)[0]
            if indices.shape[0] > 0:
                new_prototypes[k] = embeddings[indices].mean(dim=0)
            elif self.prototypes is not None:
                # Keep old prototype if no samples for cluster
                new_prototypes[k] = self.prototypes[k]
            else:
                new_prototypes[k] = torch.randn(n_dim).to(embeddings.device)

        self.prototypes = F.normalize(new_prototypes, p=2, dim=1)

    def prototype_loss(self, embeddings, labels):
        """
                Compute prototype contrastive loss to align embeddings with prototypes.

                Args:
                    embeddings: Normalized embeddings.
                    labels: Pseudo cluster labels.

                Returns:
                    loss value (scalar tensor)
                """
        if self.prototypes is None:
            return torch.tensor(0.0).to(embeddings.device)

        embeddings = F.normalize(embeddings, p=2, dim=1)
        # Similarity matrix between embeddings and prototypes scaled by temperature
        sim_matrix = torch.mm(embeddings, self.prototypes.t()) / self.temperature
        # Similarity of positive pairs
        pos_sim = torch.gather(sim_matrix, 1, labels.view(-1, 1)).squeeze()
        exp_sim = torch.exp(sim_matrix)
        exp_sum = exp_sim.sum(dim=1)
        exp_pos = torch.exp(pos_sim)
        # Contrastive loss (negative log likelihood of positive similarity normalized by all similarities)
        loss = -torch.log(exp_pos / exp_sum).mean()
        return loss


    def prototype_kl_loss(self, prototypes_a, prototypes_b):
        """
               Compute symmetric KL divergence loss between two sets of prototypes.

               Args:
                   prototypes_a, prototypes_b: Prototype embeddings from two branches.

               Returns:
                   Symmetric KL divergence scalar loss.
               """
        pa = F.softmax(prototypes_a, dim=1)
        pb = F.softmax(prototypes_b, dim=1)

        return F.kl_div(pa.log(), pb.detach(), reduction='batchmean') + \
               F.kl_div(pb.log(), pa.detach(), reduction='batchmean')
