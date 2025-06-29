import torch
import torch.nn as nn
import torch.nn.functional as F



class AdaptiveGraphLearning(nn.Module):
    """
            Adaptive graph learning module.

            Args:
                k (int): Number of top neighbors to keep for sparsity.
                lambda_init (float): Initial value for balancing parameter λ.
                learn_lambda (bool): Whether λ is learnable.
            """
    def __init__(self,k=10, lambda_init=0.5, learn_lambda=True):
        super(AdaptiveGraphLearning, self).__init__()
        self.k = k

        if learn_lambda:
            self.lambda_param = nn.Parameter(torch.tensor(lambda_init), requires_grad=True)
        else:
            self.register_buffer('lambda_param', torch.tensor(lambda_init))

    def forward(self, X, A_raw):
        """
               Forward pass to learn adaptive adjacency matrix.

               Args:
                   X (tensor): Node feature matrix (N x D).
                   A_raw (sparse tensor): Original adjacency matrix.

               Returns:
                   A_final (sparse tensor): Combined adjacency matrix.
                   A_learned (sparse tensor): Learned adjacency matrix from features.
               """
        # Normalize node features for cosine similarity
        X_norm = F.normalize(X, p=2, dim=1)
        # Compute similarity matrix (cosine similarity)
        S = torch.mm(X_norm, X_norm.t())
        # Keep only top-k similarities for sparsity
        A_learned = self._sparse_topk(S, self.k)
        # Sigmoid to map λ to (0,1)
        lambda_value = torch.sigmoid(self.lambda_param)  # 将λ映射到(0,1)区间
        # Combine original adjacency with learned adjacency weighted by λ
        A_final = self._combine_adj(A_raw, A_learned, lambda_value)

        return A_final,A_learned

    def _sparse_topk(self, S, k):
        """
                Keep only top-k entries per row to sparsify similarity matrix.

                Args:
                    S (tensor): Similarity matrix.
                    k (int): Number of neighbors to keep.

                Returns:
                    A_learned_norm (sparse tensor): Row-normalized sparse adjacency matrix.
                """
        N = S.size(0)

        # Get top-k+1 values and indices (including self similarity)
        values, indices = torch.topk(S, k=min(k + 1, N), dim=1)

        # Prepare row indices for COO format
        rows = torch.arange(0, N).view(-1, 1).repeat(1, min(k + 1, N))
        rows = rows.view(-1).to(S.device)
        cols = indices.view(-1)

        # Remove self-loops by filtering where row == col
        mask = rows != cols
        rows = rows[mask]
        cols = cols[mask]
        values = values.view(-1)[mask]

        # Construct sparse adjacency matrix in COO format
        edge_index = torch.stack([rows, cols], dim=0)
        A_learned = torch.sparse_coo_tensor(edge_index, values, (N, N))

        # Convert to dense for row-normalization
        A_dense = A_learned.to_dense()
        row_sum = A_dense.sum(dim=1, keepdim=True) + 1e-6  # avoid division by zero
        A_norm = A_dense / row_sum

        # Convert normalized adjacency back to sparse format
        indices = torch.nonzero(A_norm).t()
        values = A_norm[indices[0], indices[1]]
        A_learned_norm = torch.sparse_coo_tensor(indices, values, (N, N))

        return A_learned_norm

    def _combine_adj(self, A_raw, A_learned, lambda_value):
        """
                Combine original and learned adjacency matrices weighted by λ.

                Args:
                    A_raw (sparse tensor): Original adjacency.
                    A_learned (sparse tensor): Learned adjacency.
                    lambda_value (float): Weighting factor between 0 and 1.

                Returns:
                    A_final (sparse tensor): Combined adjacency matrix.
                """
        A_raw_dense = A_raw.to_dense()
        A_learned_dense = A_learned.to_dense()

        # Weighted sum of original and learned adjacency
        A_final_dense = lambda_value * A_raw_dense + (1 - lambda_value) * A_learned_dense

        # Convert back to sparse format
        indices = torch.nonzero(A_final_dense).t()
        values = A_final_dense[indices[0], indices[1]]
        A_final = torch.sparse_coo_tensor(indices, values, A_raw.size())

        return A_final


