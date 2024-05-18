# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn
import numpy as np
import cupy as cp
from tqdm import tqdm

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048

    # do_kv_compress: bool = False
    dim_compress: int = 1024
    kv_compress_layers: List[int] = field(default_factory=list)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def rope_q(
    xq: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to ONLY query 'xq' using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        xq_out (torch.Tensor): modified query tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

def cupy_decompose_matrix(matrix, target_dim):
    # Assuming matrix is a PyTorch tensor on a GPU, convert it to a CuPy array
    # using DLPack to avoid data copying.
    print('target_dim:', target_dim)
    matrix_cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(matrix.type(torch.float64)))

    # Perform SVD using CuPy, utilizing GPU acceleration.
    U_cp, S_cp, V_cp = cp.linalg.svd(matrix_cp, full_matrices=True)

    # Reduce the dimensions of U, S, and V matrices as per target_dim.
    U_reduced_cp = U_cp[:, :target_dim]
    S_reduced_cp = S_cp[:target_dim]
    V_reduced_cp = V_cp[:target_dim, :]

    # Convert the reduced matrices back to PyTorch tensors, using DLPack
    # for efficient conversion between CuPy and PyTorch.
    U_reduced = torch.utils.dlpack.from_dlpack(U_reduced_cp.toDlpack())
    S_reduced = torch.utils.dlpack.from_dlpack(S_reduced_cp.toDlpack())
    V_reduced = torch.utils.dlpack.from_dlpack(V_reduced_cp.toDlpack())
    print('U_reduced shape:', U_reduced.shape)
    print('S_reduced shape:', S_reduced.shape)
    print('V_reduced shape:', V_reduced.shape)

    # Construct the diagonal matrix from S in PyTorch.
    S_reduced_diag = torch.diag(S_reduced)

    # Calcuate the reconstruction error
    matrix_reconstructed = torch.mm(torch.mm(U_reduced, S_reduced_diag), V_reduced)
    error = torch.norm(matrix - matrix_reconstructed, p='fro')
    print("Frobenius norm of the reconstruction error:", error)
    relative_error = error / torch.norm(matrix, p='fro')
    print("Relative reconstruction error:", relative_error)

    # Return the reduced U, diagonal S, and transposed V to match the original format.
    return U_reduced.type(matrix.dtype), S_reduced_diag.type(matrix.dtype), V_reduced.type(matrix.dtype)

def numpy_decompose_matrix(matrix, target_dim):
    # Assuming matrix is a PyTorch tensor on a GPU, move it to CPU and convert to a NumPy array.
    matrix_np = matrix.float().cpu().numpy()
    print('Matrix Shape:', matrix_np.shape)

    # Perform SVD using NumPy.
    U_np, S_np, V_np = np.linalg.svd(matrix_np, full_matrices=False)

    # Reduce the dimensions of U, S, and V matrices as per target_dim.
    U_reduced_np = U_np[:, :target_dim]
    S_reduced_np = S_np[:target_dim]
    V_reduced_np = V_np[:target_dim, :]

    # Convert the reduced matrices back to PyTorch tensors.
    U_reduced = torch.tensor(U_reduced_np, dtype=matrix.dtype).to(matrix.device)
    S_reduced = torch.tensor(S_reduced_np, dtype=matrix.dtype).to(matrix.device)
    V_reduced = torch.tensor(V_reduced_np, dtype=matrix.dtype).to(matrix.device)

    # Construct the diagonal matrix from S in PyTorch.
    S_reduced_diag = torch.diag(S_reduced)

    # Calcuate the reconstruction error
    matrix_reconstructed = torch.mm(torch.mm(U_reduced, S_reduced_diag), V_reduced)
    error = torch.norm(matrix - matrix_reconstructed, p='fro')
    print("Frobenius norm of the reconstruction error:", error)
    relative_error = error / torch.norm(matrix, p='fro')
    print("Relative reconstruction error:", relative_error)

    return U_reduced, S_reduced_diag, V_reduced


def adaptive_svd(matrix, error_threshold=0.3):
    # Assuming matrix is a PyTorch tensor on a GPU, convert it to a CuPy array
    # using DLPack to avoid data copying.
    matrix_cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(matrix.type(torch.float64)))

    # Perform SVD using CuPy, utilizing GPU acceleration.
    U_cp, S_cp, V_cp = cp.linalg.svd(matrix_cp, full_matrices=False)

    # Initialize to use all singular values initially
    total_singular_values = len(S_cp)
    optimal_dim = total_singular_values  # Start with the maximum number of singular values

    for dim_compress in range(total_singular_values, 0, -50):
        # Reduce the dimensions of U, S, and V matrices as per dim_compress
        U_reduced_cp = U_cp[:, :dim_compress]
        S_reduced_cp = S_cp[:dim_compress]
        V_reduced_cp = V_cp[:dim_compress, :]

        # Convert the reduced matrices back to PyTorch tensors, using DLPack
        # for efficient conversion between CuPy and PyTorch.
        U_reduced = torch.utils.dlpack.from_dlpack(U_reduced_cp.toDlpack())
        S_reduced = torch.utils.dlpack.from_dlpack(S_reduced_cp.toDlpack())
        V_reduced = torch.utils.dlpack.from_dlpack(V_reduced_cp.toDlpack())

        # Construct the diagonal matrix from S in PyTorch
        S_reduced_diag = torch.diag(S_reduced)

        # Calculate the reconstruction of the matrix
        matrix_reconstructed = torch.mm(torch.mm(U_reduced, S_reduced_diag), V_reduced)
        error = torch.norm(matrix - matrix_reconstructed, p='fro')

        # Check if the error is within the acceptable threshold
        if error <= error_threshold:
            optimal_dim = dim_compress
            print(f"Optimal dimension found: {optimal_dim} with error: {error}")
            break

    # Return the optimal dimension, and the corresponding U, S, and V matrices
    return optimal_dim, U_reduced.type(matrix.dtype), S_reduced_diag.type(matrix.dtype), V_reduced.type(matrix.dtype)

def adaptive_svd_combined(key_matrix, value_matrix, key_error_threshold=0.1, value_error_threshold=0.1):
    # Convert to CuPy arrays to perform SVD
    key_matrix_cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(key_matrix.type(torch.float64)))
    value_matrix_cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(value_matrix.type(torch.float64)))

    # Perform SVD using CuPy
    U_k_cp, S_k_cp, V_k_cp = cp.linalg.svd(key_matrix_cp, full_matrices=False)
    U_v_cp, S_v_cp, V_v_cp = cp.linalg.svd(value_matrix_cp, full_matrices=False)

    optimal_dim = None

    for dim_compress in range(500, len(S_k_cp), 2):
        # Reduce dimensions for both key and value matrices
        U_k_reduced_cp = U_k_cp[:, :dim_compress]
        S_k_reduced_cp = S_k_cp[:dim_compress]
        V_k_reduced_cp = V_k_cp[:dim_compress, :]

        U_v_reduced_cp = U_v_cp[:, :dim_compress]
        S_v_reduced_cp = S_v_cp[:dim_compress]
        V_v_reduced_cp = V_v_cp[:dim_compress, :]

        # Convert back to PyTorch tensors
        U_k_reduced = torch.utils.dlpack.from_dlpack(U_k_reduced_cp.toDlpack())
        S_k_reduced = torch.utils.dlpack.from_dlpack(S_k_reduced_cp.toDlpack())
        V_k_reduced = torch.utils.dlpack.from_dlpack(V_k_reduced_cp.toDlpack())

        U_v_reduced = torch.utils.dlpack.from_dlpack(U_v_reduced_cp.toDlpack())
        S_v_reduced = torch.utils.dlpack.from_dlpack(S_v_reduced_cp.toDlpack())
        V_v_reduced = torch.utils.dlpack.from_dlpack(V_v_reduced_cp.toDlpack())

        # Construct diagonal matrices from S
        S_k_reduced_diag = torch.diag(S_k_reduced)
        S_v_reduced_diag = torch.diag(S_v_reduced)

        # Calculate reconstructions
        key_reconstructed = torch.mm(torch.mm(U_k_reduced, S_k_reduced_diag), V_k_reduced)
        value_reconstructed = torch.mm(torch.mm(U_v_reduced, S_v_reduced_diag), V_v_reduced)

        # Calculate errors
        error_k = torch.norm(key_matrix - key_reconstructed, p='fro') / torch.norm(key_matrix, p='fro')
        error_v = torch.norm(value_matrix - value_reconstructed, p='fro') / torch.norm(value_matrix, p='fro')
        # print('Dim Compress', dim_compress, 'Key Error:', error_k, 'Value Error:', error_v)

        # Check if both errors are within the acceptable threshold
        if error_k <= key_error_threshold and error_v <= value_error_threshold:
            optimal_dim = dim_compress
            # print(f"Optimal dimension found: {optimal_dim} with key error: {error_k}, value error: {error_v}")
            break

    if optimal_dim is not None:
        print(f"Optimal dimension found: {optimal_dim} with key error: {error_k}, value error: {error_v}")
    else:
        print("No dimension found that satisfies the error thresholds.")

    return optimal_dim, U_k_reduced.type(key_matrix.dtype), S_k_reduced_diag.type(key_matrix.dtype), V_k_reduced.type(key_matrix.dtype), \
           U_v_reduced.type(value_matrix.dtype), S_v_reduced_diag.type(value_matrix.dtype), V_v_reduced.type(value_matrix.dtype)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, do_kv_compress: bool = False, dim_compress: int = 1024):
        super().__init__()
        self.max_batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.dim_compress = dim_compress
        self.do_kv_compress = do_kv_compress

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.initialize_caches()  # Initialize with full dimension if not compressing

    def initialize_caches(self):
        """Initialize caches with default dimension settings."""
        if not self.do_kv_compress:
            self.cache_k = torch.zeros(
                (
                    self.max_batch_size,
                    self.max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim,
                )
            ).cuda()
            self.cache_v = torch.zeros(
                (
                    self.max_batch_size,
                    self.max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim,
                )
            ).cuda()
        else:
            self.cache_k = torch.zeros(
                (
                    self.max_batch_size,
                    self.max_seq_len,
                    self.dim_compress
                )
            ).cuda()
            self.cache_v = torch.zeros(
                (
                    self.max_batch_size,
                    self.max_seq_len,
                    self.dim_compress
                )
            ).cuda()

    def update_cache_sizes(self):
        """Update cache sizes after adaptive SVD has determined new dimensions."""
        if self.do_kv_compress:
            self.cache_k = torch.zeros(
                (
                    self.max_batch_size,
                    self.max_seq_len,
                    self.dim_compress
                )
            ).cuda()
            self.cache_v = torch.zeros(
                (
                    self.max_batch_size,
                    self.max_seq_len,
                    self.dim_compress
                )
            ).cuda()

    def svd_weight(self, adaptive=False):
        # decompose key and value matrices
        if not adaptive:
            print('---------Decompose Start----------')
            # wk.weight.data -> (dim, n_heads * head_dim)
            print('wk weight data shape:', self.wk.weight.data.shape)
            self.wk_U, wk_S, wk_V = numpy_decompose_matrix(self.wk.weight.data, target_dim=self.dim_compress)
            # wk_U (n_heads * head_dim, dim_compress), wk_S (dim_compress, dim_compress)
            # wk_V (n_heads * head_dim, dim_compress), we need to transpose it manually
            self.wv_U, wv_S, wv_V = numpy_decompose_matrix(self.wv.weight.data, target_dim=self.dim_compress)
            # Precompute (UΣ) for updating the query matrix.
            # self.wk_A = torch.mm(self.wk_U, self.wk_S)  # Matrix A = (UΣ)_{n_heads * head_dim, dim_compress}
            # self.wv_B = torch.mm(self.wv_U, self.wv_S)  # Matrix B = (UΣ)_{n_heads * head_dim, dim_compress}
            self.wk_C = torch.mm(wk_S, wk_V)  # Matrix C = (ΣV^T)_{dim_compress, n_heads * head_dim}
            self.wv_D = torch.mm(wv_S, wv_V)  # Matrix D = (ΣV^T)_{dim_compress, n_heads * head_dim}
            print('---------Decompose End----------')
            # Incorporate wv_U into wo
            reshaped_wv_U = self.wv_U.view(self.n_local_kv_heads, self.head_dim, self.dim_compress)
            reshaped_wv_U = reshaped_wv_U.permute(0, 2, 1)  # (n_local_kv_heads, dim_compress, head_dim)
            # repeat wv_U to recover from n_kv_heads to n_heads
            reshaped_wv_U = reshaped_wv_U[:, None, :, :].expand(self.n_local_kv_heads, self.n_rep, self.dim_compress, self.head_dim).reshape(self.n_local_heads,self.dim_compress, self.head_dim)
            # reshaped_wv_U = reshaped_wv_U.transpose(1, 2)  # (n_local_heads, dim_compress, head_dim)
            wo_weight = self.wo.weight.data.transpose(1, 0).view(self.n_local_heads, self.head_dim, -1) # (n_local_heads, head_dim, dim)
            self.combined_weights = torch.matmul(reshaped_wv_U, wo_weight) # (n_local_heads, dim_compress, dim)
            print('----------------wo weight updated!--------------')
        else:
            print('---------Adaptive SVD Start----------')
            self.dim_compress, self.wk_U, self.wk_S, self.wk_V, self.wv_U, self.wv_S, self.wv_V= adaptive_svd_combined(self.wk.weight.data, self.wv.weight.data)
            # self.dim_compress_k, self.wk_U, self.wk_S, self.wk_V = adaptive_svd(self.wk.weight.data)
            # self.dim_compress_v, self.wv_U, self.wv_S, self.wv_V = adaptive_svd(self.wv.weight.data)
            self.wk_C = torch.mm(self.wk_S, self.wk_V)  # Matrix C = (ΣV^T)_{dim_compress, n_heads * head_dim}
            self.wv_D = torch.mm(self.wv_S, self.wv_V)  # Matrix D = (ΣV^T)_{dim_compress, n_heads * head_dim}
            print('---------Adaptive SVD End----------')
            self.update_cache_sizes()
            print('Initial Cache updated!')

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        cur_freqs_cis: torch.Tensor,
        entire_freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        if not self.do_kv_compress:
            bsz, seqlen, _ = x.shape
            xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

            xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=cur_freqs_cis)

            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk # (bs, cache_len + seqlen, n_local_kv_heads, head_dim)
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen] # (bs, cache_len + seqlen, n_local_kv_heads, head_dim)
            values = self.cache_v[:bsz, : start_pos + seqlen]

            # repeat k/v heads if n_kv_heads < n_heads
            keys = repeat_kv(
                keys, self.n_rep
            )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
            values = repeat_kv(
                values, self.n_rep
            )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

            xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
            keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
            values = values.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1) # (bs, seqlen, model_dim)
            return self.wo(output)
        else:
            # print('------------------KVC Forward------------------')
            bsz, seqlen, _ = x.shape
            xq = self.wq(x)
            xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xq = rope_q(xq, freqs_cis=cur_freqs_cis) # (bs, seqlen, n_local_heads, head_dim)

            xk = torch.matmul(x.view(bsz * seqlen, -1), self.wk_C.t())
            xk = xk.view(bsz, seqlen, -1)  # (bsz, seqlen, dim_compress)
            xv = torch.matmul(x.view(bsz * seqlen, -1), self.wv_D.t())  # Shape: (bsz * seqlen, dim_compress)
            xv = xv.view(bsz, seqlen, -1)  # Shape: (bsz, seqlen, dim_compress)

            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)
            self.cache_k[:bsz, start_pos: start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos: start_pos + seqlen] = xv
            keys = self.cache_k[:bsz, : start_pos + seqlen] # (bs, cache_len + seqlen, dim_compress)
            values = self.cache_v[:bsz, : start_pos + seqlen] # (bs, cache_len + seqlen, dim_compress)

            # for each key, multiply V_k, then apply RoPE
            # keys = keys.view(-1, self.dim_compress)
            keys = torch.matmul(keys, self.wk_U.t()) # (bs, cache_len + seqlen, total_head_dim)
            keys = keys.view(bsz, start_pos + seqlen, self.n_local_kv_heads, self.head_dim)
            keys = rope_q(keys, freqs_cis=entire_freqs_cis) # (bs, cache_len + seqlen, n_local_kv_heads, head_dim)

            # repeat k/v heads if n_kv_heads < n_heads
            keys = repeat_kv(keys, self.n_rep)
            # values = repeat_kv(values, self.n_rep)
            xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
            keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)

            # Apply the attention mechanism using the compressed representations
            scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim) # (bs, n_local_heads, seqlen, cache_len + seqlen)
            if mask is not None:
                scores += mask  # Apply mask if provided
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            values = values.unsqueeze(1) # (bs, 1, cache_len + seqlen, dim_compress)
            context = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, dim_compress)

            # reshaped_wv_U = self.wv_U.view(self.n_local_kv_heads, self.head_dim, self.dim_compress)
            # reshaped_wv_U = reshaped_wv_U.permute(0, 2, 1)  # (n_local_kv_heads, dim_compress, head_dim)
            # reshaped_wv_U = reshaped_wv_U[:, None, :, :].expand(self.n_local_kv_heads, self.n_rep, self.dim_compress, self.head_dim).reshape(self.n_local_heads, self.dim_compress, self.head_dim)
            # context = context.transpose(1, 2)  # (bs, seqlen, n_local_heads, dim_compress)
            # context_transformed = torch.einsum('bsjh,jhd->bsjd', context,
            #                                    reshaped_wv_U)  # (bs, seqlen, n_local_heads, head_dim)
            # output = context_transformed.contiguous().view(bsz, seqlen, -1)  # (bs, seqlen, model_dim)
            #
            # return self.wo(output) # (bs, seqlen, model_dim)
            context = context.transpose(1, 2) # (bs, seqlen, n_local_heads, dim_compress)
            output = torch.einsum('bsjh,jhd->bsd', context, self.combined_weights) # (bs, seqlen, dim)
            return output


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, do_kv_compress: bool = False, dim_compress: int = 1024):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, do_kv_compress, dim_compress)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        cur_freqs_cis: torch.Tensor,
        entire_freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, cur_freqs_cis, entire_freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def svd_attention_weight(self, adaptive=False):
        self.attention.svd_weight(adaptive=adaptive)

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs, custom_kvc_config: Optional[List[int]] = None):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = VocabParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.kv_compress_layers = params.kv_compress_layers
        if self.kv_compress_layers is None:
            self.kv_compress_layers = []

        self.layers = torch.nn.ModuleList()
        if custom_kvc_config:
            assert len(custom_kvc_config) == params.n_layers
            self.kvc_config = custom_kvc_config
            self.kv_compress_layers = [layer_id for layer_id, dim_compress in enumerate(custom_kvc_config) if dim_compress < params.dim//(params.n_heads//params.n_kv_heads)]
            print('KVC Config:', self.kvc_config)
            print('To be compressed layers:', self.kv_compress_layers)
        else:
            self.kvc_config = [params.dim//(params.n_heads//params.n_kv_heads) for _ in range(params.n_layers)]
            for layer_id in self.kv_compress_layers:
                self.kvc_config[layer_id] = params.dim_compress

        for layer_id in tqdm(range(params.n_layers), desc="Building Transformer Blocks"):
            do_kv_compress = layer_id in self.kv_compress_layers
            self.layers.append(TransformerBlock(layer_id, params, do_kv_compress=do_kv_compress, dim_compress=self.kvc_config[layer_id]))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        cur_freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]
        entire_freqs_cis = self.freqs_cis[: start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, cur_freqs_cis, entire_freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output

    def layerwise_svd(self, adaptive=False):
        # self.layers[0].svd_attention_weight()
        for i, layer in enumerate(tqdm(self.layers)):
            if i in self.kv_compress_layers:
                layer.svd_attention_weight(adaptive=adaptive)

    def get_layerwise_dim_compress(self):
        return [layer.attention.dim_compress for layer in self.layers]

    def get_model_stats(self):
        layerwise_dim_compress = self.get_layerwise_dim_compress()
        compression_ratio = [dim_compress / 1024 for dim_compress in layerwise_dim_compress]
        overall_compression_ratio = np.mean(compression_ratio)
        ret_dict = {
            "overall_compression_ratio": overall_compression_ratio,
            "layerwise_dim_compress": layerwise_dim_compress,
            "compression_ratio": compression_ratio
        }
        return ret_dict


