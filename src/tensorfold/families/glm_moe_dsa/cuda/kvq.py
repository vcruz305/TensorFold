"""GLM-5.3's quantized MLA latent cache (``--kv-dtype int4 | int8``): Flash Next's KV-cache format (``qwen4_exp``'s
``kvcache`` / ``kvquant``: H32-rotated groups of 32, one fp16 absmax scale a group, ExLlamaV3's midpoint grid, codes
kept rotated) on the 512-wide latent, with the quantizer arithmetic of ExLlamaV3's compiled ``quant_cache_cont``.

What a token's row holds, per layer (``LatentCache``):

- ``c``: the 512 latent values as codes - int4: 256 bytes, value j of a group in nibble j of the group's 16 bytes
  (low nibble = even j), byte for byte ExLlamaV3's ``-cq 4`` words (little-endian uint32, value j at bits 4j);
  int8: 512 int8 ``q - 128`` (ExLlamaV3's ``-cq 8`` bytes are these XOR 0x80);
- ``s``: 16 fp16 group scales (absmax of the rotated group, + 1e-10, ``__float2half_rn``) - ExLlamaV3's ``sk``;
- ``r``: the 64 RoPE dims, bf16 and unquantized (ExLlamaV3 keeps them fp16: "the shared RoPE key stays fp16
  unconditionally"); bf16 keeps the bf16 cache's exact RoPE bits and the same 128 bytes.

int4: 256 + 32 + 128 = 416 bytes a token and layer (bf16: 1152); int8: 512 + 32 + 128 = 672. The indexer's keys stay
bf16 (ExLlamaV3 keeps its ``k_idx`` plane fp16 in every cache mode), so the DSA selection reads the bf16 path's keys.

The quantizer is ExLlamaV3's, op for op (exllamav3_ext/cache/q_cache_kernels.cuh ``quant_block_x4``; its SASS for
sm_121, built with --use_fast_math: FADD butterfly, FMUL by 1/sqrt(32), FMNMX absmax, FADD 1e-10, MUFU.RCP,
FMUL, FFMA(., 2^(b-1), 2^(b-1)), F2I.FLOOR, clamp). Two places differ from ``kvquant.quant_groups_4/8``: the code is
``floor(fma(x * inv, m, m))`` (one rounding after the add), not ``floor(x * inv * m) + m``, and ``inv`` is the
approximate reciprocal (``rcp.approx.ftz.f32``, KV_RCP=approx) rather than IEEE division. Dequantization is
``kvquant``'s (bf16, still rotated); attention rotates the query in and the output back (H32 is its own inverse).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from tensorfold.families.qwen4_exp.cuda.kvcache import BITS_OF, GROUP, SCALE_DTYPE, check
from tensorfold.families.qwen4_exp.cuda.kvquant import dequant_group_4, dequant_group_8, h32

__all__ = ["BITS_OF", "GROUP", "LatentCache", "check", "row_bytes", "quant_exl3", "lat_tile", "h32",
           "KV_RCP", "rcp_flag"]

# 1/s in the quantizer: "approx" = rcp.approx.ftz.f32 (MUFU.RCP, what ExLlamaV3's fast-math build runs);
# "ieee" = correctly rounded division (the Triton interpreter cannot run inline PTX: CPU tests use it)
KV_RCP = os.environ.get("TF_GLM53_KV_RCP", "approx")
if KV_RCP not in ("approx", "ieee"):
    raise ValueError(f"TF_GLM53_KV_RCP={KV_RCP!r}: approx or ieee")
# the attention dot over dequantized keys: fp16 (ExLlamaV3's precision: 11-bit tiles and rotated queries) or bf16
KV_TILE = os.environ.get("TF_GLM53_KV_TILE", "fp16")
if KV_TILE not in ("fp16", "bf16"):
    raise ValueError(f"TF_GLM53_KV_TILE={KV_TILE!r}: fp16 or bf16")
TILE_F16 = KV_TILE == "fp16"


def rcp_flag() -> int:
    """The RCP constexpr the writer kernels take (0 under TRITON_INTERPRET: no inline assembly there)."""
    if os.environ.get("TRITON_INTERPRET", "0") == "1":
        return 0
    return 1 if KV_RCP == "approx" else 0


def row_bytes(kv_lora_rank: int, rope_dim: int, dtype: str) -> int:
    """Bytes one token's latent row takes in one layer (codes + scales + RoPE)."""
    check(dtype)
    if dtype == "bf16":
        return (kv_lora_rank + rope_dim) * 2
    codes = kv_lora_rank if dtype == "int8" else kv_lora_rank // 2
    return codes + (kv_lora_rank // GROUP) * 2 + rope_dim * 2


class LatentCache:
    """One layer's latent cache rows for a quantized ``--kv-dtype``: codes ``c``, fp16 scales ``s``, bf16 RoPE ``r``
    (row p of each plane is position p). Slicing rows (``lc[a:e]``) gives views, as for the bf16 tensor."""

    def __init__(self, rows: int, kv_lora_rank: int, rope_dim: int, device, dtype: str = "int4",
                 _planes: tuple | None = None) -> None:
        check(dtype)
        if dtype == "bf16":
            raise ValueError("a bf16 latent cache is a plain [rows, kv_lora_rank + rope] bf16 tensor")
        if kv_lora_rank % GROUP:
            raise ValueError(f"a quantized latent cache needs kv_lora_rank a multiple of {GROUP}, not {kv_lora_rank}")
        self.dtype, self.bits = dtype, BITS_OF[dtype]
        self.lw, self.rd = int(kv_lora_rank), int(rope_dim)
        if _planes is not None:
            self.c, self.s, self.r = _planes
            return
        rows = int(rows)
        if dtype == "int4":
            self.c = torch.zeros((rows, self.lw // 2), dtype=torch.uint8, device=device)
        else:
            self.c = torch.zeros((rows, self.lw), dtype=torch.int8, device=device)
        self.s = torch.zeros((rows, self.lw // GROUP), dtype=SCALE_DTYPE, device=device)
        self.r = torch.zeros((rows, self.rd), dtype=torch.bfloat16, device=device)

    def __getitem__(self, rows: slice) -> "LatentCache":
        if not isinstance(rows, slice):
            raise TypeError("LatentCache rows are sliced with a slice")
        return LatentCache(0, self.lw, self.rd, None, self.dtype, _planes=(self.c[rows], self.s[rows], self.r[rows]))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.c.shape[0], self.lw + self.rd)

    @property
    def nbytes(self) -> int:
        return self.c.nbytes + self.s.nbytes + self.r.nbytes

    @property
    def device(self):
        return self.c.device


def cache_nbytes(t) -> int:
    """Bytes of a bf16 cache tensor or a LatentCache."""
    if isinstance(t, LatentCache):
        return t.nbytes
    return t.numel() * t.element_size()


# -- the quantizer ------------------------------------------------------------------------------------------
@triton.jit
def _rcp(s, RCP: tl.constexpr):
    if RCP:
        return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=r,r", [s], dtype=tl.float32,
                                         is_pure=True, pack=1)
    else:
        return tl.math.div_rn(1.0, s)


@triton.jit
def quant_exl3(x, M: tl.constexpr, BITS: tl.constexpr, RCP: tl.constexpr):
    """(M, 32) fp32 groups -> (codes, fp16 scales) as ExLlamaV3's quant_block_x4 computes them: H32 (kvquant.h32:
    the same butterfly, bit 0 first, then 1/sqrt(32)), s = absmax + 1e-10, inv = 1/s, q = floor(fma(x inv, m, m))
    clamped to [0, 2m - 1]. int4: uint8 (M, 16), low nibble = even index; int8: int8 (M, 32) = q - 128."""
    x = h32(x, M)
    s = tl.max(tl.abs(x), axis=1) + 1e-10
    inv = _rcp(s, RCP)
    if BITS == 4:
        q = tl.floor(tl.fma(x * inv[:, None], 8.0, 8.0))
        q = tl.minimum(tl.maximum(q, 0.0), 15.0).to(tl.int32)
        lo, hi = tl.split(tl.reshape(q, (M, 16, 2)))
        code = (lo | (hi << 4)).to(tl.uint8)
    else:
        q = tl.floor(tl.fma(x * inv[:, None], 128.0, 128.0))
        code = (tl.minimum(tl.maximum(q, 0.0), 255.0) - 128.0).to(tl.int8)
    return code, s.to(tl.float16)


@triton.jit
def lat_tile(LC, LS, key, ok, LW: tl.constexpr, KT: tl.constexpr, BITS: tl.constexpr, F16: tl.constexpr = True):
    """Rows ``key`` (int64 [KT], masked by ``ok``) of a quantized latent cache -> [KT, LW], still rotated: fp16 as
    ExLlamaV3's plane loaders produce it ((q - (2^(b-1) - 0.5)) * (s / 2^(b-1)), one rounding to fp16), or with F16
    off kvquant's bf16 dequant. Masked rows read as zeros.
    One return statement: the Triton compiler (unlike the interpreter) visits every statement after a constexpr-if
    that returned, so an early return there is a "Return type mismatch" (fp16 vs bf16) at compile time."""
    gs = tl.arange(0, LW // 32)
    sc = tl.load(LS + key[:, None] * (LW // 32) + gs[None, :], mask=ok[:, None], other=0.0)
    if F16:
        s = tl.reshape(sc.to(tl.float32), (KT, LW // 32, 1))
        if BITS == 4:
            db = tl.arange(0, LW // 2)
            raw = tl.load(LC + key[:, None] * (LW // 2) + db[None, :], mask=ok[:, None], other=0).to(tl.int32)
            q = tl.reshape(tl.join((raw & 15).to(tl.float32), ((raw >> 4) & 15).to(tl.float32)), (KT, LW // 32, 32))
            v = (q - 7.5) * (s * 0.125)
        else:
            d = tl.arange(0, LW)
            code = tl.load(LC + key[:, None] * LW + d[None, :], mask=ok[:, None], other=0).to(tl.float32)
            v = (tl.reshape(code, (KT, LW // 32, 32)) + 0.5) * (s * 0.0078125)     # (q - 127.5) s / 128, q = code + 128
        out = tl.reshape(v, (KT, LW)).to(tl.float16)
    else:
        if BITS == 4:
            db = tl.arange(0, LW // 2)
            code = tl.load(LC + key[:, None] * (LW // 2) + db[None, :], mask=ok[:, None], other=0)
            out = dequant_group_4(code, sc, M=KT, W=LW)
        else:
            d = tl.arange(0, LW)
            code = tl.load(LC + key[:, None] * LW + d[None, :], mask=ok[:, None], other=0)
            out = dequant_group_8(code, sc, M=KT, W=LW)
    return out


@triton.jit
def rot_rows(x, NR: tl.constexpr, LW: tl.constexpr):
    """H32 over each 32-group of an (NR, LW) fp32 block (its own inverse)."""
    return tl.reshape(h32(tl.reshape(x, (NR * LW // 32, 32)), M=NR * LW // 32), (NR, LW))
