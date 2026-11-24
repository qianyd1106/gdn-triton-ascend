import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch_npu
import fla_npu  # Register captured OPP before the first input initializes NPU.

# Reference ACLNN wrapper used for the measurements.
# Set GDR_VEOMNI_ROOT to a VeOmni checkout that contains
#   veomni/ops/kernels/gated_delta_rule/npu_varlen/flash_chunk_gated_delta_rule.py
# Set GDR_SKIP_SOURCE_HASH=1 if your checkout is not the exact measured snapshot.
_FRAMEWORK_ROOT = os.environ.get('GDR_VEOMNI_ROOT', '')
if not _FRAMEWORK_ROOT:
    raise RuntimeError('Set GDR_VEOMNI_ROOT to a VeOmni checkout before importing this wrapper')
_SOURCE = os.environ.get(
    'GDR_FLASH_SOURCE',
    str(Path(_FRAMEWORK_ROOT) / 'veomni/ops/kernels/gated_delta_rule/npu_varlen/flash_chunk_gated_delta_rule.py'),
)
_SOURCE_SHA = '481838f80757d4e020f074bcf3d1d3ac5002646b9024ad56cc49a98af27ddb4a'
_CASE_SHA = '8308f7c9f6bd49ca43257b82773df412d222140d61fa3fe129c5ebac561ff602'
if os.environ.get('GDR_SKIP_SOURCE_HASH') != '1':
    if hashlib.sha256(Path(_SOURCE).read_bytes()).hexdigest() != _SOURCE_SHA:
        raise RuntimeError('Captured GDR source changed')
if _FRAMEWORK_ROOT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_ROOT)
from veomni.ops.kernels.gated_delta_rule.npu_varlen import flash_chunk_gated_delta_rule as _actual
if os.environ.get('GDR_SKIP_SOURCE_HASH') != '1':
    if Path(_actual.__file__).resolve() != Path(_SOURCE).resolve():
        raise RuntimeError('GDR oracle imported from a different checkout')

class Model(torch.nn.Module):
    def forward(self, q, k, v, g, beta, cu_seqlens, cu_seqlens_list, grad_o):
        # All normalization, real forward state, backward, and all five gradients
        # are inside this measured call. Input creation does no reference work.
        with torch.enable_grad():
            q1, k1, v1, g1, b1 = [x.detach().requires_grad_(True) for x in (q, k, v, g, beta)]
            out, final_state = _actual.chunk_gated_delta_rule(
                q1, k1, v1, g1, b1, scale=None, initial_state=None,
                output_final_state=False, use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens, cu_seqlens_list=cu_seqlens_list,
                chunk_size=64, head_first=False)
            if final_state is not None:
                raise RuntimeError('Captured GDR final-state ABI changed')
            grads = torch.autograd.grad(out, (q1, k1, v1, g1, b1), grad_outputs=grad_o,
                                        create_graph=False, retain_graph=False, allow_unused=False)
            return tuple(x.detach() for x in (out, *grads))

def _tensor(spec, generator):
    shape, stride = tuple(spec['shape']), tuple(spec['stride'])
    offset = int(spec.get('storage_offset', 0))
    count = offset + 1 + sum((d-1)*s for d,s in zip(shape,stride))
    dtype = getattr(torch, spec['dtype'])
    device = torch.device('npu:0')
    if 'values' in spec:
        value = torch.tensor(spec['values'], dtype=dtype, device=device)
    else:
        base = torch.empty(count, dtype=dtype, device=device)
        value = base.as_strided(shape, stride, offset)
        distribution = spec.get('sample_distribution')
        if distribution == 'negative':
            value.copy_(-torch.rand(shape, dtype=torch.float32, device=device, generator=generator))
        elif distribution == 'unit_interval':
            value.copy_(torch.rand(shape, dtype=torch.float32, device=device, generator=generator))
        else:
            value.copy_(torch.randn(shape, dtype=torch.float32, device=device, generator=generator))
    fmt = spec.get('npu_format')
    if fmt is not None and torch_npu.get_npu_format(value) != fmt:
        value = torch_npu.npu_format_cast(value, fmt)
    if tuple(value.shape) != shape or tuple(value.stride()) != stride or value.storage_offset() != offset:
        raise RuntimeError('Input provider failed to retain captured layout for '+spec['name'])
    if fmt is not None and torch_npu.get_npu_format(value) != fmt:
        raise RuntimeError('Input provider failed to retain captured NPU format')
    return value

class _LazyCases:
    # The native verifier needs len()+iteration, the native benchmark may index.
    # Materialize one original case at a time so 576 large cases never coexist.
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        if isinstance(index, slice): return [self[i] for i in range(*index.indices(len(self)))]
        row = self.rows[index]
        generator = torch.Generator(device='npu:0').manual_seed(int(row['precision_seed']))
        return [_tensor(s, generator) if s['type']=='tensor' else s.get('value') for s in row['inputs']]
    def __iter__(self):
        for i in range(len(self)): yield self[i]

def get_input_groups():
    path = Path(__file__).with_name('GatedDeltaRuleTraining.json')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _CASE_SHA:
        raise RuntimeError('Frozen captured GDR cases changed')
    return _LazyCases([json.loads(line) for line in raw.decode().splitlines() if line.strip()])

def get_init_inputs(): return []
