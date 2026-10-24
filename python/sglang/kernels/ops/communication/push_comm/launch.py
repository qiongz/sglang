# SPDX-License-Identifier: Apache-2.0
"""Low-overhead relaunch of compiled Gluon kernels.

FastLauncher caches the CompiledKernel per (caller key, argument
specialization, as Triton computes it) and launches it directly, skipping the
JIT's per-call argument binding. Instances compile every variant with warm();
after seal() a call that still makes Triton compile is logged as an error."""

import logging

import torch

try:  # the runtime launch hooks (profilers): with one set, the full launch path
    from triton import knobs as _knobs
except Exception:  # noqa: BLE001
    _knobs = None

logger = logging.getLogger(__name__)
_SEALED = [False]


def seal():
    """Mark the end of the warm-up: later misses are reported."""
    _SEALED[0] = True


def _spec(a, nospec):
    if isinstance(a, torch.Tensor):
        return (
            a.dtype,
            a.data_ptr() % 16 == 0,
            a.untyped_storage().size() <= 2147483647,
        )
    if isinstance(a, bool):
        return ("b", a)
    if isinstance(a, int):
        if nospec:
            return ("i", -(2**31) <= a < 2**31)
        return ("i", a == 1, a % 16 == 0, -(2**31) <= a < 2**31)
    if isinstance(a, float):
        return ("f",)
    return (type(a).__name__,)  # pointer-like helper objects: one specialization


def _no_hooks():
    """No runtime launch hook is installed (Triton 3.7: hook chains with no
    calls; older versions: None)."""
    if _knobs is None:
        return False
    rt = _knobs.runtime
    for h in (rt.launch_enter_hook, rt.launch_exit_hook):
        if h is not None and getattr(h, "calls", True):
            return False
    return True


# the spec of a bf16 operand the caller vouches for: 16-byte aligned, storage
# < 2 GiB (an input through the communicator's alignment check, a fresh output)
TRUSTED_BF16 = (torch.bfloat16, True, True)


class FastLauncher:
    def __init__(self, fn):
        self.fn = fn
        self.name = getattr(getattr(fn, "fn", None), "__name__", "kernel")
        self.names = list(fn.arg_names)
        dns = fn.do_not_specialize or ()
        self.nospec = {self.names[d] if isinstance(d, int) else d for d in dns}
        self.cache = {}
        self.hits = self.misses = self.late_compiles = 0
        self._reported = set()

    def _key(self, args, key):
        # a tensor passed several times is specialized once
        seen, out = {}, []
        for i, a in enumerate(args):
            if isinstance(a, torch.Tensor):
                sp = seen.get(id(a))
                if sp is None:
                    sp = seen[id(a)] = _spec(a, False)
                out.append(sp)
            else:
                out.append(_spec(a, self.names[i] in self.nospec))
        return (key, tuple(out))

    def _key_trusted(self, args, key, tspec):
        """_key with every tensor argument's spec given by the caller (tspec),
        without querying the tensors."""
        names, nospec = self.names, self.nospec
        return (
            key,
            tuple(
                tspec if isinstance(a, torch.Tensor) else _spec(a, names[i] in nospec)
                for i, a in enumerate(args)
            ),
        )

    def bound(self, grid, args, key, tspec):
        """A closure launching the cached variant a call with these arguments
        (tensors vouched for by tspec) would hit, with the key and lookup done
        once; None on a miss. The closure takes the full argument list."""
        ent = self.cache.get(self._key_trusted(args, key, tspec))
        if ent is None:
            return None
        ck, tail = ent
        if ck.module is None:
            ck._init_handles()
        g = grid if isinstance(grid, tuple) else (grid,)
        g0, g1, g2 = (g + (1, 1))[:3]
        fn, meta, run = ck.function, ck.packed_metadata, ck.run
        raw, dev = torch._C._cuda_getCurrentRawStream, torch._C._cuda_getDevice

        def launch(*a):
            if not _no_hooks():  # a launch hook installed since: the full path
                ck[(g0, g1, g2)](*a, *tail)
                return
            self.hits += 1
            run(g0, g1, g2, raw(dev()), fn, meta, None, None, None, *a, *tail)

        return launch

    def _tail(self, n, kw):
        # every parameter after the positional ones, in signature order
        tail = []
        for name in self.names[n:]:
            if name in kw:
                tail.append(kw[name])
            else:
                tail.append(self.fn.params[self.names.index(name)].default)
        return tuple(tail)

    def _jit_entries(self):
        try:
            return len(self.fn.device_caches[torch.cuda.current_device()][0])
        except Exception:  # no count on this Triton version: no compile detection
            return None

    def warm(self, grid, args, kw, key, fast=True):
        """Compile (nothing is launched) the variant a call with these arguments
        would use and prime the cache with it; returns the CompiledKernel."""
        kw = kw() if callable(kw) else kw
        ck = self.fn.warmup(*args, grid=grid, **kw)
        if fast and ck is not None:
            self.cache[self._key(args, key)] = (ck, self._tail(len(args), kw))
        return ck

    def __call__(self, grid, args, kw, key, fast=True):
        """kw: dict of the remaining (constexpr) arguments, or a zero-argument
        callable returning it (evaluated on a miss only); key: everything the
        caller varies that changes kw (instance identity, variant, ...)."""
        if not fast:
            return self.fn[grid](*args, **(kw() if callable(kw) else kw))
        sk = self._key(args, key)
        ent = self.cache.get(sk)
        if ent is not None:
            self.hits += 1
            ck, tail = ent
            g = grid if isinstance(grid, tuple) else (grid,)
            g = (g + (1, 1))[:3]
            if _no_hooks() and ck.module is not None:
                # what CompiledKernel[grid](*args) ends with, without its
                # per-call closure and launch metadata (used by hooks only)
                ck.run(
                    g[0],
                    g[1],
                    g[2],
                    torch._C._cuda_getCurrentRawStream(torch._C._cuda_getDevice()),
                    ck.function,
                    ck.packed_metadata,
                    None,
                    None,
                    None,
                    *args,
                    *tail,
                )
            else:
                ck[g](*args, *tail)
            return ck
        self.misses += 1
        kw = kw() if callable(kw) else kw
        late = _SEALED[0]
        n0 = self._jit_entries() if late else None
        ck = self.fn[grid](*args, **kw)
        if late and sk not in self._reported:
            self._reported.add(sk)
            n1 = self._jit_entries()
            if n0 is not None and n1 is not None and n1 > n0:
                self.late_compiles += 1
                logger.error(
                    "push-comm: %s compiled after the warm-up (key %s, spec %s); "
                    "a JIT inside a collective stalls every peer",
                    self.name,
                    sk[0],
                    sk[1],
                )
        self.cache[sk] = (ck, self._tail(len(args), kw))
        return ck


class PushInst:
    """The call surface shared by the kernel instances (hidden, max_tokens and
    _launch / _launch_tail come from the subclass)."""

    can_tail = True  # a masked-tail kernel in every form, failure handling too

    def supports(self, x):
        return (
            x.dim() == 2
            and x.shape[1] == self.hidden
            and x.dtype == torch.bfloat16
            and 0 < x.shape[0] <= self.max_tokens
            and x.is_contiguous()
        )

    def all_reduce(self, x):
        out = torch.empty_like(x)
        self._launch(x, out)
        return out

    def all_reduce_tail(self, x):
        """All-reduce of a flat bf16 x whose size H does not divide."""
        n = x.numel()
        assert (
            x.dtype == torch.bfloat16
            and x.is_contiguous()
            and 0 < -(-n // self.hidden) <= self.max_tokens
        )
        out = torch.empty(n, dtype=x.dtype, device=x.device)
        self._launch_tail(x.view(-1), out, n)
        return out
