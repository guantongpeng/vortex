# Copyright © 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TTIR -> C transpiler for triton-vortex (plan P4-02).

Source-to-source codegen for the elementwise kernel class: the TTIR module
produced by Triton's generic frontend passes is serialized to text and
lowered to a single KMU C kernel (``#include <hip/hip_runtime.h>`` +
``__global__ void name(args_t* arg)``), which ``VortexBackend.make_vxbin``
then compiles with ``ci/hipcc_vortex.py --kernel-lib=vortex2`` into a real
.vxbin.  Anything outside the whitelisted subset raises ``NotImplementedError``
naming the offending op — never a silent fallback.

Whitelisted surface
-------------------
types   f32, f64, i1, i8/16/32/64 (+u variants); tensors thereof;
        !tt.ptr<E> and tensors of pointers
ops     arith.constant (incl. dense<> splats), tt.get_program_id,
        tt.make_range, tt.splat, tt.addptr (scalar/tensor),
        tt.load / tt.store (optional mask, optional ``other``),
        arith.{addi,subi,muli,divsi,divui,remsi,remui,andi,ori,xori,shli,
        shri,negf,addf,subf,mulf,divf,maxnumf,minnumf,cmpi,cmpf,select,
        extsi,extui,trunci,extf,truncf,sitofp,uitofp,fptosi,fptoui},
        math.{exp,log,sqrt,fabs}, tt.reduce (axis 0), tt.return
shape   exactly one tt.func, straight-line body (loops are unrolled by the
        generic TTIR passes in this subset; scf.* control flow is rejected)

Element mapping
---------------
One Triton program == one CTA.  Every SSA tensor value becomes a C array;
every elementwise op becomes a strided loop

    for (uint32_t _i = threadIdx.x; _i < N; _i += blockDim.x)

so the per-element dataflow each thread consumes is always thread-local
(all loops use the identical stride partition).  ``tt.reduce`` lowers to a
two-phase ``__local_mem()`` reduction: per-thread partial over the thread's
partition, ``__syncthreads()``, thread-0 combine, broadcast slot, second
``__syncthreads()``.  Arg-block fields are read from ``arg->`` at every
use (never cached into locals — measured VOLT -O3 workaround).

The launcher-side contract is recorded in the returned info dict:
``fmt`` (triton_vortex.hip.pack_args format string), ``args_size``
(must equal the C ``sizeof(args_t)``; the kernel image gets it in its
VXKMDATA record) and ``lmem`` bytes (launch ``lmem_size`` / ``shared``).
"""

from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys

# ---- IR text parsing -------------------------------------------------------

_VAL = r"%[A-Za-z0-9_.]+"
_FUNC_RE = re.compile(r"^\s*tt\.func\s+(?:public\s+)?@([\w.]+)\((.*)\)\s+"
                      r"attributes\s+\{.*\}\s*\{$")
_ARG_RE = re.compile(r"^(" + _VAL + r")\s*:\s*(.+?)(?:\s+loc\(.*\))?$")
_TENSOR_RE = re.compile(r"^tensor<(\d+)x(.+)>$")
_PTR_RE = re.compile(r"^!tt\.ptr<(.+)>$")
_DEF_RE = re.compile(r"^(" + _VAL + r")\s*=\s*(.*)$", re.S)

# ops that carry an optional attribute bag between name and operands
_ATTR_BAG = r"(?:\s+[\w.]+<[^>]*>)?"


class Ty:
    """Parsed TTIR type."""

    __slots__ = ("kind", "bits", "elem", "count")

    def __init__(self, kind, bits=None, elem=None, count=None):
        self.kind = kind      # bool|int|uint|float|ptr|tensor
        self.bits = bits
        self.elem = elem      # pointee / element Ty
        self.count = count

    @property
    def is_tensor(self):
        return self.kind == "tensor"

    def __repr__(self):
        return f"Ty({self.kind},{self.bits},{self.elem},{self.count})"


_SCALARS = {
    "i1": Ty("bool", 1), "i8": Ty("int", 8), "i16": Ty("int", 16),
    "i32": Ty("int", 32), "i64": Ty("int", 64),
    "ui8": Ty("uint", 8), "ui16": Ty("uint", 16), "ui32": Ty("uint", 32),
    "ui64": Ty("uint", 64),
    "f32": Ty("float", 32), "f64": Ty("float", 64),
}


def parse_ty(s):
    s = s.strip()
    if s in _SCALARS:
        return _SCALARS[s]
    m = _PTR_RE.match(s)
    if m:
        return Ty("ptr", elem=parse_ty(m.group(1)))
    m = _TENSOR_RE.match(s)
    if m:
        try:
            return Ty("tensor", count=int(m.group(1)),
                      elem=parse_ty(m.group(2)))
        except NotImplementedError:
            raise NotImplementedError(
                f"triton_vortex ttir_to_c: unsupported TTIR type '{s}' "
                "(whitelist: 1-D tensors of f32/f64, i1/i8/i16/i32/i64 "
                "(+u variants), and !tt.ptr of those)") from None
    raise NotImplementedError(
        f"triton_vortex ttir_to_c: unsupported TTIR type '{s}' "
        "(whitelist: f32/f64, i1/i8/i16/i32/i64 (+u variants), tensors and "
        "!tt.ptr of those)")


class FuncIR:
    def __init__(self, name, args):
        self.name = name
        self.args = args            # [(ssa_name, Ty)]
        self.ops = []               # raw op text (multi-line for tt.reduce)


def _parse_funcs(text):
    funcs = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _FUNC_RE.match(lines[i])
        if not m:
            i += 1
            continue
        name, argstr = m.group(1), m.group(2)
        args = []
        # Top-level comma split; whitelisted types contain no commas.
        for part in argstr.split(","):
            part = part.strip()
            if not part:
                continue
            am = _ARG_RE.match(part)
            if not am:
                raise NotImplementedError(
                    f"triton_vortex ttir_to_c: cannot parse tt.func "
                    f"argument '{part}' (kernel '{name}')")
            args.append((am.group(1), parse_ty(am.group(2))))
        i += 1
        fn = FuncIR(name, args)
        while i < len(lines) and not lines[i].strip().startswith("}"):
            stripped = lines[i].strip()
            if not stripped:
                i += 1
                continue
            if '"tt.reduce"' in stripped:
                block = [stripped]
                depth = stripped.count("(") - stripped.count(")")
                i += 1
                while True:
                    if i >= len(lines):
                        raise NotImplementedError(
                            "triton_vortex ttir_to_c: unterminated tt.reduce")
                    s = lines[i].strip()
                    block.append(s)
                    depth += s.count("(") - s.count(")")
                    i += 1
                    if s.startswith("})") and depth <= 0:
                        break
                fn.ops.append("\n".join(block))
            else:
                fn.ops.append(stripped)
                i += 1
        funcs.append(fn)
    if not funcs:
        raise NotImplementedError(
            "triton_vortex ttir_to_c: no tt.func found in module")
    if len(funcs) > 1:
        raise NotImplementedError(
            f"triton_vortex ttir_to_c: {len(funcs)} tt.func ops in one "
            "module; exactly one kernel per module is supported")
    return funcs[0]


def _arg_names_from_debug(text, n):
    """Best-effort recovery of the Triton parameter names from the debug
    (loc-annotated) serialization; falls back to positional a0.. names."""
    names = []
    for line in text.splitlines():
        m = _FUNC_RE.match(line)
        if m:
            argstr = m.group(2)
            # debug form: %x_ptr: !tt.ptr<f32> loc("x_ptr"(#loc))
            for part in re.finditer(_VAL + r"(?=:)", argstr):
                v = part.group(0)
                names.append(v[1:] if v[1:] != "" else None)
            break
    if len(names) != n or any(x is None for x in names):
        return [f"a{i}" for i in range(n)]
    return names


# ---- C type mapping --------------------------------------------------------

_C_MAP = {
    ("bool", 1): ("bool", "b", 1),
    ("int", 8): ("int8_t", "b", 1), ("int", 16): ("int16_t", "h", 2),
    ("int", 32): ("int32_t", "i", 4), ("int", 64): ("int64_t", "q", 8),
    ("uint", 8): ("uint8_t", "B", 1), ("uint", 16): ("uint16_t", "H", 2),
    ("uint", 32): ("uint32_t", "I", 4), ("uint", 64): ("uint64_t", "Q", 8),
    ("float", 32): ("float", "f", 4), ("float", 64): ("double", "d", 8),
}


def scalar_c(ty):
    """(ctype, pack fmt char, sizeof) for a non-tensor type."""
    if ty.kind == "ptr":
        raise ValueError("ptr handled by c_ptr()")
    key = (ty.kind, ty.bits)
    if key not in _C_MAP:
        raise NotImplementedError(
            f"triton_vortex ttir_to_c: no C mapping for type {ty}")
    return _C_MAP[key]


def _compute_layout(args, ptr_bits):
    """C struct layout of the arg block, computed with the platform ABI
    rules (natural member alignment, struct padded to max member align).
    On rv64 a struct of 3 uintptr_t + 1 int32_t is 24+4 -> sizeof 32.
    Returns ([(cname, ctype, fmt, size, align, offset)], sizeof)."""
    fields = []
    off = 0
    max_align = 1
    for cname, ty in args:
        if ty.kind == "ptr":
            size = align = ptr_bits // 8
            ctype, fmt = ("uintptr_t", "p")
        else:
            ctype, fmt, size = scalar_c(ty)
            align = size
        off = (off + align - 1) // align * align
        fields.append((cname, ctype, fmt, size, align, off))
        max_align = max(max_align, align)
        off += size
    size = (off + max_align - 1) // max_align * max_align
    return fields, size


# ---- constant literal rendering --------------------------------------------

def _float_literal(text, ty):
    """Bit-exact C literal for an MLIR float constant (decimal or 0x bits),
    narrowed through the target width."""
    text = text.strip()
    bits = ty.bits
    if text.startswith("0x") or text.startswith("-0x"):
        # MLIR prints the raw APInt bit pattern for values without a short
        # decimal form (e.g. 0xFF800000 for -inf).
        neg = text.startswith("-")
        hexpart = text.lstrip("-")
        u = int(hexpart, 16)
        # struct: big-endian pack of the bit pattern, then reinterpret
        raw = struct.pack(">Q" if bits == 64 else ">I", u)
        ftype = ">d" if bits == 64 else ">f"
        val = struct.unpack(ftype, raw)[0]
        if neg:
            val = -val
    else:
        low = text.lower()
        if low in ("inf", "-inf", "nan", "-nan"):
            sign = "-" if text.startswith("-") else ""
            word = "INFINITY" if "inf" in low else "NAN"
            return f"({sign}{word})"
        val = float(text)
        if bits == 32:
            val = struct.unpack("<f", struct.pack("<f", val))[0]
    if val != val:  # nan via hex bits
        return "(NAN)"
    if val in (float("inf"), float("-inf")):
        return ("(-INFINITY)" if val < 0 else "(INFINITY)")
    lit = float(val).hex()
    if bits == 32:
        lit += "f"
    return f"({lit})"


def _int_literal(text, ty):
    return f"({_c_int(ty)})({int(text, 0)})"


def _c_int(ty):
    if ty.kind == "bool":
        return "bool"
    return scalar_c(ty)[0]


def _c_unsigned(ty):
    return {1: "uint8_t", 2: "uint16_t", 4: "uint32_t", 8: "uint64_t"}[
        ty.bits // 8]


# ---- the emitter ------------------------------------------------------------

class _Emitter:
    def __init__(self, fn, ptr_bits):
        self.fn = fn
        self.ptr_bits = ptr_bits
        self.types = {}          # ssa -> Ty
        self.fields = {}         # ssa -> arg-block field name
        for (ssa, ty) in fn.args:
            self.types[ssa] = ty
        self.out = []            # emitted body lines
        self.lmem_off = 0

    # -- helpers -------------------------------------------------------------

    def xlat(self, ssa):
        """C expression for an SSA value: computed values are C variables,
        function arguments read the arg block directly at every use."""
        if ssa in self.fields:
            return f"arg->{self.fields[ssa]}"
        return self.cname(ssa)

    @staticmethod
    def cname(ssa):
        n = re.sub(r"\W", "_", ssa[1:])
        if not n or n[0].isdigit():
            n = "v" + n
        return "v" + n

    def ty(self, ssa):
        if ssa not in self.types:
            raise NotImplementedError(
                f"triton_vortex ttir_to_c: use of undefined value {ssa}")
        return self.types[ssa]

    def ctype(self, ssa_or_ty):
        ty = self.ty(ssa_or_ty) if isinstance(ssa_or_ty, str) else ssa_or_ty
        if ty.kind == "ptr":
            return "uintptr_t"
        return scalar_c(ty)[0]

    def loop(self, count, body):
        return [f"for (uint32_t _i = threadIdx.x; _i < {count}; "
                f"_i += blockDim.x) {{", f"    {body}", "}"]

    def emit(self, line):
        self.out.append("    " + line)

    def emit_raw(self, lines):
        self.out.extend(lines)

    def declare(self, ssa, ty):
        if ty.is_tensor:
            self.emit(f"{self.ctype(ty.elem)} {self.cname(ssa)}[{ty.count}];")
        else:
            self.emit(f"{self.ctype(ty)} {self.cname(ssa)};")

    def unsupported(self, what):
        raise NotImplementedError(
            f"triton_vortex ttir_to_c: unsupported TTIR op '{what}' "
            f"(kernel '{self.fn.name}'). The P4-02 whitelist covers the "
            "elementwise class (constant/get_program_id/make_range/splat/"
            "addptr/load/store/arith+math elementwise/select/casts, "
            "axis-0 tt.reduce); anything else must be lowered by hand.")

    # -- op implementations ---------------------------------------------------

    def op(self, text):
        dm = _DEF_RE.match(text)
        if dm:
            dst, rest = dm.group(1), dm.group(2)
        else:
            dst, rest = None, text
        if rest.startswith("tt.return"):
            self.emit("return;")
            return
        if rest.startswith("arith.constant"):
            self._constant(dst, rest)
            return
        if rest.startswith("tt.get_program_id"):
            self.types[dst] = Ty("int", 32)
            axis = rest.split()[1]
            self.declare(dst, self.types[dst])
            self.emit(f"{self.cname(dst)} = (int32_t)blockIdx.{axis};")
            return
        if rest.startswith("tt.make_range"):
            self._make_range(dst, rest)
            return
        if rest.startswith("tt.splat"):
            m = re.match(r"tt\.splat\s+(" + _VAL + r")\s*:\s*.+?\s*->\s*(.+)$",
                         rest)
            src, rty = m.group(1), parse_ty(m.group(2))
            self.types[dst] = rty
            self.declare(dst, rty)
            for line in self.loop(rty.count,
                                  f"{self.cname(dst)}[_i] = "
                                  f"({self.ctype(rty.elem)})({self.xlat(src)});"):
                self.emit(line)
            return
        if rest.startswith("tt.addptr"):
            self._addptr(dst, rest)
            return
        if rest.startswith("tt.load"):
            self._load(dst, rest)
            return
        if rest.startswith("tt.store"):
            self._store(rest)
            return
        if rest.startswith('"tt.reduce"'):
            self._reduce(dst, text)
            return
        if rest.startswith("arith.select"):
            self._select(dst, rest)
            return
        if rest.startswith("arith.cmpi") or rest.startswith("arith.cmpf"):
            self._cmp(dst, rest)
            return
        if rest.startswith("math.") or rest.startswith("arith.") or \
                rest.startswith("tt."):
            self._elementwise(dst, rest)
            return
        self.unsupported(rest.split()[0].split("(")[0])

    # constant ----------------------------------------------------------------

    def _constant(self, dst, rest):
        m = re.match(r"arith\.constant\s+(.+?)\s*:\s*(.+)$", rest)
        val, ty = m.group(1).strip(), parse_ty(m.group(2))
        self.types[dst] = ty
        dm = re.match(r"^dense<(.+)>$", val)
        if dm:
            inner = dm.group(1).strip()
            if not ty.is_tensor:
                self.unsupported("arith.constant dense<> with scalar type")
            lit = self._literal(inner, ty.elem)
            self.declare(dst, ty)
            for line in self.loop(ty.count,
                                  f"{self.cname(dst)}[_i] = {lit};"):
                self.emit(line)
            return
        if ty.is_tensor:
            self.unsupported(
                "arith.constant with a non-splat tensor value "
                f"('{val}')")
        lit = self._literal(val, ty)
        self.declare(dst, ty)
        self.emit(f"{self.cname(dst)} = {lit};")

    @staticmethod
    def _literal(val, ty):
        if ty.kind == "bool":
            return "true" if val in ("true", "1") else "false"
        if ty.kind == "float":
            return _float_literal(val, ty)
        return _int_literal(val, ty)

    # make_range ----------------------------------------------------------------

    def _make_range(self, dst, rest):
        m = re.match(r"tt\.make_range\s+\{end\s*=\s*(-?\d+)\s*:\s*i\d+,\s*"
                     r"start\s*=\s*(-?\d+)\s*:\s*i\d+\}\s*:\s*(.+)$", rest)
        end, start, ty = int(m.group(1)), int(m.group(2)), parse_ty(m.group(3))
        count = end - start
        if count != ty.count:
            raise NotImplementedError(
                f"triton_vortex ttir_to_c: make_range extent {count} != "
                f"tensor size {ty.count}")
        self.types[dst] = ty
        self.declare(dst, ty)
        for line in self.loop(
                count,
                f"{self.cname(dst)}[_i] = ({self.ctype(ty.elem)})"
                f"((int64_t){start} + (int64_t)_i);"):
            self.emit(line)

    # addptr ---------------------------------------------------------------------

    def _addptr(self, dst, rest):
        m = re.match(r"tt\.addptr\s+(" + _VAL + r"),\s*(" + _VAL + r")"
                     r"\s*:\s*(.+)$", rest)
        base, off = m.group(1), m.group(2)
        tail = m.group(3)
        parts = [p.strip() for p in tail.split(",")]
        if len(parts) != 2:
            self.unsupported("tt.addptr with more than one offset tensor")
        bty, oty = parse_ty(parts[0]), self.ty(off)
        if bty.kind != "ptr" and not (bty.is_tensor and bty.elem.kind == "ptr"):
            self.unsupported("tt.addptr on a non-pointer base")
        if bty.is_tensor:
            esize = bty.elem.elem.bits // 8
            rty = Ty("tensor", count=bty.count,
                     elem=Ty("ptr", elem=bty.elem.elem))
        else:
            esize = bty.elem.bits // 8
            rty = Ty("ptr", elem=bty.elem)
        self.types[dst] = rty
        self.declare(dst, rty)
        scale = f"(uint64_t){esize}"
        if rty.is_tensor:
            for line in self.loop(
                    rty.count,
                    f"{self.cname(dst)}[_i] = (uintptr_t)({self.idx(base)}) + "
                    f"(uint64_t)(int64_t)({self.idx(off)}) * {scale};"):
                self.emit(line)
        else:
            self.emit(f"{self.cname(dst)} = (uintptr_t)({self.xlat(base)}) + "
                      f"(uint64_t)(int64_t)({self.xlat(off)}) * {scale};")

    def idx(self, ssa):
        """Index/pointer array element or scalar expression."""
        if self.ty(ssa).is_tensor:
            return f"{self.xlat(ssa)}[_i]"
        return self.xlat(ssa)

    # load / store ---------------------------------------------------------------

    def _load(self, dst, rest):
        m = re.match(r"tt\.load\s+(.+?)\s*:\s*(.+)$", rest)
        ops = [s.strip() for s in m.group(1).split(",")]
        if len(ops) < 1 or len(ops) > 3:
            self.unsupported(f"tt.load with {len(ops)} operands")
        ptr = ops[0]
        mask = ops[1] if len(ops) > 1 else None
        other = ops[2] if len(ops) > 2 else None
        pty = self.ty(ptr)
        if pty.kind == "ptr":
            rty = pty.elem
        elif pty.is_tensor and pty.elem.kind == "ptr":
            rty = Ty("tensor", count=pty.count, elem=pty.elem.elem)
        else:
            self.unsupported("tt.load from a non-pointer value")
        self.types[dst] = rty
        self.declare(dst, rty)
        ect = self.ctype(rty if not rty.is_tensor else rty.elem)
        if rty.is_tensor:
            cond = self.idx(mask) if mask else None
            oexpr = self.idx(other) if other else f"({ect})0"
            deref = f"(*(({ect}*)(uintptr_t){self.idx(ptr)}))"
            body = (f"{self.cname(dst)}[_i] = {cond} ? {deref} : {oexpr};"
                    if cond else
                    f"{self.cname(dst)}[_i] = {deref};")
            for line in self.loop(rty.count, body):
                self.emit(line)
        else:
            cond = self.xlat(mask) if mask else None
            oexpr = self.xlat(other) if other else f"({ect})0"
            deref = f"(*(({ect}*)(uintptr_t){self.xlat(ptr)}))"
            self.emit(f"{self.cname(dst)} = {cond} ? {deref} : {oexpr};"
                      if cond else
                      f"{self.cname(dst)} = {deref};")

    def _store(self, rest):
        m = re.match(r"tt\.store\s+(.+?)\s*:\s*(.+)$", rest)
        ops = [s.strip() for s in m.group(1).split(",")]
        if len(ops) < 2 or len(ops) > 3:
            self.unsupported(f"tt.store with {len(ops)} operands")
        ptr, val = ops[0], ops[1]
        mask = ops[2] if len(ops) > 2 else None
        pty = self.ty(ptr)
        if pty.kind == "ptr":
            ect = self.ctype(pty.elem)
            cond = self.xlat(mask) if mask else None
            stmt = (f"*(({ect}*)(uintptr_t){self.xlat(ptr)}) = {self.xlat(val)};"
                    if not cond else
                    f"if ({cond}) *(({ect}*)(uintptr_t){self.xlat(ptr)}) = "
                    f"{self.xlat(val)};")
            self.emit(stmt)
            return
        if not (pty.is_tensor and pty.elem.kind == "ptr"):
            self.unsupported("tt.store to a non-pointer value")
        ect = self.ctype(pty.elem.elem)
        body = (f"*(({ect}*)(uintptr_t){self.idx(ptr)}) = {self.idx(val)};")
        if mask:
            body = f"if ({self.idx(mask)}) " + \
                   f"*(({ect}*)(uintptr_t){self.idx(ptr)}) = {self.idx(val)};"
        for line in self.loop(pty.count, body):
            self.emit(line)

    # reduce -----------------------------------------------------------------------

    _REDUCE_COMBINE = {
        # op -> (C combiner fmt, identity fmt)
        "addf": ("{a} + {b}", "({t})0"),
        "mulf": ("{a} * {b}", "({t})1"),
        "addi": ("{a} + {b}", "({t})0"),
        "muli": ("{a} * {b}", "({t})1"),
        "andi": ("{a} & {b}", "(~({t})0)"),
        "ori": ("{a} | {b}", "({t})0"),
        "maxnumf": ("fmax{f}({a}, {b})", "(-INFINITY)"),
        "minnumf": ("fmin{f}({a}, {b})", "(INFINITY)"),
        "maxsi": ("({a} > {b} ? {a} : {b})", "INT{w}_MIN"),
        "minsi": ("({a} < {b} ? {a} : {b})", "INT{w}_MAX"),
        "maxui": ("({a} > {b} ? {a} : {b})", "UINT{w}_MAX"),
        "minui": ("({a} < {b} ? {a} : {b})", "({t})0"),
    }

    _REDUCE_PARTIALS = 1024   # covers blockDim up to 1024 threads

    def _reduce(self, dst, text):
        m = re.match(r'(?:' + _VAL + r'\s*=\s*)?"tt\.reduce"\((' + _VAL +
                     r')\)\s+<\{axis\s*=\s*(\d+)'
                     r"[^}]*\}>\s*\(\{\n(.*)\n\s*\}\)\s*:\s*\(([^)]*)\)\s*->\s*(.+)$",
                     text, re.S)
        if not m:
            self.unsupported("tt.reduce (generic form)")
        src, axis, region, _, rty_s = m.groups()
        if int(axis) != 0:
            self.unsupported(f"tt.reduce with axis={axis} (only axis 0)")
        ops = re.findall(r"^\s*(%\w+)\s*=\s*(arith\.\w+)\s+%arg\d+,\s*%arg\d+"
                         r"\s*:\s*(\S+)\s*$", region, re.M)
        if len(ops) != 1:
            self.unsupported(
                "tt.reduce whose combine region is not a single binary op")
        combine = ops[0][1].split(".", 1)[1]
        sty = self.ty(src)
        rty = parse_ty(rty_s)
        if not sty.is_tensor:
            self.unsupported("tt.reduce over a scalar")
        if combine not in self._REDUCE_COMBINE:
            self.unsupported(f"tt.reduce combine op arith.{combine}")
        elem = sty.elem
        ect = self.ctype(elem)
        cfmt, ifmt = self._REDUCE_COMBINE[combine]

        def comb(a, b):
            return cfmt.format(a=a, b=b, f="" if elem.bits == 64 else "f",
                               t=ect, w=elem.bits)

        ident = ifmt.format(t=ect, w=elem.bits)

        # __local_mem() scratch: partials[] then broadcast slot
        esz = elem.bits // 8
        blk = (self._REDUCE_PARTIALS * esz + 63) // 64 * 64
        off = self.lmem_off
        if off + blk + esz > 16 * 1024:
            raise NotImplementedError(
                f"triton_vortex ttir_to_c: tt.reduce scratch exceeds the "
                f"16 KiB CTA local memory (need {off + blk + esz} bytes); "
                "reduce the number of reductions or the block size")
        self.lmem_off = off + blk + esz
        dstc = self.cname(dst)
        self.types[dst] = rty
        self.declare(dst, rty)
        self.emit("{")
        self.emit(f"    {ect}* _rp = ({ect}*)((char*)__local_mem() + {off});")
        self.emit(f"    {ect}* _rb = _rp + {self._REDUCE_PARTIALS};")
        self.emit(f"    {ect} _acc = {ident};")
        for line in self.loop(sty.count, f"_acc = {comb('_acc', self.idx(src))};"):
            self.emit("    " + line)
        self.emit("    _rp[threadIdx.x] = _acc;")
        self.emit("    __syncthreads();")
        self.emit("    if (threadIdx.x == 0) {")
        self.emit(f"        {ect} _fin = {ident};")
        self.emit("        for (uint32_t _t = 0; _t < blockDim.x; ++_t)")
        self.emit(f"            _fin = {comb('_fin', '_rp[_t]')};")
        self.emit("        _rb[0] = _fin;")
        self.emit("    }")
        self.emit("    __syncthreads();")
        self.emit(f"    {dstc} = _rb[0];")
        self.emit("}")

    # select / compare ----------------------------------------------------------------

    def _select(self, dst, rest):
        m = re.match(r"arith\.select\s+(" + _VAL + r"),\s*(" + _VAL + r"),\s*"
                     r"(" + _VAL + r")\s*:\s*(.+)$", rest)
        c, a, b = m.group(1), m.group(2), m.group(3)
        ty = self.ty(a)
        self.types[dst] = ty
        self.declare(dst, ty)
        if ty.is_tensor:
            body = (f"{self.cname(dst)}[_i] = {self.idx(c)} ? "
                    f"{self.idx(a)} : {self.idx(b)};")
            for line in self.loop(ty.count, body):
                self.emit(line)
        else:
            self.emit(f"{self.cname(dst)} = {self.xlat(c)} ? "
                      f"{self.xlat(a)} : {self.xlat(b)};")

    _CMPI = {"eq": "==", "ne": "!=", "slt": "<", "sle": "<=", "sgt": ">",
             "sge": ">=", "ult": "<", "ule": "<=", "ugt": ">", "uge": ">="}
    _CMPF = {"oeq": "==", "one": "!=", "olt": "<", "ole": "<=",
             "ogt": ">", "oge": ">="}

    def _cmp(self, dst, rest):
        m = re.match(r"arith\.cmp([fi])\s+(\w+),\s*(" + _VAL + r"),\s*(" +
                     _VAL + r")\s*:\s*(.+)$", rest)
        kind, pred, a, b, ty_s = m.groups()
        oty = self.ty(a)
        if oty.is_tensor:
            rty = Ty("tensor", count=oty.count, elem=Ty("bool", 1))
        else:
            rty = Ty("bool", 1)
        self.types[dst] = rty
        self.declare(dst, rty)
        table = self._CMPI if kind == "i" else self._CMPF
        if pred not in table:
            self.unsupported(f"arith.cmp{kind} predicate '{pred}' "
                             "(NaN-comparing predicates excluded on purpose)")
        op = table[pred]
        unsigned = kind == "i" and pred.startswith("u")
        if unsigned:
            ut = _c_unsigned(oty.elem if oty.is_tensor else oty)
            ae, be = f"({ut}){self.idx(a)}", f"({ut}){self.idx(b)}"
        else:
            ae, be = self.idx(a), self.idx(b)
        expr = f"{ae} {op} {be}"
        if rty.is_tensor:
            for line in self.loop(rty.count,
                                  f"{self.cname(dst)}[_i] = {expr};"):
                self.emit(line)
        else:
            self.emit(f"{self.cname(dst)} = {expr};")

    # elementwise arith/math/casts ------------------------------------------------------

    _CASTS = {"extsi", "extui", "trunci", "extf", "truncf", "sitofp",
              "uitofp", "fptosi", "fptoui"}

    def _elementwise(self, dst, rest):
        m = re.match(r"(arith|math)\.(\w+)" + _ATTR_BAG +
                     r"\s+(.*)$", rest)
        ns, op, tail = m.group(1), m.group(2), m.group(3)

        # casts: "arith.sitofp %0 : i32 to f32"
        cm = re.match("(" + _VAL + r")\s*:\s*(\S+)\s+to\s+(\S+)$", tail) \
            if op in self._CASTS else None
        if cm:
            src = cm.group(1)
            self.types[dst] = parse_ty(cm.group(3))
            ty = self.types[dst]
            self.declare(dst, ty)
            ct = self.ctype(ty if not ty.is_tensor else ty.elem)
            if ty.is_tensor:
                for line in self.loop(ty.count,
                                      f"{self.cname(dst)}[_i] = "
                                      f"({ct}){self.idx(src)};"):
                    self.emit(line)
            else:
                self.emit(f"{self.cname(dst)} = ({ct}){self.xlat(src)};")
            return

        # unary (math.*, arith.negf): "math.exp %1 : tensor<64xf32>"
        um = re.match(r"(" + _VAL + r")\s*:\s*(.+)$", tail)
        if um and (ns == "math" or op == "negf"):
            src = um.group(1)
            ty = self.ty(src)
            self.types[dst] = ty
            self.declare(dst, ty)
            fn = {"exp": "exp", "log": "log", "sqrt": "sqrt",
                  "fabs": "fabs"}.get(op)
            if ns == "math" and fn is None:
                self.unsupported(f"{ns}.{op}")
            if ns == "math":
                suffix = "" if (ty.is_tensor and ty.elem.bits == 64) or \
                    (not ty.is_tensor and ty.bits == 64) else "f"
                fn += suffix
            else:
                fn = None
            expr = (f"{fn}({self.idx(src)})" if fn else f"-{self.idx(src)}")
            if ty.is_tensor:
                for line in self.loop(ty.count,
                                      f"{self.cname(dst)}[_i] = {expr};"):
                    self.emit(line)
            else:
                self.emit(f"{self.cname(dst)} = {expr};")
            return

        # binary: "%a, %b : type"
        bm = re.match(r"(" + _VAL + r"),\s*(" + _VAL + r")\s*:\s*(.+)$", tail)
        if not bm:
            self.unsupported(f"{ns}.{op} (unrecognized operand layout '{tail}')")
        a, b, ty_s = bm.group(1), bm.group(2), bm.group(3)
        ty = parse_ty(ty_s)
        if ns == "math" or op in self._CASTS:
            self.unsupported(f"{ns}.{op}")
        self.types[dst] = ty
        self.declare(dst, ty)
        if op in ("divui", "remui"):
            ut = _c_unsigned(ty.elem if ty.is_tensor else ty)
            ae, be = f"({ut}){self.idx(a)}", f"({ut}){self.idx(b)}"
            expr = f"{ae} {op[3]} {be}" if op != "remui" else f"{ae} % {be}"
        elif op in ("maxnumf", "minnumf"):
            f64 = (ty.is_tensor and ty.elem.bits == 64) or \
                (not ty.is_tensor and ty.bits == 64)
            expr = (f"{op.replace('numf', '')}{'' if f64 else 'f'}"
                    f"({self.idx(a)}, {self.idx(b)})")
        else:
            sym = {"addi": "+", "subi": "-", "muli": "*", "divsi": "/",
                   "remsi": "%", "andi": "&", "ori": "|", "xori": "^",
                   "shli": "<<", "shri": ">>",
                   "addf": "+", "subf": "-", "mulf": "*", "divf": "/"}.get(op)
            if sym is None:
                self.unsupported(f"{ns}.{op}")
            expr = f"{self.idx(a)} {sym} {self.idx(b)}"
        if ty.is_tensor:
            for line in self.loop(ty.count,
                                  f"{self.cname(dst)}[_i] = {expr};"):
                self.emit(line)
        else:
            self.emit(f"{self.cname(dst)} = {expr};")


# ---- top level -----------------------------------------------------------------

def transpile(mod, ptr_bits=64):
    """Serialize + transpile one TTIR module. Returns an info dict:
    name, c_src, args ([(field, ctype, fmt, size, align, offset)]),
    fmt, args_size, lmem."""
    nodebug = mod.str_nodebug()
    debug = mod.str()
    if isinstance(nodebug, bytes):
        nodebug = nodebug.decode()
    if isinstance(debug, bytes):
        debug = debug.decode()
    fn = _parse_funcs(nodebug)
    names = _arg_names_from_debug(debug, len(fn.args))
    fields = {}
    for (ssa, _), name in zip(fn.args, names):
        fields[ssa] = name
    em = _Emitter(fn, ptr_bits)
    em.fields = fields
    for text in fn.ops:
        em.op(text)

    layout_fields, args_size = _compute_layout(
        [(fields[ssa], ty) for (ssa, ty) in fn.args], ptr_bits)

    doc = ["// Arg-block layout (single-pointer KMU ABI, "
           f"rv{ptr_bits}):"]
    for (cname, ctype, fmt, size, align, offset) in layout_fields:
        doc.append(f"//   +{offset:<3d} {ctype:<10s} {cname:<12s} "
                   f"(fmt '{fmt}', size {size}, align {align})")
    doc.append(f"//   sizeof(args_t) = {args_size} "
               "(natural alignment, struct padded to max member align)")
    doc.append(f"// CTA local memory requested: {em.lmem_off} bytes")

    body = "\n".join(em.out)
    decls = "\n".join(
        f"    {ctype} {cname};"
        for (cname, ctype, _f, _s, _a, _o) in layout_fields)
    src = f"""// Generated by triton_vortex.ttir_to_c from Triton TTIR.
// Kernel: {fn.name}  (one Triton program == one CTA; elementwise loops are
// threadIdx-strided; tt.reduce lowers to a two-phase __local_mem() reduce).
// DO NOT EDIT BY HAND — regenerate via triton.compile.
{chr(10).join(doc)}
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

struct args_t {{
{decls}
}};

__global__ void {fn.name}(args_t* arg) {{
{body}
}}
"""
    return {
        "name": fn.name,
        "c_src": src,
        "args": layout_fields,
        "fmt": "".join(f[2] for f in layout_fields),
        "args_size": args_size,
        "lmem": em.lmem_off,
    }


# ---- vxbin compilation (used by VortexBackend.make_vxbin) ------------------------

def _repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))  # <repo>/triton-vortex/..


def _cache_dir_for(hash_key):
    from triton.runtime.cache import FileCacheManager, _base32
    try:
        cm = FileCacheManager(_base32(hash_key))
        return cm.cache_dir
    except Exception:
        import tempfile
        d = tempfile.mkdtemp(prefix="triton_vortex_vxbin_")
        return d


def compile_vxbin(c_src, name, args_size, lmem, hash_key, ptr_bits=64,
                  tooldir=None, build_dir=None):
    """Write the transpiled C next to the cache metadata, compile it with
    ci/hipcc_vortex.py --kernel-lib=vortex2, and return the .vxbin path."""
    out_dir = _cache_dir_for(hash_key)
    hip_path = os.path.join(out_dir, f"{name}.hip")
    meta_path = os.path.join(out_dir, f"{name}.vxbin_meta.json")
    vxbin_path = os.path.join(out_dir, f"{name}.vxbin")
    with open(hip_path, "w", encoding="utf-8") as f:
        f.write(c_src)
    records = [{"name": name, "args_size": int(args_size),
                "static_lmem_bytes": int(lmem)}]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    if build_dir is None:
        from . import hip as _hip
        build_dir = _hip._BUILD
    if tooldir is None:
        tooldir = os.environ.get("TOOLDIR", "/data/vortex-tools")
    hipcc = os.path.join(_repo_root(), "ci", "hipcc_vortex.py")
    arch = f"vortex{ptr_bits}"
    cmd = [sys.executable, hipcc,
           "--tooldir", tooldir,
           "--build-dir", build_dir,
           f"--offload-arch={arch}",
           "--kernel-lib=vortex2",
           hip_path, "-o", vxbin_path]
    env = dict(os.environ, VX_KERNEL_METADATA=meta_path)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"triton_vortex ttir_to_c: hipcc-vortex failed for kernel "
            f"'{name}' (rc={proc.returncode}):\n"
            f"cmd: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n"
            f"{proc.stderr}")
    if not os.path.isfile(vxbin_path):
        raise RuntimeError(
            f"triton_vortex ttir_to_c: hipcc-vortex produced no vxbin at "
            f"{vxbin_path}")
    return vxbin_path
