"""tools/test_ds4_experts_from.py - pack_gguf's --experts-from: a quant's pack with ONLY its routed experts
(blk.N.ffn_{gate,up,down}_exps.weight) taken from another quant's shards of the same model - every entry pointing at
exactly those bytes, every other tensor the quant's own, experts-from.json saying what was taken - and each way the
other set can fail to be that (a clear error, no pack).  Small made-up GGUFs, stdlib only: no GPU, no internet.

    python -m unittest tools.test_ds4_experts_from -v
"""
from __future__ import annotations

import json
import os
import pathlib
import random
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
PACK = HERE / "ds4" / "pack_gguf.py"
BF16, F32, IQ2_XXS, IQ3_XXS, IQ1_S, IQ2_XS, IQ1_M = 30, 0, 16, 18, 19, 17, 29
BLOCK = {BF16: (1, 2), F32: (1, 4), IQ2_XXS: (256, 66), IQ3_XXS: (256, 98), IQ1_S: (256, 50), IQ2_XS: (256, 74),
         IQ1_M: (256, 56)}
GEO = {"block_count": 2, "embedding_length": 256, "expert_count": 2, "expert_used_count": 1,
       "expert_feed_forward_length": 256}
ROUTED_DIMS = [256, 2, 2]   # (ggml order: 256 values a row, 2 rows, 2 experts)


def s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def nbytes(typ: int, dims: list) -> int:
    per, bpb = BLOCK[typ]
    n = 1
    for d in dims:
        n *= d
    return n // per * bpb


def gguf(path: pathlib.Path, tensors: list, meta: bool = True, split=None, geo=None, arch="deepseek41", cut=0) -> dict:
    """A GGUF of tensors [(name, type, dims)] with distinct made-up bytes; meta: the model's metadata (architecture,
    geometry, a vocabulary), split: (no, count) -> {name: its bytes}."""
    kv = []
    if meta:
        kv.append(s("general.architecture") + struct.pack("<I", 8) + s(arch))
        for k, v in (geo or GEO).items():
            kv.append(s("deepseek41.%s" % k) + struct.pack("<II", 4, v))
        kv.append(s("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 4) + b"".join(s(t) for t in "abcd"))
    if split is not None:
        kv.append(s("split.no") + struct.pack("<IH", 2, split[0]))
        kv.append(s("split.count") + struct.pack("<IH", 2, split[1]))
    info, data, off, out = b"", b"", 0, {}
    rng = random.Random(str(path))
    for name, typ, dims in tensors:
        info += s(name) + struct.pack("<I", len(dims)) + struct.pack("<%dQ" % len(dims), *dims) + struct.pack("<IQ", typ, off)
        b = bytes(rng.randrange(256) for _ in range(nbytes(typ, dims)))
        b += b"\0" * ((-len(b)) % 32)
        out[name] = b[:nbytes(typ, dims)]
        data += b
        off += len(b)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)) + b"".join(kv) + info
    head += b"\0" * ((-len(head)) % 32)
    body = head + data
    path.write_bytes(body[:len(body) - cut])
    return out


def routed(layer: int, gu: int, down: int) -> list:
    return [("blk.%d.ffn_%s_exps.weight" % (layer, r), t, ROUTED_DIMS)
            for r, t in (("gate", gu), ("up", gu), ("down", down))]


def entries(pack: pathlib.Path):
    files, ts = {}, {}
    for line in (pack / "tensors.txt").read_text().splitlines():
        p = line.split()
        if p[0] == "file":
            files[int(p[1])] = p[2]
        else:
            ts[p[1]] = (p[2], int(p[3]), int(p[4]), int(p[5]))
    return files, ts


class ExpertsFrom(unittest.TestCase):
    def setUp(self):
        self.base = pathlib.Path(tempfile.mkdtemp())
        self.q = self.base / "Maya-S"
        self.u = self.base / "Unsloth"
        self.q.mkdir()
        self.u.mkdir()
        # the quant: its own embedding, router, routed experts (IQ2_XXS / IQ3_XXS), one shard
        self.mine = gguf(self.q / "m-00001-of-00001.gguf",
                         [("token_embd.weight", BF16, [256, 4]), ("blk.0.ffn_gate_inp.weight", F32, [256, 2])]
                         + routed(0, IQ2_XXS, IQ3_XXS) + routed(1, IQ2_XXS, IQ3_XXS))
        # the other set, 4 shards: 1 the metadata, 2 layer 0's experts and a router passed over, 3 layer 1's, 4 not
        # there (as unsloth's Engram table shard)
        self.theirs = {}
        gguf(self.u / "u-00001-of-00004.gguf", [], split=(0, 4))
        self.theirs.update(gguf(self.u / "u-00002-of-00004.gguf",
                                [("blk.0.ffn_gate_inp.weight", F32, [256, 2])] + routed(0, IQ1_S, IQ2_XXS),
                                meta=False, split=(1, 4)))
        self.theirs.update(gguf(self.u / "u-00003-of-00004.gguf", routed(1, IQ1_S, IQ2_XS), meta=False, split=(2, 4)))
        self.shards = [str(self.u / ("u-0000%d-of-00004.gguf" % i)) for i in (1, 2, 3)]

    def pack(self, *extra, out="maya-ds4-hybrid"):
        cmd = [sys.executable, str(PACK), "--model", str(self.q)]
        if out:
            cmd += ["--out", str(self.q / out)]   # (an absolute out stays as it is)
        return subprocess.run(cmd + list(extra), capture_output=True, text=True)

    def test_only_the_routed_experts_taken_bit_for_bit(self):
        before = {p.name: p.read_bytes() for p in list(self.q.glob("*.gguf")) + list(self.u.glob("*.gguf"))}
        r = self.pack("--experts-from", *self.shards)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("routed experts from 2 shards of 4", r.stdout)
        own = self.pack(out="maya-ds4-own")
        self.assertEqual(own.returncode, 0, own.stderr)
        files, ts = entries(self.q / "maya-ds4-hybrid")
        ofiles, ots = entries(self.q / "maya-ds4-own")
        self.assertEqual(files, {0: "m-00001-of-00001.gguf", 1: "../Unsloth/u-00002-of-00004.gguf",
                                 2: "../Unsloth/u-00003-of-00004.gguf"})
        self.assertEqual(sorted(ts), sorted(ots))                       # (the same entries, by name)
        types = {(0, "w1"): "IQ1_S", (0, "w3"): "IQ1_S", (0, "w2"): "IQ2_XXS",
                 (1, "w1"): "IQ1_S", (1, "w3"): "IQ1_S", (1, "w2"): "IQ2_XS"}
        role = {"w1": "gate", "w3": "up", "w2": "down"}
        for name, (dt, fi, off, nb) in ts.items():
            if ".ffn.experts." not in name:
                self.assertEqual((dt, fi, off, nb), ots[name], name)    # (every other tensor the quant's own)
                self.assertEqual(fi, 0, name)
                continue
            il, e, w = int(name.split(".")[1]), int(name.split(".")[4]), name.split(".")[5]
            self.assertEqual(dt, types[(il, w)], name)
            self.assertEqual(fi, 1 + il, name)
            src = self.theirs["blk.%d.ffn_%s_exps.weight" % (il, role[w])]
            with open(os.path.join(str(self.q), files[fi]), "rb") as f:
                f.seek(off)
                self.assertEqual(f.read(nb), src[e * nb:(e + 1) * nb], name)
        desc = json.loads((self.q / "maya-ds4-hybrid" / "experts-from.json").read_text())
        self.assertEqual((desc["split_count"], desc["routed"], desc["files"]),
                         (4, 6, ["../Unsloth/u-00002-of-00004.gguf", "../Unsloth/u-00003-of-00004.gguf"]))
        self.assertEqual([x["split_no"] for x in desc["shards"]], [0, 1, 2])
        self.assertEqual(desc["passed_over"], {"tensors": 1, "bytes": 256 * 2 * 4})
        self.assertEqual(desc["layers"], {"0": ["IQ1_S", "IQ1_S", "IQ2_XXS"], "1": ["IQ1_S", "IQ1_S", "IQ2_XS"]})
        self.assertEqual(desc["replaced"]["blk.1.ffn_down_exps.weight"], "IQ3_XXS")
        self.assertFalse((self.q / "maya-ds4-own" / "experts-from.json").exists())
        after = {p.name: p.read_bytes() for p in list(self.q.glob("*.gguf")) + list(self.u.glob("*.gguf"))}
        self.assertEqual(after, before)                                 # (no file touched)

    def refused(self, want, *paths, out="maya-ds4-hybrid"):
        r = self.pack("--experts-from", *paths, out=out)
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn(want, r.stderr)
        self.assertFalse((self.q / "maya-ds4-hybrid" / "tensors.txt").exists())

    def test_out_required(self):
        self.refused("name the pack's folder with --out", *self.shards, out=None)
        self.assertFalse((self.q / "maya-ds4").exists())

    def test_only_a_fresh_destination(self):
        """The quant's own pack (written first), by its path, a link to it or a path through one; any existing folder,
        empty or not; the model's folder: each refused before anything is written, every existing file unchanged."""
        own = self.pack(out="maya-ds4")
        self.assertEqual(own.returncode, 0, own.stderr)
        other = self.q / "maya-ds4-old"
        other.mkdir()
        (other / "note.txt").write_text("kept")
        (self.q / "maya-ds4-empty").mkdir()
        (self.base / "alias").symlink_to(self.q / "maya-ds4")
        snap = lambda: {str(p): p.read_bytes() for p in sorted(self.base.rglob("*")) if p.is_file()}   # noqa: E731
        before = snap()
        for out, want in ((str(self.q / "maya-ds4"), "is there - the hybrid pack goes to a fresh folder"),
                          ("maya-ds4", "is there"),
                          (str(self.base / "alias"), "is there"),
                          (str(self.base / "alias" / "new"), "is the model's folder or the quant's own pack"),
                          (str(other), "is there"),
                          (str(self.q / "maya-ds4-empty"), "is there"),
                          (str(self.q), "is there")):
            with self.subTest(out):
                r = self.pack("--experts-from", *self.shards, out=out)
                self.assertNotEqual(r.returncode, 0, r.stdout)
                self.assertIn(want, r.stderr)
                self.assertEqual(snap(), before)
                self.assertFalse((self.q / "maya-ds4" / "new").exists())
        self.assertFalse(any(self.base.rglob("experts-from.json")))

    def test_the_first_shard_required(self):
        self.refused("its first shard (split.no 0, the model's metadata) is not among the files", *self.shards[1:])

    def test_a_layer_missing(self):
        self.refused("routed tensors missing ['blk.1.ffn_down_exps.weight', 'blk.1.ffn_gate_exps.weight', "
                     "'blk.1.ffn_up_exps.weight'], extra []", *self.shards[:2])

    def test_an_extra_layer(self):
        gguf(self.u / "u-00004-of-00004.gguf", routed(2, IQ1_S, IQ2_XXS), meta=False, split=(3, 4))
        self.refused("extra ['blk.2.ffn_down_exps.weight'", *self.shards, str(self.u / "u-00004-of-00004.gguf"))

    def test_other_dims(self):
        gguf(self.u / "u-00003-of-00004.gguf", routed(1, IQ1_S, IQ2_XS)[:2] +
             [("blk.1.ffn_down_exps.weight", IQ2_XS, [256, 3, 2])], meta=False, split=(2, 4))
        self.refused("blk.1.ffn_down_exps.weight dims [256, 3, 2], the quant's [256, 2, 2]", *self.shards)

    def test_gate_and_up_differ(self):
        gguf(self.u / "u-00003-of-00004.gguf", [("blk.1.ffn_gate_exps.weight", IQ1_S, ROUTED_DIMS),
                                               ("blk.1.ffn_up_exps.weight", IQ2_XXS, ROUTED_DIMS),
                                               ("blk.1.ffn_down_exps.weight", IQ2_XS, ROUTED_DIMS)],
             meta=False, split=(2, 4))
        self.refused("layer 1's gate IQ1_S and up IQ2_XXS differ", *self.shards)

    def test_a_type_the_engine_does_not_read(self):
        gguf(self.u / "u-00003-of-00004.gguf", routed(1, IQ1_M, IQ2_XS), meta=False, split=(2, 4))
        self.refused("blk.1.ffn_gate_exps.weight has ggml type 29, which the engine does not read", *self.shards)

    def test_another_geometry_or_model(self):
        gguf(self.u / "u-00001-of-00004.gguf", [], split=(0, 4), geo=dict(GEO, expert_count=3))
        self.refused("deepseek41.expert_count 3, the quant's 2", *self.shards)
        gguf(self.u / "u-00001-of-00004.gguf", [], split=(0, 4), arch="qwen4exp")
        self.refused("architecture 'qwen4exp', the quant's 'deepseek41'", *self.shards)

    def test_not_one_split(self):
        gguf(self.u / "v-00003-of-00005.gguf", routed(1, IQ1_S, IQ2_XS), meta=False, split=(2, 5))
        self.refused("not one split's shards (split.count [4, 5]", *self.shards[:2], str(self.u / "v-00003-of-00005.gguf"))
        self.refused("not one split's shards", *self.shards, self.shards[2])

    def test_a_tensor_twice(self):
        gguf(self.u / "u-00004-of-00004.gguf", routed(1, IQ1_S, IQ2_XS)[:1], meta=False, split=(3, 4))
        self.refused("blk.1.ffn_gate_exps.weight in u-00003-of-00004.gguf and u-00004-of-00004.gguf",
                     *self.shards, str(self.u / "u-00004-of-00004.gguf"))

    def test_a_shard_not_whole(self):
        gguf(self.u / "u-00003-of-00004.gguf", routed(1, IQ1_S, IQ2_XS), meta=False, split=(2, 4), cut=40)
        self.refused("--experts-from: not whole (u-00003-of-00004.gguf:", *self.shards)


if __name__ == "__main__":
    unittest.main()
