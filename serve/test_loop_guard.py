"""serve/test_loop_guard.py - the loop guard (serve/loop_guard.py): what it calls a loop and what not, and what the
server does with one - against engines that continue an answer from wherever the prompt says it got to.

    python -m unittest serve.test_loop_guard -v
"""
from __future__ import annotations

import itertools
import json
import os
import random
import re
import sys
import threading
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import loop_guard as LG  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (IM_END, LOOP_STOP_NOTE, ByteTokenizer, Service, check_request_sampling, loop_guard_setting,  # noqa: E402
                          serve)

ROOT = Path(__file__).resolve().parents[1]


class Words:
    """A stand-in for a BPE vocabulary: a word with its leading space, a run of punctuation or of white space."""
    PIECE = re.compile(r" ?\w+| ?[^\w\s]+|\s+")

    def __init__(self):
        self.ids: dict[str, int] = {}

    def __call__(self, text: str) -> list[int]:
        return [self.ids.setdefault(p, len(self.ids)) for p in self.PIECE.findall(text)]


def first_loop(ids):
    """(index of the token at which the detector fires, Loop) or None."""
    det = LG.LoopDetector()
    for i, t in enumerate(ids):
        loop = det.push(t)
        if loop is not None:
            return i, loop
    return None


def numbers_table(rows):
    return "| id | name | score | city |\n|---|---|---|---|\n" + "".join(
        f"| {i} | user{i * 7 % 13} | {(i * 37) % 100} | city{i % 5} |\n" for i in range(rows))


LONG_BLOCK = " ".join(random.Random(11).choices(   # a plan of several hundred tokens (Maya-S thought one 4 times)
    "check the metric logs then the deploy diff and the gc pauses before the p99 of each host".split(), k=600)) + "\n"

LEGIT = {
    "markdown table": numbers_table(60),
    "a long block twice": "Draft:\n" + LONG_BLOCK + "Final:\n" + LONG_BLOCK + "Done.",
    "JSON array": json.dumps(list(range(300))),
    "JSON objects": json.dumps([{"id": i, "name": f"item {i}", "tags": ["a", "b"], "price": i * 3}
                                for i in range(40)], indent=2),
    "one function name": "".join(f"result = process_item(items[{i}], options)\n" for i in range(80)),
    "similar code lines": "".join(f"    x[{i}] = a[{i}] * w[{i % 4}] + b[{i}];\n" for i in range(64)),
    "a block twice": ("def area(r):\n    \"\"\"The area of a circle of radius r.\"\"\"\n    return 3.14159 * r * r\n\n"
                      * 2) + "print(area(2))\n",
    "a short line three times": "Hello there, world!\n" * 3 + "Done.",
    "a short row of zeros": "board = [" + ", ".join(["0"] * 30) + "]\n",
    "prose": " ".join(random.Random(7).choices(
        "the a model token answer loop guard server engine test of to and in on with for is are was".split(), k=3000)),
}


class Detection(unittest.TestCase):
    def test_real_loops_fire(self):
        w = Words()
        cases = {   # text before the loop, the block, at least how many copies there are
            "number sequence": ("The values are: ", "1017, 245, 26, 100, 47, 187, ", 40),
            "sentence in thinking": ("Let me think. ", "Wait, I should double-check the second step again. ", 20),
            "paragraph": ("Summary:\n", "- Also: logging added to a path that previously short-circuited (e.g., "
                                        "cached response now logs and populates cache differently)\n", 10),
            "code block": ("```python\n", "def step(state):\n    state.count += 1\n    if state.count > 10:\n"
                                          "        state.reset()\n    return state\n\n", 10),
            "one token": ("Result: ", "0 ", 300),
            "a long plan in thinking": ("Let me plan. ", LONG_BLOCK, 3),
        }
        for name, (head, block, n) in cases.items():
            with self.subTest(name):
                h, b = w(head), w(block)
                ids = h + b * n
                got = first_loop(ids)
                self.assertIsNotNone(got, name)
                at, loop = got
                p = len(b)
                period = p if p >= LG.P_MIN else p * -(-LG.P_MIN // p)    # (a short one as its multiple)
                self.assertEqual(loop.period, period)
                reps = LG.REPS_SHORT if period < LG.SHORT_BELOW else LG.REPS_LONG
                copies = max(reps, -(-LG.MIN_SPAN // period))
                # it fires on the last token of the copies it needs, and not before
                self.assertLessEqual(at + 1 - len(h), copies * period + period)
                self.assertGreaterEqual(loop.copies, reps)
                self.assertGreaterEqual(loop.repeats + loop.period, LG.MIN_SPAN)

    def test_legit_repetition_does_not_fire(self):
        for name, text in LEGIT.items():
            with self.subTest(name):
                self.assertIsNone(first_loop(Words()(text)), name)
        # byte by byte too (the tests' tokenizer below), the harder case: a byte block is longer in tokens
        for name, text in LEGIT.items():
            if name != "prose":
                with self.subTest(name + " (bytes)"):
                    self.assertIsNone(first_loop(list(text.encode())), name)

    def test_hold_covers_the_repeats(self):
        w = Words()
        head, block = w("Here we go:\n"), w("alpha beta gamma delta epsilon zeta eta theta iota kappa ")
        det, held = LG.LoopDetector(), []
        for t in head + block * 12:
            loop = det.push(t)
            if loop is not None:
                # everything that is dropped was still held back a token earlier
                self.assertGreaterEqual(det.hold, loop.repeats)
                break
            held.append(det.hold)
        self.assertIsNotNone(loop)
        self.assertEqual(held[:len(head) + len(block)], [0] * (len(head) + len(block)))   # nothing repeats yet
        prose = Words()(LEGIT["prose"])
        det = LG.LoopDetector()
        holds = [det.push(t) or det.hold for t in prose]
        # ordinary text: hardly held at all (this prose draws on 22 words, so a 2048-token window finds a few more
        # chance runs of a token or two than 400 did: 2.01 against 1.9; recorded answers hold back the same)
        self.assertLess(sum(holds) / len(holds), 2.5)

    def test_reset_keeps_checking(self):
        w = Words()
        block = w("one two three four five six seven eight nine ten eleven twelve ")
        det = LG.LoopDetector()
        ids = w("start\n") + block * 9
        loops = [det.push(t) for t in ids]
        self.assertIsNotNone(loops[-1])
        kept = ids[:len(ids) - loops[-1].repeats]
        det.reset(kept)
        self.assertEqual(det.hold, 0)
        self.assertTrue(any(det.push(t) for t in block * 9))       # the same loop again is seen again
        # what is kept may end in copies that were already sent: a new loop still needs copies of its own
        det.reset(ids[:-1])
        firsts = [i for i, t in enumerate(block * 9) if det.push(t)]
        self.assertTrue(firsts)
        self.assertGreaterEqual(firsts[0] + 1, 3 * len(block))


class Policy(unittest.TestCase):
    def loop_ids(self, n=12):
        return list(range(100, 100 + 20)) * n

    def fire(self, g, ids):
        for t in ids:
            act = g.push(t)
            if act is not None:
                return act
        return None

    def test_answer_penalty_then_stop(self):
        g = LG.LoopGuard()
        a1 = self.fire(g, self.loop_ids())
        self.assertEqual((a1.kind, a1.where, a1.period), (LG.PENALTY, "answer", 20))
        g.resume(list(range(100, 120)), a1)
        a2 = self.fire(g, self.loop_ids())
        self.assertEqual(a2.kind, LG.STOP)

    def test_thinking_penalties_then_close_then_stop(self):
        g = LG.LoopGuard(think_end=7, in_think=True)
        for _ in range(LG.THINK_PENALTIES):                         # the penalty first, twice
            a = self.fire(g, self.loop_ids())
            self.assertEqual((a.kind, a.where), (LG.PENALTY, "thinking"))
            g.resume([], a)
        a2 = self.fire(g, self.loop_ids())
        self.assertEqual((a2.kind, a2.where), (LG.CLOSE_THINKING, "thinking"))
        g.resume([7], a2)
        self.assertFalse(g.in_think)
        a3 = self.fire(g, self.loop_ids())
        self.assertEqual((a3.kind, a3.where), (LG.STOP, "answer"))  # the fourth in all

    def test_never_a_stop_while_thinking(self):
        g = LG.LoopGuard(think_end=7, in_think=True)
        kinds = []
        for _ in range(6):
            a = self.fire(g, self.loop_ids(20))
            kinds.append(a.kind)
            if a.kind == LG.CLOSE_THINKING:
                break
            g.resume([], a)
        self.assertEqual(kinds, [LG.PENALTY] * LG.THINK_PENALTIES + [LG.CLOSE_THINKING])

    def test_thinking_needs_more_copies(self):
        ids = list(range(100, 140)) * 3 + [5] * 50                  # a 40-token block 3 times: a loop in the answer
        self.assertIsNotNone(self.fire(LG.LoopGuard(), ids))
        self.assertIsNone(self.fire(LG.LoopGuard(think_end=7, in_think=True), ids))   # not yet in the thinking
        self.assertIsNotNone(self.fire(LG.LoopGuard(think_end=7, in_think=True), list(range(100, 140)) * 6))

    def test_the_hold_is_bounded(self):
        g = LG.LoopGuard()
        block = list(range(200, 200 + 500))                         # a 500-token block written again
        for t in block + block[:400]:
            self.assertIsNone(g.push(t))
            self.assertLessEqual(g.hold, LG.HOLD_MAX)
        self.assertEqual(g.hold, LG.HOLD_MAX)                       # (the detector's own run: 400)
        self.assertEqual(g.det.hold, 400)

    def test_a_loop_in_the_answer_after_one_in_the_thinking(self):
        g = LG.LoopGuard(think_end=7, in_think=True)
        g.resume([], self.fire(g, self.loop_ids()))
        self.assertIsNone(g.push(7))                                # the model closes its thinking itself
        a2 = self.fire(g, self.loop_ids())
        self.assertEqual((a2.kind, a2.where), (LG.PENALTY, "answer"))   # the answer's first: a penalty
        g.resume([], a2)
        self.assertEqual(self.fire(g, self.loop_ids()).kind, LG.STOP)   # the answer's second

    def test_no_thinking_token_no_thinking_close(self):
        g = LG.LoopGuard(think_end=None, in_think=True)            # (a model without one: the answer's rules)
        g.resume([], self.fire(g, self.loop_ids()))
        self.assertEqual(self.fire(g, self.loop_ids()).kind, LG.STOP)

    def test_sampling(self):
        g = LG.LoopGuard()
        base = {"temperature": 0, "top_k": 20}
        self.assertIs(g.sampling(base, 0, 500)[0], base)            # before a loop: the request's own, as it is
        self.assertEqual(g.sampling(base, 0, 500)[1], 500)
        act = self.fire(g, self.loop_ids())
        kept = list(range(1000))
        g.resume(kept, act)
        smp, n = g.sampling(base, len(kept), 5000)
        self.assertEqual(smp, {"temperature": 0, "top_k": 20, "repetition_penalty": LG.LOOP_PENALTY,
                               "penalty_last_n": LG.PENALTY_MIN_N})
        self.assertEqual(n, 2 * 20 + LG.PENALTY_EXTRA)
        self.assertEqual(base, {"temperature": 0, "top_k": 20})    # (not changed in place)
        self.assertEqual(g.sampling(base, len(kept), 10)[1], 10)   # never past what is left
        self.assertIs(g.sampling(base, len(kept) + n, 500)[0], base)   # the window over: the request's own
        # a request's own stronger penalty or longer window stays
        smp, _ = g.sampling({"repetition_penalty": 1.3, "penalty_last_n": -1}, len(kept), 5000)
        self.assertEqual((smp["repetition_penalty"], smp["penalty_last_n"]), (1.3, -1))
        smp, _ = g.sampling({"penalty_last_n": 0}, len(kept), 5000)   # "no penalties": the guard's own window
        self.assertEqual(smp["penalty_last_n"], LG.PENALTY_MIN_N)
        self.assertEqual(LG.penalty_window(300), 600)


def chars(text):
    """A text as ids (its code points); the guard's `piece` is chr."""
    return [ord(c) for c in text]


class ToolCalls(unittest.TestCase):
    """The guard never cuts inside a tool call: calls written alike, three in a row, stay whole."""
    CALL = "<tool_call>read_file<arg_key>path</arg_key><arg_value>/src/main.py</arg_value></tool_call>"
    PROSE = "xyz" + "1017, 245, 26, 100, 47, 187, " * 12
    SHORT = "xyz" + "1017, 245, 26, 100, 47, 187, " * 4     # (one loop, no more)

    def guard(self, **kw):
        return LG.ToolSafeGuard(piece=chr, **kw)

    def feed(self, g, text):
        """Every action the guard returns for `text`, and what a server keeps (the ids minus what an action drops)."""
        acts, kept = [], []
        for t in chars(text):
            kept.append(t)
            act = g.push(t, len(kept))
            if act is not None:
                acts.append(act)
                del kept[len(kept) - act.drop:]
                g.resume(kept, act)
                if act.kind == LG.STOP:
                    break
        return acts, "".join(map(chr, kept))

    def test_three_identical_calls_stay_whole(self):
        text = "Reading the files.\n" + (self.CALL + "\n") * 3 + "Done."
        self.assertIsNotNone(first_loop(chars(text)))       # (the plain detector does see it as a loop)
        plain, _ = self.feed(LG.LoopGuard(), text)
        self.assertTrue(plain and plain[0].drop > 0)
        g = self.guard()
        acts, kept = self.feed(g, text)
        self.assertEqual((acts, kept), ([], text))
        self.assertEqual(kept.count(self.CALL), 3)
        self.assertEqual((g.events, g.hits), ([], {"thinking": 0, "answer": 0}))

    def test_three_identical_calls_with_json_arguments_stay_valid_json(self):
        args = json.dumps({"path": "/src/main.py", "lines": [1, 2, 3]})
        call = "<tool_call>read_file<arg_key>args</arg_key><arg_value>" + args + "</arg_value></tool_call>"
        acts, kept = self.feed(self.guard(), "\n".join(["ok"] + [call] * 4))
        self.assertEqual(acts, [])
        bodies = re.findall(r"<arg_value>(.*?)</arg_value>", kept)
        self.assertEqual([json.loads(b) for b in bodies], [json.loads(args)] * 4)

    def test_calls_of_the_other_families(self):
        ns = "]<]minimax[>["
        mm = (f'{ns}<tool_call>{ns}<invoke name="read_file">{ns}<path>/src/main.py{ns}</path>{ns}</invoke>'
              f"{ns}</tool_call>")
        one = ('<｜DSML｜ invoke name="read_file"><｜DSML｜ parameter name="path" string="true">/src/main.py'
               "</｜DSML｜ parameter></｜DSML｜ invoke>")
        dsml = "<｜DSML｜ calls>" + one * 3 + "</｜DSML｜ calls>"
        for text in ("Reading.\n" + (mm + "\n") * 3 + "Done.", "Reading.\n" + dsml + "\nDone.",
                     "Reading.\n" + (dsml + "\n") * 3 + "Done."):
            g = self.guard()
            acts, kept = self.feed(g, text)
            self.assertEqual(kept, text)
            self.assertTrue(all(a.kind == LG.PENALTY and a.drop == 0 for a in acts), acts)
            self.assertEqual(g.hits, {"thinking": 0, "answer": 0})

    def test_a_loop_in_prose_is_still_cut(self):
        text = "Reading the files.\n" + self.PROSE
        acts, kept = self.feed(self.guard(), text)
        plain, plain_kept = self.feed(LG.LoopGuard(), text)
        self.assertEqual([(a.kind, a.drop) for a in acts], [(a.kind, a.drop) for a in plain])
        self.assertEqual(kept, plain_kept)
        self.assertEqual(acts[0].kind, LG.PENALTY)
        self.assertGreater(acts[0].drop, 0)
        self.assertLess(len(kept), len(text))

    def test_a_loop_in_prose_after_a_call_is_cut_and_the_call_stays(self):
        text = "Reading.\n" + self.CALL + "\nThe numbers:\n" + self.PROSE
        acts, kept = self.feed(self.guard(), text)
        self.assertEqual(acts[0].kind, LG.PENALTY)
        self.assertGreater(acts[0].drop, 0)
        self.assertIn(self.CALL, kept)
        self.assertLess(len(kept), len(text))

    def test_a_cut_in_prose_then_three_calls(self):
        g = self.guard()
        acts, kept = self.feed(g, self.SHORT)
        self.assertEqual([a.kind for a in acts], [LG.PENALTY])
        acts, kept2 = self.feed(g, "\n" + self.CALL * 3)
        self.assertEqual(kept2.count(self.CALL), 3)
        self.assertTrue(all(a.drop == 0 for a in acts))

    def test_a_loop_inside_one_call_only_gets_the_penalty_and_keeps_every_token(self):
        body = "".join(f"line {i % 3}: alpha beta gamma delta\n" for i in range(40))
        text = "<tool_call>write_file<arg_key>content</arg_key><arg_value>" + body + "</arg_value></tool_call>"
        g = self.guard(think_end=1, in_think=True)
        acts, kept = self.feed(g, text)
        self.assertEqual(kept, text)
        self.assertEqual([(a.kind, a.drop, a.where) for a in acts], [(LG.PENALTY, 0, "tool call")])
        self.assertEqual(g.hits, {"thinking": 0, "answer": 0})
        self.assertTrue(g.in_think)                         # (the thinking is not closed)
        smp, _ = g.sampling({"temperature": 0}, acts[0].at, 100)
        self.assertEqual(smp["repetition_penalty"], LG.LOOP_PENALTY)

    def test_without_a_piece_it_is_the_plain_guard(self):
        acts, _ = self.feed(LG.ToolSafeGuard(), "x" + self.CALL * 3)
        self.assertTrue(acts and acts[0].drop > 0)

    def test_a_marker_split_across_tokens(self):
        g = self.guard()
        for t in chars("<tool_"):
            g.push(t)
        self.assertFalse(g.in_tool)
        for t in chars("call>abc"):
            g.push(t)
        self.assertTrue(g.in_tool)
        for t in chars("</tool_call>"):
            g.push(t)
        self.assertFalse(g.in_tool)


class Settings(unittest.TestCase):
    def test_on_by_default_and_the_ways_to_turn_it_off(self):
        self.assertTrue(loop_guard_setting({}, {}))
        self.assertFalse(loop_guard_setting({}, {"MAYA_LOOP_GUARD": "0"}))
        self.assertFalse(loop_guard_setting({}, {"STRATA_LOOP_GUARD": "off"}))
        self.assertFalse(loop_guard_setting({"loop_guard": False}, {}))
        self.assertFalse(loop_guard_setting({"loop_guard": "false"}, {}))
        self.assertTrue(loop_guard_setting({"loop_guard": True}, {}))
        self.assertTrue(loop_guard_setting({"loop_guard": False}, {"MAYA_LOOP_GUARD": "1"}))   # the environment wins

    def test_the_request_field(self):
        for value, want in ((True, True), (False, False), ("false", False), ("on", True), (0, False), (None, None)):
            req = {"loop_guard": value}
            check_request_sampling(req)
            self.assertEqual(req.get("loop_guard"), want)
        with self.assertRaises(ValueError):
            check_request_sampling({"loop_guard": "maybe"})


class Tok(ByteTokenizer):
    SPECIALS = ByteTokenizer.SPECIALS + ["<think>", "</think>"]


class LoopingEngine:
    """One answer, continued from wherever the prompt says it got to (the first prompt + what was written, as the
    server's restarts send it): `head`, then `block` again and again - unless a call samples with a repetition penalty
    and there is an `escape`: then that (and the end), and later calls go on with the rest of it; once the prompt holds
    the end of thinking, `answer` and the end.  Records every call: (prompt, max_new, sampling)."""

    def __init__(self, tok, head, block, escape=None, answer=None, think_end=None, script=None):
        self.tok, self.max_context = tok, 1 << 20
        self.end = tok.encode(IM_END, parse_special=True)
        enc = lambda s: tok.encode(s, parse_special=True) if s is not None else None   # noqa: E731
        self.head, self.block, self.escape, self.answer = enc(head), enc(block), enc(escape), enc(answer)
        self.script = enc(script)                   # (a plain answer, no loop: head/block unused)
        self.think_end, self.base, self.escaped_at = think_end, None, None
        self.calls: list[tuple[list[int], int, dict]] = []
        self.made = 0                               # tokens written in all (the streaming test reads it)

    def _at(self, k):
        return self.head[k] if k < len(self.head) else self.block[(k - len(self.head)) % len(self.block)]

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls.append((list(ids), max_new, dict(sampling or {})))
        if self.base is None:
            self.base = list(ids)
        written = ids[len(self.base):]
        rp = (sampling or {}).get("repetition_penalty") or 1.0
        if self.script is not None:
            out = self.script[len(written):] + self.end
        elif self.think_end is not None and self.think_end in written:
            out = self.answer + self.end
        elif self.escaped_at is not None and len(written) >= self.escaped_at:
            out = self.escape[len(written) - self.escaped_at:] + self.end
        elif self.escape is not None and rp > 1.0:
            self.escaped_at = len(written)
            out = self.escape + self.end
        else:
            out = (self._at(len(written) + i) for i in itertools.count())
        for i, t in enumerate(out):
            if i >= max_new or cancel.is_set():
                return
            self.made += 1
            yield t


HEAD = "Here are the numbers you asked for:\n"    # (its end is not the block's: the first copy stays as it is)
BLOCK = "1017, 245, 26, 100, 47, 187, "
ESCAPE = "and so on.\n\nIn short, the sequence repeats with period six."


def kept_block(test, content, head=HEAD, after=""):
    """The block text that stays between `head` and `after`: the first copy, and what streamed before the loop was
    clear (the stream holds back at most HOLD_MAX tokens; the tests' tokens are bytes).  Returns it."""
    test.assertTrue(content.startswith(head) and content.endswith(after), content)
    body = content[len(head):len(content) - len(after)]
    test.assertTrue((BLOCK * 100).startswith(body), body)
    test.assertGreaterEqual(len(body), len(BLOCK))                  # the first copy stays
    return body


def run(svc, max_new=3000, sampling=None, thinking=False, prompt="go"):
    """svc.run's stream: (reasoning text, content text, done, [(content chars so far, engine tokens then)])."""
    ids = svc.tok.encode(prompt)
    reasoning, content, seen, done = "", "", [], None
    for kind, val in svc.run(ids, thinking, None, max_new, dict(sampling or {}), threading.Event()):
        if kind == "event" and val.kind == "reasoning":
            reasoning += val.text
        elif kind == "event" and val.kind == "content":
            content += val.text
            seen.append((len(content), svc.engine.made))
        elif kind == "done":
            done = val
    return reasoning, content, done, seen


class Server(unittest.TestCase):
    def service(self, engine, tok=None):
        tok = tok or engine.tok
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        return svc

    def test_a_loop_is_dropped_and_penalised_then_escapes(self):
        eng = LoopingEngine(Tok(), HEAD, BLOCK, escape=ESCAPE)
        svc = self.service(eng)
        _, content, done, _ = run(svc, sampling={"temperature": 0})
        body = kept_block(self, content, after=ESCAPE)
        # the copies it needs are 4 x 30 tokens: all but the last HOLD_MAX of them went out as they came
        self.assertLessEqual(len(body), 4 * len(BLOCK) - LG.HOLD_MAX)
        self.assertEqual(done["finish"], "stop")
        self.assertEqual(len(eng.calls), 2)
        (p1, n1, s1), (p2, n2, s2) = eng.calls
        self.assertEqual((n1, s1), (3000, {"temperature": 0}))     # the request's own until the loop
        period = len(BLOCK)
        self.assertEqual(p2, p1 + svc.tok.encode(HEAD + body))     # on from what was sent
        self.assertEqual(s2, {"temperature": 0, "repetition_penalty": LG.LOOP_PENALTY,
                              "penalty_last_n": max(LG.PENALTY_MIN_N, 2 * period)})
        self.assertEqual(n2, 2 * period + LG.PENALTY_EXTRA)
        m = svc.metrics()
        self.assertEqual({k: m["loop_guard"][k] for k in ("detected", "penalised", "stopped")},
                         {"detected": 1, "penalised": 1, "stopped": 0})
        self.assertTrue(m["loop_guard"]["on"])
        self.assertEqual(len(m["requests"][0]["loop_guard"]), 1)
        self.assertIn(f"a {period}-token block", m["requests"][0]["loop_guard"][0])
        self.assertIn("maya:loop_guard_detected_total 1", svc.prometheus())

    def test_three_identical_tool_calls_in_one_reply_are_not_cut(self):
        call = "<tool_call>read_file<arg_key>path</arg_key><arg_value>/src/main.py</arg_value></tool_call>"
        script = "Reading the file three times.\n" + (call + "\n") * 3
        eng = LoopingEngine(Tok(), None, None, script=script)
        svc = self.service(eng)
        calls, text = [], ""
        for kind, val in svc.run(svc.tok.encode("go"), False, [{"name": "read_file", "parameters": {
                "type": "object", "properties": {"path": {"type": "string"}}}}], 3000, {}, threading.Event()):
            if kind == "event" and val.kind == "tool_call" and val.complete:
                calls.append(json.dumps(val.call.arguments))
            elif kind == "event" and val.kind == "content":
                text += val.text
        self.assertEqual(calls, [json.dumps({"path": "/src/main.py"})] * 3)
        self.assertEqual(len(eng.calls), 1)                 # (no restart: the guard did nothing)
        m = svc.metrics()
        self.assertEqual({k: m["loop_guard"][k] for k in ("detected", "penalised", "thinking_closed", "stopped")},
                         dict.fromkeys(("detected", "penalised", "thinking_closed", "stopped"), 0))

    def test_the_request_sampling_comes_back_after_the_window(self):
        long_escape = " ".join(f"Step {i}: the term at {i} is {i * i % 97}." for i in range(30))
        eng = LoopingEngine(Tok(), HEAD, BLOCK, escape=long_escape)
        svc = self.service(eng)
        _, content, done, _ = run(svc, sampling={"temperature": 0.7, "seed": 5})
        body = kept_block(self, content, after=long_escape)
        self.assertEqual(done["finish"], "stop")
        self.assertEqual(len(eng.calls), 3)
        window = 2 * len(BLOCK) + LG.PENALTY_EXTRA
        self.assertEqual(eng.calls[1][1], window)
        self.assertEqual(eng.calls[2][2], {"temperature": 0.7, "seed": 5})   # the request's own again
        written = len(svc.tok.encode(HEAD + body)) + window
        self.assertEqual(eng.calls[2][1], 3000 - written)

    def test_a_loop_that_persists_in_the_answer_ends_it(self):
        eng = LoopingEngine(Tok(), HEAD, BLOCK)                     # loops whatever the penalty
        svc = self.service(eng)
        _, content, done, _ = run(svc)
        body = kept_block(self, content)
        self.assertLessEqual(len(body), 2 * 4 * len(BLOCK))         # two loops' copies at most, then the end
        self.assertEqual(done["finish"], "stop")
        self.assertLess(done["completion_tokens"], 300)
        self.assertEqual(len(eng.calls), 2)
        self.assertEqual(svc.loop_guard_counts, {"detected": 2, "penalised": 1, "thinking_closed": 0, "stopped": 1})

    def test_a_third_loop_in_the_thinking_closes_it(self):
        tok = Tok()
        think_end = tok.encode("</think>", parse_special=True)[0]
        eng = LoopingEngine(tok, "I need to list them:\n", BLOCK, answer="\n\nThe numbers: 1017, 245, 26.",
                            think_end=think_end)
        svc = self.service(eng)
        self.assertEqual(svc.think_end_id, think_end)
        reasoning, content, done, _ = run(svc, thinking=True)
        kept_block(self, reasoning, head="I need to list them:\n")
        self.assertEqual(content.strip(), "The numbers: 1017, 245, 26.")
        self.assertEqual(done["finish"], "stop")
        self.assertEqual(len(eng.calls), 4)                         # two penalties first, then the close
        self.assertEqual(eng.calls[3][0][-1], think_end)           # the thinking closed by its own token
        self.assertEqual(eng.calls[3][0].count(think_end), 1)
        self.assertEqual(svc.loop_guard_counts, {"detected": 3, "penalised": 2, "thinking_closed": 1, "stopped": 0})

    def test_a_stop_before_any_answer_text_says_so(self):
        # a model with no end-of-thinking token that loops in its reasoning: the guard can only end it - never with
        # finish "stop" and no answer at all (a chat client shows the thinking and nothing else)
        eng = LoopingEngine(Tok(), "I need to list them:\n", BLOCK)
        svc = self.service(eng)
        svc.think_end_id = None
        reasoning, content, done, _ = run(svc, thinking=True)
        self.assertTrue(reasoning.startswith("I need to list them:\n" + BLOCK))
        self.assertEqual(content, LOOP_STOP_NOTE)
        self.assertEqual(done["finish"], "stop")

    def test_a_copied_block_streams_on(self):
        # a long block the model writes again (a draft in the thinking, then the answer): not a loop, and never held
        # back whole - the stream is at most HOLD_MAX tokens behind the engine
        draft = "".join(f"step {i}: check item {i * 7 % 23} and log it\n" for i in range(12))
        eng = LoopingEngine(Tok(), "", "", script="Draft:\n" + draft + "Final:\n" + draft + "Done.")
        svc = self.service(eng)
        _, content, done, seen = run(svc)
        self.assertEqual(content, "Draft:\n" + draft + "Final:\n" + draft + "Done.")
        self.assertGreater(len(draft), 4 * LG.HOLD_MAX)
        self.assertLessEqual(max(made - chars for chars, made in seen), LG.HOLD_MAX + 8)
        self.assertEqual(svc.loop_guard_counts["detected"], 0)

    def test_streaming_holds_back_only_the_repeats(self):
        eng = LoopingEngine(Tok(), HEAD, BLOCK, escape=ESCAPE)
        svc = self.service(eng)
        _, content, _, seen = run(svc)
        self.assertEqual(content.count(BLOCK), 1)                  # a dropped copy never reached the client
        # what is not a repeat goes out as it comes: the head while the engine is still writing it
        n_head = len(svc.tok.encode(HEAD))
        early = [made for chars, made in seen if chars <= len(HEAD) - 4]
        self.assertTrue(early)
        self.assertLessEqual(max(made - chars for chars, made in seen if chars <= len(HEAD) - 4), 8)
        self.assertLessEqual(max(early), n_head + 4)

    def test_no_loop_same_calls_and_same_answer_on_and_off(self):
        # (shorter than LEGIT's: the tests' byte tokenizer decodes the whole answer again for every token)
        text = (numbers_table(12) + json.dumps(list(range(60))) + "\n" +
                "".join(f"result = process_item(items[{i}], options)\n" for i in range(12)) +
                "".join(f"    x[{i}] = a[{i}] * w[{i % 4}] + b[{i}];\n" for i in range(12)) +
                LEGIT["a block twice"] + LEGIT["a short line three times"])
        out = {}
        for mode in ("server on", "server off", "request off", "request on"):
            eng = LoopingEngine(Tok(), "", "", script=text)
            svc = self.service(eng)
            svc.loop_guard = mode != "server off"
            sampling = {"temperature": 0.6, "top_p": 0.95, "seed": 3}
            if mode.startswith("request"):
                svc.loop_guard = mode == "request off"              # (the request's field wins both ways)
                sampling["loop_guard"] = mode == "request on"
            r, c, done, _ = run(svc, max_new=20000, sampling=sampling)
            calls = [(p, n, {k: v for k, v in s.items() if k != "loop_guard"}) for p, n, s in eng.calls]
            out[mode] = (r, c, done["finish"], done["completion_tokens"], calls, svc.loop_guard_counts["detected"])
        self.assertEqual(out["server on"][1], text)
        for mode in ("server off", "request off", "request on"):
            self.assertEqual(out[mode], out["server on"], mode)
        self.assertEqual(len(out["server on"][4]), 1)              # one engine call, its sampling the request's

    def test_off_lets_the_loop_run(self):
        eng = LoopingEngine(Tok(), HEAD, BLOCK, escape=ESCAPE)
        svc = self.service(eng)
        svc.loop_guard = False
        _, content, done, _ = run(svc, max_new=600)
        self.assertEqual(done["finish"], "length")
        self.assertGreater(content.count(BLOCK), 15)
        self.assertEqual(len(eng.calls), 1)


class Http(unittest.TestCase):
    """The same through /v1/chat/completions: streamed deltas and the whole answer agree, one copy of the block."""

    @classmethod
    def setUpClass(cls):
        cls.eng = LoopingEngine(Tok(), HEAD, BLOCK, escape=ESCAPE)
        cls.svc = Service(cls.eng, cls.eng.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, body):
        self.eng.base, self.eng.escaped_at = None, None             # (a new conversation each time)
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read().decode()

    def test_whole_and_streamed(self):
        msg = {"messages": [{"role": "user", "content": "numbers"}], "max_tokens": 2000,
               "chat_template_kwargs": {"enable_thinking": False}}
        whole = json.loads(self.post(msg))
        body = kept_block(self, whole["choices"][0]["message"]["content"], after=ESCAPE)
        self.assertEqual(whole["choices"][0]["finish_reason"], "stop")
        text = ""
        for line in self.post(dict(msg, stream=True)).splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                for ch in json.loads(line[6:]).get("choices") or []:
                    text += (ch.get("delta") or {}).get("content") or ""
        self.assertEqual(text, HEAD + body + ESCAPE)                # streamed and whole alike
        off = json.loads(self.post(dict(msg, loop_guard=False, max_tokens=400)))
        self.assertEqual(off["choices"][0]["finish_reason"], "length")
        self.assertGreater(off["choices"][0]["message"]["content"].count(BLOCK), 8)


class GuardedUsage(unittest.TestCase):
    """After the guard, the engine goes on from prompt + the kept tokens: the request reports its own prompt's
    figures, not that call's (a guarded MiniMax answer said 5,919 cached tokens of a 246-token prompt)."""

    def test_closed_resumed_stream_keeps_initial_prompt_cache(self):
        class TimedEngine(LoopingEngine):
            def __init__(self, initial_cache):
                super().__init__(Tok(), HEAD, BLOCK, escape=ESCAPE)
                self.initial_cache, self.last = initial_cache, {}

            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                first, made = not self.calls, 0
                try:
                    for token in super().generate(ids, max_new, sampling, cancel, embeddings):
                        made += 1
                        yield token
                finally:  # the native engine's DONE is parsed while generate.close() drains it
                    self.last = dict(prompt_tokens=len(ids), reused=self.initial_cache if first else len(self.base),
                                     generated=made, prompt_ms=100.0, decode_ms=100.0, finish="cancel")

        for cached in (0, 17):
            with self.subTest(initial_cached=cached):
                engine = TimedEngine(cached)
                svc = Service(engine, engine.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
                stream = svc._sliced([42] * 125, 3000, {"temperature": 0}, threading.Event(), None,
                                     [True, True], {}, LG.LoopGuard(svc.think_end_id, False))
                for token in stream:
                    if len(engine.calls) >= 2:
                        break  # Service.run does this on a stop token, tool stop, or client disconnect
                stream.close()
                self.assertEqual(len(engine.calls), 2)
                self.assertEqual((engine.last["prompt_tokens"], engine.last["reused"]), (125, cached))
                self.assertEqual((engine.last["prompt_ms"], engine.last["decode_ms"]), (200.0, 200.0))
                self.assertGreater(engine.last["generated"], 2)  # includes the first segment too

    def test_error_before_done_does_not_relabel_prior_request(self):
        class FailedEngine(LoopingEngine):
            def generate(self, *args, **kwargs):
                raise ValueError("before any DONE")
                yield
        engine = FailedEngine(Tok(), HEAD, BLOCK)
        prior = dict(prompt_tokens=1000, reused=700)
        engine.last = prior
        svc = Service(engine, engine.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        stream = svc._sliced([42] * 125, 3000, {}, threading.Event(), None, [True, True], {}, None)
        with self.assertRaises(ValueError):
            list(stream)
        self.assertIs(engine.last, prior)

    def test_the_request_prompt_not_the_resumed_call(self):
        from serve.server import own_prompt_figures
        got = own_prompt_figures({"prompt_tokens": 246 + 5546, "reused": 5546 + 127, "prompt_ms": 9.5}, 246)
        self.assertEqual((got["prompt_tokens"], got["reused"], got["prompt_ms"]), (246, 127, 9.5))
        self.assertEqual(own_prompt_figures({"prompt_tokens": 1300, "reused": 300}, 1000)["reused"], 0)
        self.assertEqual(own_prompt_figures({"prompt_tokens": 1300, "reused": 5000}, 1000)["reused"], 1000)


if __name__ == "__main__":
    unittest.main()
