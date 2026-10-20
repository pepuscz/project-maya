"""serve/test_multi.py - several conversations at once (READY ... multi=<n>): StrataEngine's per-request routing over a
scripted engine (serve/fake_multi_engine.py), and the service's FIFO with n places.

    python -m unittest serve.test_multi -v
"""
from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, FairLock, Service, StrataEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Engine(unittest.TestCase):
    def setUp(self):
        self.engine = StrataEngine(str(ROOT / "serve/fake_multi_engine.py"), ["3"])

    def tearDown(self):
        self.engine.close()

    def test_ready_says_how_many(self):
        self.assertEqual(self.engine.multi, 3)

    def test_concurrent_requests_each_get_their_own_tokens(self):
        out = {}

        def one(k):
            ids = list(range(k * 100, k * 100 + 40))
            toks = [t for t in self.engine.generate(ids, 40, {}, threading.Event()) if t is not None]
            out[k] = (ids, toks)

        ths = [threading.Thread(target=one, args=(k,)) for k in range(3)]
        [t.start() for t in ths]
        [t.join(30) for t in ths]
        for k, (ids, toks) in out.items():
            self.assertEqual(toks, [i + 1000 for i in ids], k)
        self.assertEqual(len(out), 3)

    def test_a_consumer_that_stops_ends_only_its_request(self):
        out = {}

        def long_one():
            ids = list(range(500, 600))
            out["long"] = [t for t in self.engine.generate(ids, 100, {}, threading.Event()) if t is not None]

        th = threading.Thread(target=long_one)
        th.start()
        gen = self.engine.generate(list(range(10)), 50, {}, threading.Event())
        got = []
        for t in gen:
            if t is not None:
                got.append(t)
            if len(got) == 5:
                break
        gen.close()                                   # STOP <rid>, drained to its DONE
        self.assertEqual(self.engine.last.get("finish"), "cancel")
        th.join(30)
        self.assertEqual(out["long"], [i + 1000 for i in range(500, 600)])
        self.assertEqual(got, [1000, 1001, 1002, 1003, 1004])
        self.assertEqual(self.engine._routes, {})

    def test_an_engine_error_reaches_its_request(self):
        # a fourth request on a 3-state engine is refused for that request alone
        evs = [threading.Event() for _ in range(3)]
        gens = [self.engine.generate(list(range(200)), 200, {}, evs[i]) for i in range(3)]
        for g in gens:
            next(g)                                   # their GEN lines are out
        with self.assertRaises(ValueError) as cm:
            list(self.engine.generate([1, 2], 2, {}, threading.Event()))
        self.assertIn("busy", str(cm.exception))
        for g in gens:
            g.close()


class AnswerTokenizer(ByteTokenizer):
    """The fake engine answers with ids from 1000 up: each is one letter here."""

    def token_bytes(self, t):
        return b"x"


class OwnFigures(unittest.TestCase):
    """A request's DONE figures are its own (timings, cached_tokens, the Monitor's history).  The requests decode in
    lockstep, so their DONE lines come within a step of each other: here B's DONE is always read after A's and before
    A reads its figures, which is when one shared value gave A the figures of B."""

    def setUp(self):
        self.engine = StrataEngine(str(ROOT / "serve/fake_multi_engine.py"), ["2"])
        a_read, self.b_ended = threading.Event(), threading.Event()
        parse = self.engine._parse_done

        def in_order(line):
            me = threading.current_thread().name
            if me == "b":
                a_read.wait(30)
            parse(line)
            if me == "a":
                a_read.set()
                self.b_ended.wait(30)
        self.engine._parse_done = in_order

    def tearDown(self):
        self.engine.close()

    def both(self, one):
        """one(prompt) on thread a (5 tokens: its answer ends first) and on thread b (9 tokens) -> {name: result}."""
        out = {}

        def run(name, n):
            try:
                out[name] = one(list(range(n)))
            finally:
                if name == "b":
                    self.b_ended.set()

        ths = [threading.Thread(target=run, name=name, args=(name, n)) for name, n in (("a", 5), ("b", 9))]
        [t.start() for t in ths]
        [t.join(30) for t in ths]
        return out

    def test_a_request_reads_its_own_done_after_another_ended(self):
        def one(ids):
            list(self.engine.generate(ids, len(ids), {}, threading.Event()))
            return dict(self.engine.last)

        out = self.both(one)
        self.assertEqual((out["a"]["generated"], out["a"]["reused"]), (5, 4))
        self.assertEqual((out["b"]["generated"], out["b"]["reused"]), (9, 8))

    def test_timings_and_cached_tokens_are_the_requests_own(self):
        svc = Service(self.engine, AnswerTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.fifo.set_capacity(2)

        def one(ids):
            return [x for kind, x in svc.run(ids, False, None, len(ids), {}, threading.Event()) if kind == "done"][0]

        out = self.both(one)
        for name, n in (("a", 5), ("b", 9)):
            self.assertEqual((out[name]["completion_tokens"], out[name]["reused"]), (n, n - 1), name)
            self.assertEqual(out[name]["timings"]["cache_n"], n - 1, name)
        self.assertEqual(sorted((h["prompt_tokens"], h["reused"], h["engine_generated"]) for h in svc.history),
                         [(5, 4, 5), (9, 8, 9)])


class Places(unittest.TestCase):
    def test_n_requests_hold_n_places_and_a_whole_waits_for_all(self):
        lock = FairLock(3)
        for _ in range(3):
            self.assertTrue(lock.acquire(units=1, timeout=1))
        self.assertFalse(lock.acquire(units=1, blocking=False))
        got = []
        th = threading.Thread(target=lambda: (lock.acquire(), got.append("all"), lock.release()))
        th.start()
        while lock.waiting() < 1:
            time.sleep(0.001)
        lock.release(units=1)
        time.sleep(0.05)
        self.assertEqual(got, [])                     # the exclusive waiter needs every place
        lock.release(units=1)
        lock.release(units=1)
        th.join(5)
        self.assertEqual(got, ["all"])
        self.assertFalse(lock.locked())

    def test_a_waiter_for_all_holds_back_later_ones(self):
        lock = FairLock(2)
        lock.acquire(units=1)
        order = []
        a = threading.Thread(target=lambda: (lock.acquire(), order.append("all"), lock.release()))
        a.start()
        while lock.waiting() < 1:
            time.sleep(0.001)
        b = threading.Thread(target=lambda: (lock.acquire(units=1), order.append("one"), lock.release(units=1)))
        b.start()
        while lock.waiting() < 2:
            time.sleep(0.001)
        time.sleep(0.05)
        self.assertEqual(order, [])                   # a free place, but the first in line wants both
        lock.release(units=1)
        a.join(5)
        b.join(5)
        self.assertEqual(order, ["all", "one"])


if __name__ == "__main__":
    unittest.main()
