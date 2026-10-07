"""Unit tests for CandidatePool (in-memory scores) and ScoreWriter (DB scores).

No live database is needed: the stores are replaced by a ``FakeScoreDB`` that
mimics the semantics of ``update_train_priority`` and
``fetch_candidate_pool_from_local_shards`` (including the SQL-side decay), so
the tests can also exercise the pool <-> writer feedback loop over time.
"""
import math
import threading
from contextlib import contextmanager
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from patchsorter.dl import datasets, scoring
from patchsorter.dl.datasets import GT_SCORE_IN_MEMORY_DECAY, CandidatePool
from patchsorter.dl.scoring import ScoreWriter

PROJECT_ID = 1


class FakeScoreDB:
    """Stand-in for the patch table: ``patch_id -> train_priority`` (None = never scored)."""

    def __init__(self, priorities):
        self.priorities = dict(priorities)
        self.update_calls = []
        self.fetch_calls = 0
        self.lock = threading.Lock()
        self.fail_next_update = False

    def update_train_priority(self, updates):
        with self.lock:
            if self.fail_next_update:
                self.fail_next_update = False
                raise RuntimeError("db down")
            self.update_calls.append(list(updates))
            for pid, prio in updates:
                self.priorities[pid] = prio
            return len(updates)

    def fetch_candidates(self, limit, decay, init_score):
        with self.lock:
            self.fetch_calls += 1
            seen = {p: (None if v is None else math.floor(v)) for p, v in self.priorities.items()}
            max_seen = max((s for s in seen.values() if s is not None), default=0)
            rows = []
            for pid, prio in self.priorities.items():
                if prio is None:
                    score = init_score
                else:
                    score = (prio - seen[pid]) * max(0.0001, math.exp(-decay * (max_seen - seen[pid])))
                rows.append({"patch_id": pid, "train_priority": prio, "computed_sorting_score": score})
            rows.sort(key=lambda r: r["computed_sorting_score"], reverse=True)
            return rows[:limit]


@pytest.fixture
def fake_db(monkeypatch):
    db = FakeScoreDB({1: None, 2: None, 3: None})

    class FakeWorkerPatchStore:
        def __init__(self, project_id, session):
            pass

        def fetch_candidate_pool_from_local_shards(self, shard_ids, limit, decay, init_score):
            return db.fetch_candidates(limit, decay, init_score)

    class FakePatchStore:
        def __init__(self, project_id, session):
            assert project_id == PROJECT_ID

        def update_train_priority(self, updates):
            return db.update_train_priority(updates)

    @contextmanager
    def session_ctx():
        yield MagicMock()

    worker_sm = MagicMock()
    worker_sm.get_session.side_effect = session_ctx
    monkeypatch.setattr(datasets.worker_client, "get_client", lambda: worker_sm)
    monkeypatch.setattr(datasets, "WorkerPatchStore", FakeWorkerPatchStore)
    monkeypatch.setattr(scoring, "PatchStore", FakePatchStore)
    return db


@pytest.fixture
def head_sm():
    @contextmanager
    def session_ctx():
        yield MagicMock()

    sm = MagicMock()
    sm.get_session.side_effect = session_ctx
    return sm


def make_pool(pool_size=2048):
    shards = MagicMock()
    shards.get_table_a_shard_list.return_value = [101, 102]
    return CandidatePool(PROJECT_ID, shards, pool_size=pool_size)


def make_writer(head_sm, **kw):
    kw.setdefault("flush_interval_s", 0.05)
    return ScoreWriter(head_sm, PROJECT_ID, **kw)


# --------------------------------------------------------------------------
# CandidatePool
# --------------------------------------------------------------------------

class TestCandidatePool:
    def test_empty_before_refresh(self, fake_db):
        pool = make_pool()
        assert pool.is_empty
        assert pool.draw_batch(5) == []

    def test_refresh_loads_rows_and_scores(self, fake_db):
        fake_db.priorities = {1: 5.5, 2: 5.2, 3: None}
        pool = make_pool()
        pool.refresh()

        assert not pool.is_empty
        assert {r["patch_id"] for r in pool._rows} == {1, 2, 3}
        # Never-scored row gets init score, ranked first.
        assert pool._rows[0]["patch_id"] == 3
        np.testing.assert_allclose(pool._scores, [r["computed_sorting_score"] for r in pool._rows])
        assert pool._scores.dtype == np.float64

    def test_refresh_respects_pool_size_and_passes_decay(self, fake_db):
        pool = make_pool(pool_size=2)
        pool.refresh()
        assert len(pool._rows) == 2

    def test_refresh_replaces_previous_state(self, fake_db):
        pool = make_pool()
        pool.refresh()
        pool.decay_in_memory_score(0)
        fake_db.priorities = {1: 3.5}
        pool.refresh()
        assert [r["patch_id"] for r in pool._rows] == [1]
        np.testing.assert_allclose(pool._scores, [0.5])

    def test_refresh_to_empty(self, fake_db):
        pool = make_pool()
        pool.refresh()
        fake_db.priorities = {}
        pool.refresh()
        assert pool.is_empty

    def test_draw_batch_distinct_and_capped(self, fake_db):
        pool = make_pool()
        pool.refresh()
        picks = pool.draw_batch(10)
        idxs = [i for _, i in picks]
        assert len(picks) == 3
        assert len(set(idxs)) == 3
        for row, idx in picks:
            assert row is pool._rows[idx]

    def test_draw_batch_biased_by_score(self, fake_db):
        fake_db.priorities = {1: 1.99, 2: 1.001}
        pool = make_pool()
        pool.refresh()
        np.random.seed(0)
        hits = sum(pool.draw_batch(1)[0][0]["patch_id"] == 1 for _ in range(300))
        assert hits > 250

    def test_draw_batch_zero_weights_falls_back_to_uniform(self, fake_db):
        fake_db.priorities = {1: 4.0, 2: 4.0}  # fractional part 0 -> score 0
        pool = make_pool()
        pool.refresh()
        assert pool._scores.sum() == 0 or np.allclose(pool._scores, 0)
        assert len(pool.draw_batch(2)) == 2

    def test_draw_batch_nonfinite_weights_falls_back_to_uniform(self, fake_db):
        pool = make_pool()
        pool.refresh()
        pool._scores[0] = np.nan
        assert len(pool.draw_batch(3)) == 3

    def test_decay_in_memory_score(self, fake_db):
        pool = make_pool()
        pool.refresh()
        before = pool._scores.copy()
        pool.decay_in_memory_score(1)
        assert pool._scores[1] == pytest.approx(before[1] * GT_SCORE_IN_MEMORY_DECAY)
        assert pool._scores[0] == before[0]
        pool.decay_in_memory_score(1)
        assert pool._scores[1] == pytest.approx(before[1] * GT_SCORE_IN_MEMORY_DECAY ** 2)

    @pytest.mark.parametrize("idx", [-1, 3, 100])
    def test_decay_out_of_range_is_noop(self, fake_db, idx):
        pool = make_pool()
        pool.refresh()
        before = pool._scores.copy()
        pool.decay_in_memory_score(idx)
        np.testing.assert_array_equal(pool._scores, before)

    def test_repeated_decay_shifts_draws_to_other_candidates(self, fake_db):
        fake_db.priorities = {1: 1.9, 2: 1.8}
        pool = make_pool()
        pool.refresh()
        idx_of = {r["patch_id"]: i for i, r in enumerate(pool._rows)}
        for _ in range(20):
            pool.decay_in_memory_score(idx_of[1])
        np.random.seed(1)
        hits = sum(pool.draw_batch(1)[0][0]["patch_id"] == 2 for _ in range(200))
        assert hits > 190

    def test_refresh_discards_in_memory_decay_and_picks_up_db_scores(self, fake_db):
        fake_db.priorities = {1: 7.9, 2: 7.1}
        pool = make_pool()
        pool.refresh()
        idx = {r["patch_id"]: i for i, r in enumerate(pool._rows)}
        pool.decay_in_memory_score(idx[1])
        decayed = pool._scores[idx[1]]

        fake_db.priorities[1] = 8.95  # DB re-scored meanwhile
        pool.refresh()
        idx = {r["patch_id"]: i for i, r in enumerate(pool._rows)}
        assert pool._scores[idx[1]] > decayed
        assert pool._rows[idx[1]]["train_priority"] == pytest.approx(8.95)


# --------------------------------------------------------------------------
# ScoreWriter
# --------------------------------------------------------------------------

class TestScoreWriter:
    def test_close_flushes_pending_to_db(self, fake_db, head_sm):
        w = make_writer(head_sm, batch_size=1000, flush_interval_s=30)
        w.enqueue(torch.tensor([1, 2]), torch.tensor([3.25, 4.5]))
        w.close()
        assert fake_db.priorities[1] == pytest.approx(3.25)
        assert fake_db.priorities[2] == pytest.approx(4.5)
        assert fake_db.priorities[3] is None

    def test_batch_size_triggers_flush_before_close(self, fake_db, head_sm):
        w = make_writer(head_sm, batch_size=2, flush_interval_s=30)
        try:
            w.enqueue(torch.tensor([1, 2]), torch.tensor([1.5, 2.5]))
            _wait_for(lambda: len(fake_db.update_calls) >= 1)
            assert sorted(fake_db.update_calls[0]) == [(1, 1.5), (2, 2.5)]
        finally:
            w.close()

    def test_interval_triggers_flush_below_batch_size(self, fake_db, head_sm):
        w = make_writer(head_sm, batch_size=1000, flush_interval_s=0.05)
        try:
            w.enqueue(torch.tensor([3]), torch.tensor([9.75]))
            _wait_for(lambda: fake_db.priorities[3] is not None)
            assert fake_db.priorities[3] == pytest.approx(9.75)
        finally:
            w.close()

    def test_scalar_tensors(self, fake_db, head_sm):
        w = make_writer(head_sm, batch_size=1000, flush_interval_s=30)
        w.enqueue(torch.tensor(1), torch.tensor(2.5))
        w.close()
        assert fake_db.priorities[1] == pytest.approx(2.5)

    def test_later_score_overwrites_earlier_over_time(self, fake_db, head_sm):
        w = make_writer(head_sm, batch_size=1, flush_interval_s=30)
        try:
            for step, score in enumerate([1.1, 2.2, 3.3], start=1):
                w.enqueue(torch.tensor([1]), torch.tensor([score]))
                _wait_for(lambda: len(fake_db.update_calls) >= step)
                assert fake_db.priorities[1] == pytest.approx(score)
        finally:
            w.close()

    def test_duplicate_ids_in_one_flush_apply_in_order(self, fake_db, head_sm):
        w = make_writer(head_sm, batch_size=1000, flush_interval_s=30)
        w.enqueue(torch.tensor([1, 1]), torch.tensor([1.1, 5.5]))
        w.close()
        assert fake_db.priorities[1] == pytest.approx(5.5)

    def test_flush_failure_is_logged_and_writer_keeps_working(self, fake_db, head_sm, caplog):
        fake_db.fail_next_update = True
        w = make_writer(head_sm, batch_size=1, flush_interval_s=30)
        try:
            w.enqueue(torch.tensor([1]), torch.tensor([1.5]))
            _wait_for(lambda: "ScoreWriter flush failed" in caplog.text)
            assert fake_db.priorities[1] is None

            w.enqueue(torch.tensor([2]), torch.tensor([2.5]))
            _wait_for(lambda: fake_db.priorities[2] is not None)
            assert fake_db.priorities[2] == pytest.approx(2.5)
        finally:
            w.close()

    def test_close_without_updates_does_not_touch_db(self, fake_db, head_sm):
        make_writer(head_sm).close()
        assert fake_db.update_calls == []

    def test_close_stops_thread(self, fake_db, head_sm):
        w = make_writer(head_sm)
        w.close()
        assert not w._thread.is_alive()


# --------------------------------------------------------------------------
# CandidatePool + ScoreWriter over time
# --------------------------------------------------------------------------

class TestPoolAndWriterOverTime:
    def test_scores_written_then_visible_after_refresh(self, fake_db, head_sm):
        pool = make_pool()
        writer = make_writer(head_sm, batch_size=1000, flush_interval_s=30)
        pool.refresh()
        # Unscored rows start at the init score.
        assert all(r["train_priority"] is None for r in pool._rows)
        np.testing.assert_allclose(pool._scores, datasets.GT_SCORE_INIT)

        # Iteration 10: patches scored with rarities 0.9 / 0.1; patch 3 untouched.
        writer.enqueue(torch.tensor([1, 2]), torch.tensor([10.9, 10.1]))
        writer.close()

        # Stale in-memory view until the next refresh.
        assert all(r["train_priority"] is None for r in pool._rows)

        pool.refresh()
        by_id = {r["patch_id"]: (r, s) for r, s in zip(pool._rows, pool._scores)}
        assert by_id[1][0]["train_priority"] == pytest.approx(10.9, abs=1e-4)
        assert by_id[1][1] == pytest.approx(0.9, abs=1e-4)
        assert by_id[2][1] == pytest.approx(0.1, abs=1e-4)
        assert by_id[3][1] == datasets.GT_SCORE_INIT  # never scored stays on top
        assert pool._rows[0]["patch_id"] == 3

    def test_stale_rows_decay_relative_to_newer_iterations(self, fake_db, head_sm):
        fake_db.priorities = {1: 1.5, 2: 1.5}
        pool = make_pool()
        pool.refresh()
        np.testing.assert_allclose(pool._scores, [0.5, 0.5])

        # Patch 2 is re-scored at a much later iteration with the same rarity.
        writer = make_writer(head_sm, batch_size=1000, flush_interval_s=30)
        writer.enqueue(torch.tensor([2]), torch.tensor([101.5]))
        writer.close()
        pool.refresh()

        scores = {r["patch_id"]: s for r, s in zip(pool._rows, pool._scores)}
        assert scores[2] == pytest.approx(0.5, abs=1e-4)
        assert scores[1] == pytest.approx(0.5 * math.exp(-datasets.GT_SCORE_DECAY * 100), abs=1e-4)
        assert scores[1] < scores[2]

    def test_draw_decay_write_refresh_cycle(self, fake_db, head_sm):
        pool = make_pool()
        writer = make_writer(head_sm, batch_size=1000, flush_interval_s=30)
        pool.refresh()

        np.random.seed(0)
        picks = pool.draw_batch(2)
        for _, idx in picks:
            pool.decay_in_memory_score(idx)
        drawn_ids = [row["patch_id"] for row, _ in picks]

        # Memory decays immediately; DB only changes once the writer flushes.
        for _, idx in picks:
            assert pool._scores[idx] == pytest.approx(datasets.GT_SCORE_INIT * GT_SCORE_IN_MEMORY_DECAY)
        assert all(v is None for v in fake_db.priorities.values())

        writer.enqueue(torch.tensor(drawn_ids), torch.tensor([5.3] * len(drawn_ids)))
        writer.close()
        assert all(fake_db.priorities[p] == pytest.approx(5.3, abs=1e-4) for p in drawn_ids)

        pool.refresh()
        undrawn = ({1, 2, 3} - set(drawn_ids)).pop()
        scores = {r["patch_id"]: s for r, s in zip(pool._rows, pool._scores)}
        assert scores[undrawn] == datasets.GT_SCORE_INIT
        for p in drawn_ids:
            assert scores[p] == pytest.approx(0.3, abs=1e-4)


def _wait_for(cond, timeout=5.0):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met in time")
