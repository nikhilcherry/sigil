"""The bounded read-ahead both hot loops are built on."""

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from sigil.concurrency import prefetch


def test_results_come_back_in_submission_order():
    """Work finishes out of order; the stream must not.

    Ranking downstream breaks ties on arrival order, so a reordering prefetch
    would make runs non-reproducible.
    """
    def work(i):
        time.sleep(random.uniform(0, 0.01))
        return i * 10

    with ThreadPoolExecutor(max_workers=8) as pool:
        got = list(prefetch(pool, range(40), work, 8))

    assert got == [(i, i * 10) for i in range(40)]


def test_it_reads_ahead_far_enough_to_keep_workers_busy():
    """The whole point: work is in flight before the consumer asks for it."""
    lock = threading.Lock()
    started = 0
    enough = threading.Event()

    def work(i):
        nonlocal started
        with lock:
            started += 1
            if started >= 4:
                enough.set()
        time.sleep(0.05)
        return i

    with ThreadPoolExecutor(max_workers=4) as pool:
        stream = prefetch(pool, range(20), work, 8)
        next(stream)  # pull one result
        # By the time one result is out, every worker must already be busy on
        # the ones behind it - that is what "overlapped" means here. Waited for
        # rather than sampled: on a loaded machine the threads are certain to
        # start, but not certain to have started by any particular instant.
        assert enough.wait(timeout=10), f"only {started} of 4 workers ever started"
        list(stream)


def test_it_does_not_queue_the_whole_stream():
    """ThreadPoolExecutor.map submits everything up front; this must not.

    On a few thousand portraits that is the difference between a bounded
    working set and holding every downloaded image in memory at once.
    """
    pulled = []

    def items():
        for i in range(1000):
            pulled.append(i)
            yield i

    with ThreadPoolExecutor(max_workers=4) as pool:
        stream = prefetch(pool, items(), lambda i: i, window=10)
        next(stream)
        # One result consumed, so at most the window may have been read ahead.
        assert len(pulled) <= 11
        list(stream)


def test_a_window_below_one_is_refused():
    with ThreadPoolExecutor(max_workers=2) as pool:
        with pytest.raises(ValueError, match="window"):
            list(prefetch(pool, [1, 2], lambda i: i, window=0))


def test_an_exception_in_the_work_reaches_the_caller():
    """Failures must not be silently swallowed into a short stream."""
    def work(i):
        if i == 3:
            raise RuntimeError("boom")
        return i

    with ThreadPoolExecutor(max_workers=4) as pool:
        with pytest.raises(RuntimeError, match="boom"):
            list(prefetch(pool, range(10), work, 4))


def test_abandoning_the_stream_cancels_the_read_ahead():
    """A consumer may stop early - islice(max_images) does exactly that.

    The work it never asked for must not still be run to completion by the
    pool's shutdown, which otherwise waits on everything already queued.
    """
    done = []

    def work(i):
        time.sleep(0.05)
        done.append(i)
        return i

    with ThreadPoolExecutor(max_workers=1) as pool:
        stream = prefetch(pool, range(30), work, window=10)
        next(stream)
        stream.close()

    # One worker, so leaving after the first result should finish at most the
    # one already in progress. Ten means the entire queued window still ran.
    assert len(done) <= 3, f"the abandoned read-ahead still ran ({len(done)} items)"
