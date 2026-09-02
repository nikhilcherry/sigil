"""A bounded read-ahead over a stream of work.

Both hot loops in this project have the same shape: a slow network fetch
feeding a slow, single-threaded face encoder. The obvious implementations both
get it wrong. Fetching a batch, then encoding the batch, leaves every network
worker idle for the whole inference phase. ``ThreadPoolExecutor.map`` overlaps
them but submits *every* task up front, so on a few thousand portraits it
queues a few thousand downloads and holds their bytes in memory ahead of a
consumer that is thousands of images behind.

A fixed window of in-flight futures fixes both: work stays queued only as far
ahead as the consumer can use, and results come back in submission order so a
run is reproducible.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def prefetch(
    pool: ThreadPoolExecutor,
    items: Iterable[T],
    work: Callable[[T], R],
    window: int,
) -> Iterator[tuple[T, R]]:
    """Run ``work`` over ``items`` with at most ``window`` results outstanding.

    Yields ``(item, result)`` in the order items arrived, never in completion
    order. ``items`` is consumed lazily, so an infinite or expensive generator
    is only advanced as far as the window requires.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    pending: deque[tuple[T, Future[R]]] = deque()
    try:
        for item in items:
            pending.append((item, pool.submit(work, item)))
            if len(pending) >= window:
                done, fut = pending.popleft()
                yield done, fut.result()
        while pending:
            done, fut = pending.popleft()
            yield done, fut.result()
    finally:
        # A consumer is free to stop early - islice(max_images) does exactly
        # that. Without this, the read-ahead it never asked for would still be
        # run to completion by the pool's shutdown, which waits on queued work.
        for _item, fut in pending:
            fut.cancel()
