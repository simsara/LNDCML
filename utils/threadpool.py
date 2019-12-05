from concurrent.futures import Future
from concurrent.futures.thread import ThreadPoolExecutor
from os import cpu_count

pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=cpu_count())


def submit(func, *args, **kwargs) -> Future:
    return pool.submit(func, *args, **kwargs)


def join():
    pool.shutdown(True)
