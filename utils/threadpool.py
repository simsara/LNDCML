from concurrent.futures import Future
from concurrent.futures.thread import ThreadPoolExecutor
from os import cpu_count

from utils.log import get_logger

log = get_logger(__name__)
pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=cpu_count())


def submit(func, *args, **kwargs) -> Future:
    return pool.submit(log_call, func=func, *args, **kwargs)


def log_call(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except Exception as e:
        log.info('Execute fail [%s]' % func.__name__)
    else:
        log.info('Execute ok [%s]' % func.__name__)


def join():
    pool.shutdown(True)
