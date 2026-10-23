import os

import cv2

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
cv2.setNumThreads(1)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running test, excluded by default (run with -m slow)")
    config.addinivalue_line("markers", "live: hits the deployed endpoint, needs NEXTVIEW_URL (run with -m live)")
