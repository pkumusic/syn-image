# -*- coding: utf-8 -*-
import time

from tensorboard_logger import configure, log_value

def test_smoke_default(tmpdir):
    configure(str(tmpdir), flush_secs=0.1)
    for step in range(10):
        log_value('v1', step * 1.5, step)
        log_value('v2', step ** 1.5 - 2)
    time.sleep(0.5)
    # tf_log, = tmpdir.listdir()
    # assert tf_log.basename.startswith('events.out.tfevents.')

test_smoke_default('./runs')

