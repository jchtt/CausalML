#!/usr/bin/env python

import os
import datetime
import time
import pathlib

# tasks = [("semi_synth_vare", 1)]
tasks = [
        # ("clusters_vard_norm", 32),
        # ("clusters_varp", 32),
        # ("cycles_varp", 32),
        # ("cycles_varn", 32),
        # ("cycles_vare", 32),
        # ("clusters_varn", 32),
        # ("rand_missing", 32),
        # ("rand_vard_norm", 32),
        # ("rand_vare", 32),

        # ("clusters_missing", 1),
        # ("cycles_missing", 32),
        # ("rand_varn", 1),
        # ("clusters_missing", 32),
        # ("cycles_missing", 32),
        # ("rand_varn", 32),
        # ("rand_varp", 32),
        # ("worst_vare", 32),
        ("rand_cv_varn", 32),
        ]

last_start = 7
last_minute = 0
cur_time = datetime.datetime.now()
# cut_off = cur_time
cut_off = cur_time + datetime.timedelta(days = 2)
cut_off = cut_off.replace(hour = last_start, minute = last_minute, second = 0)
# cut_off = cur_time.replace(minute = 57)

start_index = 3
task_index = 0
task = tasks[task_index][0]
end_index = tasks[task_index][1]

# while cur_time < cut_off and start_index <= end_index and task_index < len(tasks):
while start_index <= end_index and task_index < len(tasks):
    pathlib.Path('./logs_{}'.format(task)).mkdir(parents=True, exist_ok=True) 
    os.system("julia5 CausalMLTest.jl {} {} &> ./logs_{}/log_{}.txt".format(start_index, task, task, start_index))
    # os.system("touch test/{}".format(start_index))
    time.sleep(1)

    start_index += 1
    if start_index > end_index:
        if task_index < len(tasks) - 1:
            task_index += 1
            task = tasks[task_index][0]
            end_index = tasks[task_index][1]
            start_index = 1
        else:
            break
    cur_time = datetime.datetime.now()
