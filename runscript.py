#!/usr/bin/env python

import os
import datetime
import time

tasks = [("semi_synth_vare", 1)]
last_start = 7
last_minute = 0
cur_time = datetime.datetime.now()
# cut_off = cur_time
cut_off = cur_time + datetime.timedelta(days = 1)
cut_off = cut_off.replace(hour = last_start, minute = last_minute, second = 0)
# cut_off = cur_time.replace(minute = 57)

start_index = 1
task_index = 0
task = tasks[task_index][0]
end_index = tasks[task_index][1]

while cur_time < cut_off and start_index <= end_index and task_index < length(tasks):
    os.system("julia5 CausalMLTest.jl {} {} &> ./logs/log_{}.txt".format(start_index, task, start_index))
    # os.system("touch test/{}".format(start_index))
    time.sleep(1)

    start_index += 1
    if start_index > end_index:
        if task_index < length(tasks) - 1:
            task_index += 1
            task = tasks[task_index][0]
            end_index = tasks[task_index][1]
            start_index = 1
        else:
            break
    cur_time = datetime.datetime.now()
