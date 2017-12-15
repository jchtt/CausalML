#!/usr/bin/env python

import os
import datetime
import time

task = "semi_synth_vare"
last_start = 14
last_minute = 5
cur_time = datetime.datetime.now()
cut_off = cur_time
# cut_off = cur_time + datetime.timedelta(days = 1)
cut_off = cut_off.replace(hour = last_start, minute = last_minute, second = 0)
# cut_off = cur_time.replace(minute = 57)

start_index = 1
end_index = 1

while cur_time < cut_off and start_index <= end_index:
    os.system("julia5 CausalMLTest.jl {} {} &> ./logs/log_{}.txt".format(start_index, task, start_index))
    # os.system("touch test/{}".format(start_index))
    time.sleep(1)

    start_index += 1
    cur_time = datetime.datetime.now()
