#!/usr/bin/env bash

byobu new-session -d -s "slice" "/usr/bin/python /home/gym/projs/slices/run.py > logs/run.log 2>&1"
