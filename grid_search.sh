#!/bin/bash

python grid_search.py --environment "single_stimulus" --n_oscillators 4 &
python grid_search.py --environment "single_stimulus" --n_oscillators 5 &

python grid_search.py --environment "double_stimulus" --n_oscillators 4 &
python grid_search.py --environment "double_stimulus" --n_oscillators 5 &

