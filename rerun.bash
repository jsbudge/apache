#!/bin/bash
conda activate apache
for i in 1 150 165 1200 1500
do
  python wave_simulator.py $i
done