#!/bin/bash
current_dir=$(dirname "$(readlink -f "$0")")

#python3 -m unittest discover
python3 -m Nidelva3D.tests.test_planner
python3 -mm Nidelva3D.tests.test_hexgonal2d

echo $current_dir
# echo $v1
