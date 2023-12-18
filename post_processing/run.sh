#!/bin/bash

input_dir=$1

pvpython paraview_make_3d_snapshots.py --input "${input_dir}/3D_mask.vtr"
cd $input_dir/3d_view
ffmpeg -f image2 -i view_%02d.png output.gif