#!/bin/bash

# ls -d -1 ../*_seg/** | parallel -j32 bash run.sh {} \;

input_dir=$1

pvpython paraview_make_3d_snapshots.py --input "${input_dir}/3D_mask.vtr"

cd $input_dir/3d_view
ffmpeg -framerate 12 -i view_%03d.png -vf "palettegen" -y palette.png
ffmpeg -framerate 12 -loop 0 -i view_%03d.png -i palette.png -filter_complex "paletteuse" -y ../output.gif

cd ..
rm -rf 3d_view
