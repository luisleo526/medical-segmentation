#!/bin/bash

root="task8-dynunet-ver2"
phase=$1
patient_id=$2

mkdir -p "${root}/${phase}-${patient_id}"

ffmpeg -i "${root}/${phase}-pre-${patient_id}/3d_view/view_%2d.png" \
-i "${root}/${phase}-post-${patient_id}/3d_view/view_%2d.png" \
-filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" "${root}/${phase}-${patient_id}/merged_%2d.png"

ffmpeg -framerate 12 \
-i "${root}/${phase}-${patient_id}/merged_%2d.png" -vf "palettegen" -y \
"${root}/${phase}-${patient_id}/palette.png"

ffmpeg -framerate 12 -loop 0 \
-i "${root}/${phase}-${patient_id}/merged_%2d.png" \
-i "${root}/${phase}-${patient_id}/palette.png" \
-filter_complex "paletteuse" -y "${root}/${phase}-${patient_id}/${phase}-${patient_id}.gif"
