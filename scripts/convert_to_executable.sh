#!/bin/bash

# command to create camera_adjust executable 
python -m nuitka camera_adjust.py       \
    --follow-import-to=gauge_detection  \
    --follow-import-to=metadata

# command to create get_readings executable
python -m nuitka get_readings.py            \
    --follow-import-to=plots                \
    --follow-import-to=gauge_detection      \
    --follow-import-to=key_point_detection  \
    --follow-import-to=geometry             \
    --follow-import-to=angle_reading_fit    \
    --follow-import-to=segmentation         \
    --follow-import-to=evaluation           \
    --follow-import-to=metadata             \
    --follow-import-to=cloud