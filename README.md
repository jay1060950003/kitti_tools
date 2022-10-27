# Kitti Tools

Here are tools for kitti dataset raw data.
[jump to raw data](https://www.cvlibs.net/datasets/kitti/raw_data.php)

**the kitti raw data construction**: kitti -> 2011_09_26 -> 2011_09_26_drive_XXXX_sync

# Install

```
pip install -r requirements.txt
```

# Use

Here is a demo to use.
You can code your own code for use.

# what I do

Because of my research using ktiii`s gray image and velo data, I try to **convert velo data to cam0 and save to png in depth_00**.
The point png is not suitable for my research, I try to **completion the point png to depth png in depth_01**.
I try to **read XML and covert to label**, which is consist of **class name and xywh**.
You can show the label in gray image with the function draw2D.
