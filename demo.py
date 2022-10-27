import cv2
from pykitti.raw import raw, draw2Dbox

def main():
    # Change this to the directory where you store KITTI data
    basedir = r'D:\CodeRep\Python\datasets\kitti'

    # Specify the dataset to load
    date = '2011_09_26'
    drive = '0005'

    # Load the data. Optionally, specify the frame range to load.
    dataset = raw(basedir, date, drive, frames=range(0, 25, 5))

    # Grab some data
    gray = dataset.get_cam0(0)
    calib = dataset.calib
    depth = dataset.get_depth(0)
    filled_depth = dataset.get_filled_depth(0)
    label = dataset.get_label(0)
    
    # show
    draw2Dbox(gray, label)
    draw2Dbox(depth, label)
    draw2Dbox(filled_depth, label)
    cv2.imshow('gray', gray)
    cv2.imshow('depth', depth)
    cv2.imshow('filled_depth', filled_depth)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()