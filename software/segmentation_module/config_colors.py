import cv2
import numpy as np
import _pickle as pickle
import camera
from Color import Color
import logging

COLOR_CONFIG_FILE = 'colors/colors.pkl'
logging.basicConfig()
logging.root.setLevel(logging.NOTSET)


def nothing(x):
    # The callback functionality is not required by the code. Doing nothing.
    pass


cv2.namedWindow('image')
cv2.namedWindow('rgb')
cv2.namedWindow('mask')
cv2.moveWindow('mask', 400, 0)

try:
    logging.info(f"Opening config file: {COLOR_CONFIG_FILE!r}")
    with open(COLOR_CONFIG_FILE, 'rb') as fr:
        colors_lookup = pickle.load(fr)
except OSError as e:
    logging.info('Failed to open color config file. Recreating the file.')
    colors_lookup = np.zeros(0x1000000, dtype=np.uint8)
    with open(COLOR_CONFIG_FILE, 'wb') as fw:
        pickle.dump(colors_lookup, fw, -1)

# Camera instance for normal web cameras
# cap = camera.OpenCVCamera(cam_id = 0)

# Camera instance for realsense cameras
cap = camera.RealsenseCamera(exposure=100)

cv2.createTrackbar('brush_size', 'image', 3, 10, nothing)
cv2.createTrackbar('noise', 'image', 1, 5, nothing)


key_dict = {
    ord("g"): Color.GREEN,
    ord("m"): Color.MAGENTA,
    ord("b"): Color.BLUE,
    ord("f"): Color.ORANGE,
    ord("w"): Color.WHITE,
    ord("d"): Color.BLACK,
    ord("o"): Color.OTHER,
}


def change_color(noise, brush_size, mouse_x, mouse_y):
    ob = rgb[
         max(0, mouse_y - brush_size):min(cap.rgb_height, mouse_y + brush_size + 1),
         max(0, mouse_x - brush_size):min(cap.rgb_width, mouse_x + brush_size + 1), :].reshape((-1, 3)).astype('int32')
    noises = range(-noise, noise + 1)
    for r in noises:
        for g in noises:
            for b in noises:
                colors_lookup[
                    ((ob[:, 0] + r) + (ob[:, 1] + g) * 0x100 + (ob[:, 2] + b) * 0x10000).clip(0, 0xffffff)] = p


# mouse callback function
def choose_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        brush_size = cv2.getTrackbarPos('brush_size', 'image')
        noise = cv2.getTrackbarPos('noise', 'image')
        change_color(noise, brush_size, x, y)


cv2.namedWindow('rgb')
cv2.setMouseCallback('rgb', choose_color)
cv2.setMouseCallback('mask', choose_color)

logging.info("Quit: 'q', Save 's', Erase selected color 'e'")
logging.info("Balls 'g', Magenta basket='m', Blue basket='b', Field='f', White='w', Black='d', Other='o'")

cap.open()
col = Color.UNDEFINED

while True:
    rgb = cap.get_color_frame()

    cv2.imshow('rgb', rgb)

    fragmented = colors_lookup[
        rgb[:, :, 0].astype(int) + rgb[:, :, 1].astype(int) * 0x100 + rgb[:, :, 2].astype(int) * 0x10000]
    frame = np.zeros((cap.rgb_height, cap.rgb_width, 3), dtype=np.uint8)

    for color in Color:
        frame[fragmented == int(color)] = color.color

    cv2.imshow('mask', frame)

    k = cv2.waitKey(1) & 0xff


    if k == ord('q'):
        logging.info("Closing colors configurator")
        break
    elif k in key_dict:
        col = key_dict[k]
        logging.info(f"Selected color: {col}")
        p = int(col)
    elif k == ord('s'):
        with open(COLOR_CONFIG_FILE, 'wb') as fw:
            pickle.dump(colors_lookup, fw, -1)
        logging.info(f'Saved configuration to {COLOR_CONFIG_FILE}')
    elif k == ord('e'):
        logging.info(f'Erased color {col.name!r}')
        colors_lookup[colors_lookup == p] = 0

# When everything done, release the capture

cap.close()

cv2.destroyAllWindows()