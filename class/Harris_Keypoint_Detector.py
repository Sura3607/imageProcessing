import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image


def to_gray(image):
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0


def _gray_to_uint8(img_gray):
    img = img_gray
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        if img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _draw_points_on_gray(img_gray, coords, color=(0, 255, 255)):
    base = cv2.cvtColor(_gray_to_uint8(img_gray), cv2.COLOR_GRAY2BGR)
    for y, x in coords:
        cv2.drawMarker(
            base,
            (int(x), int(y)),
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=6,
            thickness=1,
        )
    return cv2.cvtColor(base, cv2.COLOR_BGR2RGB)


def harris_keypoint_image(
    image, k=0.04, top_n=300, thresh_ratio=0.01, color=(0, 255, 255)
):
    img_gray = to_gray(image)
    C = cv2.cornerHarris(img_gray, blockSize=2, ksize=3, k=k)

    C_thresh = C.copy()
    C_thresh[C_thresh < thresh_ratio * C_thresh.max()] = 0

    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(C_thresh, kernel)
    img_ext = (C_thresh == dilated) & (C_thresh > 0)
    coords = np.column_stack(np.nonzero(img_ext))

    if coords.size > 0:
        responses = C[coords[:, 0], coords[:, 1]]
        sort_idx = np.argsort(responses)[::-1]
        coords = coords[sort_idx[: min(top_n, len(sort_idx))]]

    return _draw_points_on_gray(img_gray, coords, color)


def show_and_pause():
    plt.show(block=False)
    plt.pause(0.001)
    input("Press Enter to continue...")


def _demo():
    image_files = ["ukbench00732.jpg", "ukbench00733.jpg"]
    for image_name in image_files:
        img = np.array(Image.open(image_name).convert("RGB"))
        img_gray = to_gray(img)

        plt.figure(1)
        plt.clf()
        plt.imshow(img)
        plt.title("Original Image")

        C = cv2.cornerHarris(img_gray, blockSize=2, ksize=3, k=0.04)
        plt.figure(2)
        plt.clf()
        plt.imshow(C, vmin=-2e-3, vmax=4e-3, cmap="gray")
        plt.title("Harris Cornerness")

        C_thresh = C.copy()
        C_thresh[C_thresh < 0.01 * C_thresh.max()] = 0

        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(C_thresh, kernel)
        img_ext = (C_thresh == dilated) & (C_thresh > 0)
        coords = np.column_stack(np.nonzero(img_ext))

        plt.figure(3)
        plt.clf()
        plt.imshow(C_thresh, vmin=-2e-3, vmax=4e-3, cmap="gray")
        plt.title("Local Maxima")

        plt.figure(4)
        plt.clf()
        plt.imshow(cv2.dilate(img_ext.astype(np.uint8), kernel), cmap="gray")
        plt.title("Local Maxima (Dilated)")

        plt.figure(5)
        plt.clf()
        plt.imshow(img)
        plt.title("Harris Keypoints")
        if coords.size > 0:
            responses = C[coords[:, 0], coords[:, 1]]
            sort_idx = np.argsort(responses)[::-1]
            top = min(300, len(sort_idx))
            top_coords = coords[sort_idx[:top]]
            plt.plot(top_coords[:, 1], top_coords[:, 0], "y+", markersize=3)

        show_and_pause()


if __name__ == "__main__":
    _demo()
