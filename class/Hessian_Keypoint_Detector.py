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


def gaussian_kernel(size, sigma):
    ax = np.arange(-(size // 2), size // 2 + 1)
    g1d = np.exp(-(ax**2) / (2.0 * sigma**2))
    g1d /= g1d.sum()
    return np.outer(g1d, g1d).astype(np.float32)


def hessian_keypoint_image(
    image, sigma=2, top_n=600, thresh_ratio=0.05, color=(0, 255, 255)
):
    img_gray = to_gray(image)

    size = 10 * sigma + 1
    g = gaussian_kernel(size, sigma)
    s = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)

    filter_xg = cv2.filter2D(
        cv2.filter2D(g, -1, s.T, borderType=cv2.BORDER_REPLICATE),
        -1,
        s.T,
        borderType=cv2.BORDER_REPLICATE,
    )
    filter_yg = cv2.filter2D(
        cv2.filter2D(g, -1, s, borderType=cv2.BORDER_REPLICATE),
        -1,
        s,
        borderType=cv2.BORDER_REPLICATE,
    )
    filter_xyg = cv2.filter2D(
        cv2.filter2D(g, -1, s.T, borderType=cv2.BORDER_REPLICATE),
        -1,
        s,
        borderType=cv2.BORDER_REPLICATE,
    )

    img_dxx = cv2.filter2D(img_gray, -1, filter_xg, borderType=cv2.BORDER_REPLICATE)
    img_dyy = cv2.filter2D(img_gray, -1, filter_yg, borderType=cv2.BORDER_REPLICATE)
    img_dxy = cv2.filter2D(img_gray, -1, filter_xyg, borderType=cv2.BORDER_REPLICATE)
    img_filt = img_dxx * img_dyy - img_dxy**2

    img_thresh = img_filt.copy()
    img_thresh[img_thresh < thresh_ratio * img_thresh.max()] = 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(img_thresh, kernel)
    img_ext = (img_thresh == dilated) & (img_thresh > 0)
    coords = np.column_stack(np.nonzero(img_ext))

    if coords.size > 0:
        responses = img_filt[coords[:, 0], coords[:, 1]]
        sort_idx = np.argsort(responses)[::-1]
        coords = coords[sort_idx[: min(top_n, len(sort_idx))]]

    return _draw_points_on_gray(img_gray, coords, color)


def show_and_pause():
    plt.show(block=False)
    plt.pause(0.001)
    input("Press Enter to continue...")


def _demo():
    image_files = ["35-a.jpg", "35-b.jpg"]
    for image_name in image_files:
        img = np.array(Image.open(image_name).convert("RGB"))
        img_gray = to_gray(img)

        plt.figure(1)
        plt.clf()
        plt.imshow(img)
        plt.title("Original Image")

        sigma = 2
        size = 10 * sigma + 1
        g = gaussian_kernel(size, sigma)
        s = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)

        filter_xg = cv2.filter2D(
            cv2.filter2D(g, -1, s.T, borderType=cv2.BORDER_REPLICATE),
            -1,
            s.T,
            borderType=cv2.BORDER_REPLICATE,
        )
        filter_yg = cv2.filter2D(
            cv2.filter2D(g, -1, s, borderType=cv2.BORDER_REPLICATE),
            -1,
            s,
            borderType=cv2.BORDER_REPLICATE,
        )
        filter_xyg = cv2.filter2D(
            cv2.filter2D(g, -1, s.T, borderType=cv2.BORDER_REPLICATE),
            -1,
            s,
            borderType=cv2.BORDER_REPLICATE,
        )

        plt.figure(2)
        plt.subplot(1, 3, 1)
        plt.imshow(filter_xg, vmin=-0.2, vmax=0.2, cmap="gray")
        plt.title("Filter x")
        plt.subplot(1, 3, 2)
        plt.imshow(filter_yg, vmin=-0.2, vmax=0.2, cmap="gray")
        plt.title("Filter y")
        plt.subplot(1, 3, 3)
        plt.imshow(filter_xyg, vmin=-0.2, vmax=0.2, cmap="gray")
        plt.title("Filter x-y")

        img_dxx = cv2.filter2D(
            img_gray, -1, filter_xg, borderType=cv2.BORDER_REPLICATE
        )
        img_dyy = cv2.filter2D(
            img_gray, -1, filter_yg, borderType=cv2.BORDER_REPLICATE
        )
        img_dxy = cv2.filter2D(
            img_gray, -1, filter_xyg, borderType=cv2.BORDER_REPLICATE
        )
        img_filt = img_dxx * img_dyy - img_dxy**2

        plt.figure(3)
        plt.clf()
        plt.imshow(img_filt, vmin=-1, vmax=1, cmap="gray")
        plt.title("Determinant of Hessian Response")

        img_thresh = img_filt.copy()
        img_thresh[img_thresh < 0.05 * img_thresh.max()] = 0
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(img_thresh, kernel)
        img_ext = (img_thresh == dilated) & (img_thresh > 0)
        coords = np.column_stack(np.nonzero(img_ext))

        plt.figure(4)
        plt.clf()
        plt.imshow(img_thresh, vmin=-1, vmax=1, cmap="gray")
        plt.title("Thresholded Response")

        plt.figure(5)
        plt.clf()
        plt.imshow(cv2.dilate(img_ext.astype(np.uint8), kernel), cmap="gray")
        plt.title("Local Maxima (Dilated)")

        plt.figure(6)
        plt.clf()
        plt.imshow(img)
        plt.title("Hessian Keypoints")
        if coords.size > 0:
            responses = img_filt[coords[:, 0], coords[:, 1]]
            sort_idx = np.argsort(responses)[::-1]
            top = min(600, len(sort_idx))
            top_coords = coords[sort_idx[:top]]
            plt.plot(top_coords[:, 1], top_coords[:, 0], "y+", markersize=3)

        show_and_pause()


if __name__ == "__main__":
    _demo()
