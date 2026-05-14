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


def laplacian_keypoint_image(
    image, sigma=4, top_n=900, thresh_ratio=0.35, color=(0, 255, 255)
):
    img_gray = to_gray(image)

    ksize = int(3 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(
        img_gray,
        (ksize, ksize),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    img_filt = cv2.Laplacian(
        blurred, cv2.CV_32F, ksize=3, borderType=cv2.BORDER_REPLICATE
    )

    img_filt_abs = np.abs(img_filt)
    thresh = thresh_ratio * np.max(img_filt_abs)
    img_thresh = img_filt.copy()
    img_thresh[img_filt_abs < thresh] = 0

    kernel = np.ones((3, 3), dtype=np.uint8)
    max_pos = cv2.dilate(img_thresh, kernel)
    max_neg = cv2.dilate(-img_thresh, kernel)
    img_ext = ((img_thresh == max_pos) & (img_thresh > 0)) | (
        (-img_thresh == max_neg) & (img_thresh < 0)
    )

    coords = np.column_stack(np.nonzero(img_ext))
    if coords.size > 0:
        border = int(sigma)
        valid = (
            (coords[:, 0] > border)
            & (coords[:, 0] < img_gray.shape[0] - border)
            & (coords[:, 1] > border)
            & (coords[:, 1] < img_gray.shape[1] - border)
        )
        coords = coords[valid]
        if coords.size > 0:
            responses = np.abs(img_filt[coords[:, 0], coords[:, 1]])
            sort_idx = np.argsort(responses)[::-1]
            coords = coords[sort_idx[: min(top_n, len(sort_idx))]]

    return _draw_points_on_gray(img_gray, coords, color)


def show_and_pause():
    plt.show(block=False)
    plt.pause(0.001)
    input("Press Enter to continue...")


def _demo():
    image_files = ["img2.ppm", "img1.ppm"]
    for idx, image_name in enumerate(image_files, start=1):
        img = np.array(Image.open(image_name).convert("RGB"), dtype=np.float32)

        if idx == 2:
            gradient = 0.7
            if img.ndim == 2:
                img = img * gradient - 0.1
            else:
                for c in range(3):
                    img[:, :, c] = img[:, :, c] * gradient - 0.1

        img_gray = to_gray(img)

        plt.figure(1)
        plt.clf()
        plt.imshow(img)
        plt.title("Original Image")

        sigma = 4
        ksize = int(3 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(
            img_gray,
            (ksize, ksize),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REPLICATE,
        )
        img_filt = cv2.Laplacian(
            blurred, cv2.CV_32F, ksize=3, borderType=cv2.BORDER_REPLICATE
        )

        plt.figure(2)
        plt.clf()
        plt.imshow(img_filt, vmin=-0.02, vmax=0.02, cmap="gray")
        plt.title("LoG Response")

        img_filt_abs = np.abs(img_filt)
        thresh = 0.35 * np.max(img_filt_abs)
        img_thresh = img_filt.copy()
        img_thresh[img_filt_abs < thresh] = 0

        kernel = np.ones((3, 3), dtype=np.uint8)
        max_pos = cv2.dilate(img_thresh, kernel)
        max_neg = cv2.dilate(-img_thresh, kernel)
        img_ext = ((img_thresh == max_pos) & (img_thresh > 0)) | (
            (-img_thresh == max_neg) & (img_thresh < 0)
        )

        plt.figure(3)
        plt.clf()
        plt.imshow(img_thresh, vmin=-0.02, vmax=0.02, cmap="gray")
        plt.title("Thresholded LoG Response")

        plt.figure(4)
        plt.clf()
        plt.imshow(cv2.dilate(img_ext.astype(np.uint8), kernel), cmap="gray")
        plt.title("Local Extrema (Dilated)")

        plt.figure(5)
        plt.clf()
        plt.imshow(img)
        plt.title("Keypoints")

        coords = np.column_stack(np.nonzero(img_ext))
        if coords.size > 0:
            border = int(sigma)
            valid = (
                (coords[:, 0] > border)
                & (coords[:, 0] < img_gray.shape[0] - border)
                & (coords[:, 1] > border)
                & (coords[:, 1] < img_gray.shape[1] - border)
            )
            coords = coords[valid]
            responses = np.abs(img_filt[coords[:, 0], coords[:, 1]])
            sort_idx = np.argsort(responses)[::-1]
            top = min(900, len(sort_idx))
            top_coords = coords[sort_idx[:top]]
            plt.plot(top_coords[:, 1], top_coords[:, 0], "y+", markersize=3)

        show_and_pause()


if __name__ == "__main__":
    _demo()
