import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image


def to_gray(image):
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if gray.dtype != np.uint8:
        gray = gray.astype(np.float32)
        if gray.max() <= 1.0:
            gray = gray * 255.0
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def _draw_keypoints_on_gray(img_gray, keypoints, color=(0, 255, 255)):
    base = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    out = cv2.drawKeypoints(
        base, keypoints, None, color, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def fast_keypoint_image(image, threshold=50, nonmax=True, color=(0, 255, 255)):
    img_gray = to_gray(image)
    detector = cv2.FastFeatureDetector_create(
        threshold=threshold, nonmaxSuppression=nonmax
    )
    keypoints = detector.detect(img_gray, None)
    return _draw_keypoints_on_gray(img_gray, keypoints, color)


def show_and_pause():
    plt.show(block=False)
    plt.pause(0.001)
    input("Press Enter to continue...")


def _demo():
    image_files = ["019_Reference.jpg", "019_Palm.jpg"]
    for image_name in image_files:
        img_pil = Image.open(image_name).convert("RGB")
        target_height = 480
        target_width = int(round(img_pil.width * target_height / img_pil.height))
        img_pil = img_pil.resize((target_width, target_height), resample=Image.BILINEAR)
        img = np.array(img_pil)

        img_gray = to_gray(img)

        plt.figure(1)
        plt.clf()
        plt.imshow(img)
        plt.title("Original Image")

        fast_raw = cv2.FastFeatureDetector_create(
            threshold=50, nonmaxSuppression=False
        )
        fast_nms = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
        keypoints_raw = fast_raw.detect(img_gray, None)
        keypoints_nms = fast_nms.detect(img_gray, None)
        cs = np.array([kp.pt for kp in keypoints_raw], dtype=np.float32)
        c = np.array([kp.pt for kp in keypoints_nms], dtype=np.float32)

        plt.figure(2)
        plt.clf()
        plt.imshow(img)
        plt.title("Fast Corners")
        if cs.size > 0:
            plt.plot(cs[:, 0], cs[:, 1], "k+", markersize=3)
            plt.plot(cs[:, 0] + 0.5, cs[:, 1] + 0.5, "y+", markersize=3)

        plt.figure(3)
        plt.clf()
        plt.imshow(img)
        plt.title("Fast Corners, Non-max Suppressed")
        if c.size > 0:
            plt.plot(c[:, 0], c[:, 1], "k+", markersize=5)
            plt.plot(c[:, 0] + 0.5, c[:, 1] + 0.5, "y+", markersize=5)

        show_and_pause()


if __name__ == "__main__":
    _demo()
