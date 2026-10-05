import os
import glob
import numpy as np
import cv2

CLASS_NAMES = ["zero_people", "one_person", "two_plus_people"]
CLASS_LABELS = {name: i for i, name in enumerate(CLASS_NAMES)}
CLASS_DISPLAY = {0: "Khong co nguoi", 1: "1 nguoi", 2: "2+ nguoi"}


def list_images(root, class_name):
    folder = os.path.join(root, class_name)
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)


def load_image(path, img_size=64, grayscale=True):
    """Doc 1 anh -> vector dac trung (flatten)."""
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise ValueError(f"Khong doc duoc anh: {path}")
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    if grayscale:
        vec = img.reshape(-1).astype(np.float32) / 255.0
    else:
        vec = img.reshape(-1).astype(np.float32) / 255.0
    return vec


def build_dataset(root, img_size=64, grayscale=True, max_per_class=None, verbose=True):
    X, y, paths = [], [], []
    for class_name in CLASS_NAMES:
        files = list_images(root, class_name)
        if max_per_class is not None:
            files = files[:max_per_class]
        label = CLASS_LABELS[class_name]
        if verbose:
            print(f"  [{class_name}] {len(files)} anh")
        for i, f in enumerate(files):
            try:
                X.append(load_image(f, img_size=img_size, grayscale=grayscale))
                y.append(label)
                paths.append(f)
            except Exception as exc:
                if verbose:
                    print(f"    bo qua {f}: {exc}")
            if verbose and (i + 1) % 2000 == 0:
                print(f"    ... da xu ly {i + 1}/{len(files)}")
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    paths = np.asarray(paths)
    return X, y, paths


def extract_single(path, img_size=64, grayscale=True):
    return load_image(path, img_size=img_size, grayscale=grayscale)
