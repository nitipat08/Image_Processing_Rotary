# train_floc_model_ROI_edge.py

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# -----------------------------
# Config: image file names
# -----------------------------
image_files = [
    "floctank1.jpg",
    "floctank2.jpg",
    "floctank3.jpg",
    "floctank4.jpg",
    "floctank5.jpg",
    "floctank6.jpg",
]

all_features = []
all_labels = []

SAMPLES_PER_IMAGE = 20000   # total pixels per image for training
EDGE_RATIO = 0.6            # fraction of samples taken from high-gradient (edge) pixels


def build_features(gray_f):
    """Build all feature maps (must be same for train and test)."""
    # blur
    blur = cv2.GaussianBlur(gray_f, (5, 5), 0)

    # gradient magnitude
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    # local mean and std (7x7)
    ksize = 7
    mean_local = cv2.blur(gray_f, (ksize, ksize))
    gray_sq = gray_f * gray_f
    mean_sq_local = cv2.blur(gray_sq, (ksize, ksize))
    var_local = mean_sq_local - mean_local * mean_local
    var_local[var_local < 0] = 0
    std_local = np.sqrt(var_local)

    return blur, grad, mean_local, std_local


for fname in image_files:
    print("Processing image:", fname)
    img = cv2.imread(fname)
    if img is None:
        print("  Cannot read this image. Skipping.")
        continue

    # ---------- 1. Select ROI ----------
    while True:
        window_name = "Select ROI for " + fname
        roi_box = cv2.selectROI(window_name, img, False, False)
        x, y, w, h = roi_box
        cv2.destroyWindow(window_name)

        if w > 0 and h > 0:
            break
        print("  No ROI selected. Please try again.")

    roi = img[y:y+h, x:x+w]

    # ---------- 2. Gray ----------
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32)

    # ---------- 3. Build features ----------
    blur, grad, mean_local, std_local = build_features(gray_f)

    # ---------- 4. Otsu threshold on blur (for auto labels in ROI) ----------
    blur_uint8 = blur.astype(np.uint8)
   # enhance contrast in each area
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray_f.astype(np.uint8))

    blur_uint8 = cv2.GaussianBlur(gray_eq, (5,5), 0)

    mask = cv2.adaptiveThreshold(
    blur_uint8,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    31,
    3
)

    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # ---------- 5. Build feature vectors ----------
    f1 = gray_f.reshape(-1, 1)           # intensity
    f2 = blur.reshape(-1, 1)             # blurred intensity
    f3 = grad.reshape(-1, 1)             # gradient magnitude
    f4 = mean_local.reshape(-1, 1)       # local mean
    f5 = std_local.reshape(-1, 1)        # local std

    X_full = np.hstack([f1, f2, f3, f4, f5])   # shape: (N_pixels, 5)
    y_full = (mask_clean.reshape(-1) == 255).astype(np.uint8)

    # ---------- 6. Edge-focused sampling ----------
    N = X_full.shape[0]
    grad_flat = f3.reshape(-1)  # same as grad.flatten()

    # number of samples from edges and from the rest
    n_samples = min(SAMPLES_PER_IMAGE, N)
    n_edge = int(n_samples * EDGE_RATIO)
    n_random = n_samples - n_edge

    # sort pixels by gradient magnitude (descending)
    idx_sorted = np.argsort(-grad_flat)  # negative for descending
    edge_idx = idx_sorted[:n_edge]

    # remaining indices (non-edge pool)
    remaining_idx = idx_sorted[n_edge:]
    if n_random > 0 and remaining_idx.size > 0:
        if remaining_idx.size < n_random:
            rand_idx = remaining_idx
        else:
            rand_idx = np.random.choice(remaining_idx, size=n_random, replace=False)
        final_idx = np.concatenate([edge_idx, rand_idx])
    else:
        final_idx = edge_idx

    X_sub = X_full[final_idx]
    y_sub = y_full[final_idx]

    all_features.append(X_sub)
    all_labels.append(y_sub)

    # ---------- 7. Preview ROI + label ----------
    preview = roi.copy()
    preview[mask_clean == 255] = (0, 255, 0)   # floc = green
    preview[mask_clean == 0]   = (0, 0, 255)   # water = red

    cv2.imshow("Training ROI", roi)
    cv2.imshow("Auto mask in ROI (green=floc, red=water)", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------- 8. Stack all training data ----------
if len(all_features) == 0:
    raise ValueError("No training data collected. Check image files and ROI selection.")

X_all = np.vstack(all_features)
y_all = np.concatenate(all_labels)

print("Total training samples:", X_all.shape[0])

# ---------- 9. Train RandomForest ----------
print("Training RandomForest model (edge-focused) ...")
clf = RandomForestClassifier(
    n_estimators=350,
    max_depth=20,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=0
)
clf.fit(X_all, y_all)
print("Training finished.")

# ---------- 10. Save model ----------
model_filename = "floc_model_roi_edge.pkl"
joblib.dump(clf, model_filename)
print("Model saved to:", model_filename)
