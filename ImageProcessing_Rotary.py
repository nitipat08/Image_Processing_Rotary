import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# -----------------------------
# Config: image file names
# -----------------------------
image_files = [
    "floctank1.jpg", "floctank2.jpg",  "floctank3.jpg", "floctank4.jpg", "floctank5.jpg",   "floctank6.jpg", "loose_1.jpg","loose_2.jpg"]


all_features = []
all_labels = []

SAMPLES_PER_IMAGE = 20000   # total pixels per image for training
EDGE_RATIO = 0.3      # fraction of samples taken from high-gradient (edge) pixels


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


def analyze_floc_aggregation(mask_clean, min_area=50):
    """
    analyze flocculation floc from mask_clean (0/255)
    return: dict ของ metrics + aggregation_index (0–1) + density_index (0–1)

    - aggregation_index: Flocculation
    - density_index: Density/compactness of floc
    """
    # ให้ mask เป็น 0/1
    mask_bin = (mask_clean == 255).astype(np.uint8)

    # หา connected components (ก้อน floc แต่ละก้อน)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_bin, connectivity=8
    )

    floc_areas = []
    circularities = []
    solidities = []

    total_floc_area = mask_bin.sum()

    for label in range(1, num_labels):  # Skip background label 0
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue  # Filter out noise/small blobs

        floc_areas.append(area)

        # Extract floc component to find contour
        component_mask = (labels == label).astype(np.uint8) * 255

        # Find contour
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 0:
            continue

        cnt = contours[0]
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4.0 * np.pi * area / (perimeter**2)
        else:
            circularity = 0.0

        # solidity = area / convex_hull_area
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = float(area) / hull_area
        else:
            solidity = 0.0

        circularities.append(circularity)
        solidities.append(solidity)

    if len(floc_areas) == 0:
        return {
            "num_flocs": 0,
            "mean_area": 0.0,
            "median_area": 0.0,
            "p90_area": 0.0,
            "large_floc_ratio": 0.0,
            "mean_circularity": 0.0,
            "mean_solidity": 0.0,
            "aggregation_index": 0.0,
            "density_index": 0.0,
        }

    floc_areas = np.array(floc_areas, dtype=np.float32)
    circularities = np.array(circularities, dtype=np.float32)
    solidities = np.array(solidities, dtype=np.float32)

    # Floc size
    mean_area = floc_areas.mean()
    median_area = np.median(floc_areas)
    p90_area = np.percentile(floc_areas, 90)

    # “Large floc” assumed if area > median
    large_mask = floc_areas > median_area
    large_area_sum = floc_areas[large_mask].sum()
    large_floc_ratio = large_area_sum / (total_floc_area + 1e-6)

    # Mean shape metrics
    mean_circularity = float(circularities.mean())
    mean_solidity = float(solidities.mean())

    # 1) base aggregation: How much flocs aggregate into large clusters
    base_agg = (
        0.6 * np.tanh(large_floc_ratio * 3.0) +
        0.4 * np.tanh(mean_area / 500.0)
    )
    base_agg = float(max(0.0, min(1.0, base_agg)))

    # 2) density: ความแน่นของก้อน floc (กันเคสฟลอคฟู/เหลว)
    density_index = 0.7 * mean_solidity + 0.3 * mean_circularity
    density_index = float(max(0.0, min(1.0, density_index)))

    # รวม: ถ้า density ต่ำ (เหลว) จะไปลด aggregation ลง
    aggregation_index = base_agg * density_index
    aggregation_index = float(max(0.0, min(1.0, aggregation_index)))

    return {
        "num_flocs": int(len(floc_areas)),
        "mean_area": float(mean_area),
        "median_area": float(median_area),
        "p90_area": float(p90_area),
        "large_floc_ratio": float(large_floc_ratio),
        "mean_circularity": mean_circularity,
        "mean_solidity": mean_solidity,
        "aggregation_index": aggregation_index,
        "density_index": density_index,
    }


# -----------------------------
# Training loop
# -----------------------------
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

    # ---------- 4. Thresholding / mask floc ----------
    # enhance contrast in each area
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray_f.astype(np.uint8))

    blur_uint8 = cv2.GaussianBlur(gray_eq, (5, 5), 0)

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

    # ---------- 4.1 วิเคราะห์การจับตัวของ floc ----------
    metrics = analyze_floc_aggregation(mask_clean)
    print("  Floc aggregation metrics for", fname, ":")
    for k, v in metrics.items():
        print("    {}: {}".format(k, v))

    agg = metrics["aggregation_index"]
    dens = metrics["density_index"]
    print("    aggregation_index = {:.2f}, density_index = {:.2f}".format(agg, dens))

    if agg > 0.6 and dens > 0.6:
        print("  => Good Flocculation (big & dense flocs)")
    elif agg > 0.4 and dens > 0.4:
        print("  => Moderate Flocculation")
    else:
        print("  => Poor / Loose / Dispersed Flocculation")

    # ---------- 5. Build feature vectors for pixel RF ----------
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
