# test_floc_with_roi_edge.py

import cv2
import numpy as np
import joblib

MODEL_PATH = "floc_model_roi_edge.pkl"   # trained edge-focused model
IMAGE_PATH = "floctank3.jpg"             # image to test
FLOC_PROB_THRESHOLD = 0.35  # adjust if needed


def build_features(gray_f):
    blur = cv2.GaussianBlur(gray_f, (3, 3), 0)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    ksize = 7
    mean_local = cv2.blur(gray_f, (ksize, ksize))
    gray_sq = gray_f * gray_f
    mean_sq_local = cv2.blur(gray_sq, (ksize, ksize))
    var_local = mean_sq_local - mean_local * mean_local
    var_local[var_local < 0] = 0
    std_local = np.sqrt(var_local)

    return blur, grad, mean_local, std_local


def analyze_floc_aggregation(mask_clean, min_area=50):
    # Convert mask to 0/1
    mask_bin = (mask_clean == 255).astype(np.uint8)

    # Find connected components (each floc cluster)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_bin, connectivity=8
    )

    floc_areas = []
    circularities = []
    solidities = []

    total_floc_area = mask_bin.sum()

    for label in range(1, num_labels):  # skip background label 0
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue  # Filter out noise/small clusters

        floc_areas.append(area)

        # Extract floc cluster for contour detection
        component_mask = (labels == label).astype(np.uint8) * 255

        # Find contours
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

    # floc size
    mean_area = floc_areas.mean()
    median_area = np.median(floc_areas)
    p90_area = np.percentile(floc_areas, 90)

    # "Large" clusters assumed if area > median
    large_mask = floc_areas > median_area
    large_area_sum = floc_areas[large_mask].sum()
    large_floc_ratio = large_area_sum / (total_floc_area + 1e-6)

    # Average shape metrics
    mean_circularity = float(circularities.mean())
    mean_solidity = float(solidities.mean())

    # 1) Base aggregation: มีการรวมตัวเป็นก้อนใหญ่แค่ไหน
    base_agg = (
        0.6 * np.tanh(large_floc_ratio * 3.0) +
        0.4 * np.tanh(mean_area / 500.0)
    )
    base_agg = float(max(0.0, min(1.0, base_agg)))

    # 2) Density: ความแน่นของก้อน floc (กัน floc ฟู/เหลว)
    density_index = 0.7 * mean_solidity + 0.3 * mean_circularity
    density_index = float(max(0.0, min(1.0, density_index)))

    # รวม: ถ้าเหลว (density ต่ำ) จะโดนลด aggregation ลง
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


def main():
    # ---------- 1. Load model ----------
    clf = joblib.load(MODEL_PATH)
    print("Model loaded from:", MODEL_PATH)

    # ---------- 2. Load image ----------
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise ValueError("Cannot read image: " + IMAGE_PATH)

    # ---------- 3. Select ROI ----------
    while True:
        window_name = "Select ROI and press ENTER"
        roi_box = cv2.selectROI(window_name, img, False, False)
        x, y, w, h = roi_box
        cv2.destroyWindow(window_name)

        if w > 0 and h > 0:
            break
        print("No ROI selected. Please try again.")

    roi = img[y:y+h, x:x+w]
    rh, rw = roi.shape[:2]
    print("ROI size:", rw, "x", rh)

    # ---------- 4. Build features (same as training) ----------
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32)

    blur, grad, mean_local, std_local = build_features(gray_f)

    f1 = gray_f.reshape(-1, 1)
    f2 = blur.reshape(-1, 1)
    f3 = grad.reshape(-1, 1)
    f4 = mean_local.reshape(-1, 1)
    f5 = std_local.reshape(-1, 1)

    X = np.hstack([f1, f2, f3, f4, f5])

    # ---------- 5. Predict probabilities and mask ----------
    print("Predicting in ROI ...")
    proba = clf.predict_proba(X)[:, 1]   # probability of class "floc"
    pred_mask = (proba > FLOC_PROB_THRESHOLD).astype(np.uint8)
    pred_mask = pred_mask.reshape(rh, rw)

    # ---------- 6. Compute floc percent ----------
    floc_pixels = np.count_nonzero(pred_mask == 1)
    total_pixels = pred_mask.size
    floc_percent = floc_pixels / total_pixels * 100.0
    water_percent = 100.0 - floc_percent

    print("Floc percent in ROI : {:.2f}%".format(floc_percent))
    print("Water percent in ROI: {:.2f}%".format(water_percent))

    # ---------- 7. Flocculation quality analysis ----------
    mask_for_analysis = (pred_mask * 255).astype(np.uint8)
    metrics = analyze_floc_aggregation(mask_for_analysis)

    for k, v in metrics.items():
        print("  {}: {}".format(k, v))

    agg = metrics["aggregation_index"]
    dens = metrics["density_index"]

    print("  aggregation_index = {:.2f}, density_index = {:.2f}".format(agg, dens))

    if agg > 0.6 and dens > 0.6:
        print("=> Summary: GOOD flocculation (big & dense flocs, not loose).")
    elif agg > 0.4 and dens > 0.4:
        print("=> Summary: MODERATE flocculation (some aggregation, medium density).")
    else:
        print("=> Summary: POOR / LOOSE flocculation (dispersed or fluffy flocs).")

    # ---------- 8. Show overlay ----------
    overlay = roi.copy()
    overlay[pred_mask == 1] = (0, 255, 0)   # floc = green
    overlay[pred_mask == 0] = (0, 0, 255)   # water = red

    cv2.imshow("ROI", roi)
    cv2.imshow("Prediction (green=floc, red=water)", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
