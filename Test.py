# test_floc_with_roi_edge.py

import cv2
import numpy as np
import joblib

MODEL_PATH = "floc_model_roi_edge.pkl"   # trained edge-focused model
IMAGE_PATH = "floctank2.jpg"             # image to test
FLOC_PROB_THRESHOLD = 0.3              # adjust if needed


def build_features(gray_f):
    blur = cv2.GaussianBlur(gray_f, (5, 5), 0)
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

    # ---------- 7. Show overlay ----------
    overlay = roi.copy()
    overlay[pred_mask == 1] = (0, 255, 0)   # floc = green
    overlay[pred_mask == 0] = (0, 0, 255)   # water = red

    cv2.imshow("ROI", roi)
    cv2.imshow("Prediction (green=floc, red=water)", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
