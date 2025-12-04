import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# -----------------------------
# Config
# -----------------------------
image_files = [
    "floctank1.jpg", "floctank2.jpg", "floctank3.jpg", "floctank4.jpg",
    "floctank5.jpg", "floctank6.jpg", "loose_1.jpg", "loose_2.jpg"
]

all_features = []
all_labels = []

SAMPLES_PER_IMAGE = 30000
STRATIFIED_SAMPLING = True


def preprocess_image(img, target_size=(512, 512), crop_border_ratio=0.1):
    """Crop border and resize to consistent size."""
    h, w = img.shape[:2]
    dx = int(w * crop_border_ratio)
    dy = int(h * crop_border_ratio)
    x0, x1 = max(0, dx), min(w, w - dx)
    y0, y1 = max(0, dy), min(h, h - dy)
    
    crop = img[y0:y1, x0:x1]
    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
    return resized


def compute_lbp(gray):
    """Compute Local Binary Pattern for texture."""
    h, w = gray.shape
    lbp = np.zeros_like(gray, dtype=np.float32)
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            center = gray[i, j]
            code = 0
            code |= (gray[i-1, j-1] >= center) << 7
            code |= (gray[i-1, j] >= center) << 6
            code |= (gray[i-1, j+1] >= center) << 5
            code |= (gray[i, j+1] >= center) << 4
            code |= (gray[i+1, j+1] >= center) << 3
            code |= (gray[i+1, j] >= center) << 2
            code |= (gray[i+1, j-1] >= center) << 1
            code |= (gray[i, j-1] >= center) << 0
            lbp[i, j] = code
    
    return lbp


def analyze_image_homogeneity(gray_f, features_dict):
    """
    วิเคราะห์ว่าภาพเป็น homogeneous (สีเดียวกัน/เหลว) หรือไม่
    ยิ่ง homogeneous มาก = floc ยิ่งแย่
    
    Returns: homogeneity_score (0-1), 0=หลากหลาย(ดี), 1=เหลวมาก(แย่)
    """
    # 1. Check intensity variation
    int_std = gray_f.std()
    int_range = gray_f.max() - gray_f.min()
    
    # 2. Check texture variation (local std)
    std_local = features_dict['std_local']
    texture_var = std_local.std()  # variation of local std
    
    # 3. Check gradient distribution
    grad_mag = features_dict['grad_mag']
    grad_mean = grad_mag.mean()
    grad_std = grad_mag.std()
    
    # 4. Check LBP histogram entropy (texture complexity)
    lbp = features_dict['lbp']
    lbp_hist, _ = np.histogram(lbp.flatten(), bins=32, range=(0, 256))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)
    lbp_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))
    
    # Normalize metrics
    # Low values = homogeneous = bad
    int_std_norm = np.clip(int_std / 25.0, 0, 1)  # เข้มงวดขึ้น: expect std > 25
    texture_var_norm = np.clip(texture_var / 4.0, 0, 1)  # เข้มงวดขึ้น
    grad_mean_norm = np.clip(grad_mean / 15.0, 0, 1)  # เข้มงวดขึ้น
    lbp_entropy_norm = np.clip(lbp_entropy / 4.0, 0, 1)
    
    # Homogeneity score: 1 = very homogeneous (bad), 0 = diverse (good)
    homogeneity_score = 1.0 - (
        0.35 * int_std_norm +      # เพิ่มน้ำหนัก intensity
        0.35 * texture_var_norm +  # เพิ่มน้ำหนัก texture
        0.2 * grad_mean_norm +
        0.1 * lbp_entropy_norm
    )
    homogeneity_score = float(np.clip(homogeneity_score, 0.0, 1.0))
    
    return {
        'homogeneity_score': homogeneity_score,
        'int_std': float(int_std),
        'int_range': float(int_range),
        'texture_var': float(texture_var),
        'grad_mean': float(grad_mean),
        'lbp_entropy': float(lbp_entropy),
        'is_homogeneous': homogeneity_score > 0.55  # เข้มงวดขึ้น จาก 0.65
    }


def build_features(gray_f):
    """Build comprehensive feature maps."""
    # 1. Blur
    blur = cv2.GaussianBlur(gray_f, (5, 5), 0)
    
    # 2. Gradient
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    grad_dir = cv2.phase(gx, gy, angleInDegrees=True)
    
    # 3. Local statistics (7x7)
    ksize = 7
    mean_local = cv2.blur(gray_f, (ksize, ksize))
    gray_sq = gray_f * gray_f
    mean_sq_local = cv2.blur(gray_sq, (ksize, ksize))
    var_local = np.maximum(0, mean_sq_local - mean_local * mean_local)
    std_local = np.sqrt(var_local)
    
    # 4. Larger window statistics (15x15)
    ksize_lg = 15
    mean_large = cv2.blur(gray_f, (ksize_lg, ksize_lg))
    mean_sq_large = cv2.blur(gray_sq, (ksize_lg, ksize_lg))
    var_large = np.maximum(0, mean_sq_large - mean_large * mean_large)
    std_large = np.sqrt(var_large)
    
    # 5. Laplacian (edge detection)
    laplacian = cv2.Laplacian(gray_f, cv2.CV_32F, ksize=3)
    laplacian = np.abs(laplacian)
    
    # 6. LBP texture
    gray_uint8 = np.clip(gray_f, 0, 255).astype(np.uint8)
    lbp = compute_lbp(gray_uint8).astype(np.float32)
    
    return {
        'blur': blur,
        'grad_mag': grad_mag,
        'grad_dir': grad_dir,
        'mean_local': mean_local,
        'std_local': std_local,
        'mean_large': mean_large,
        'std_large': std_large,
        'laplacian': laplacian,
        'lbp': lbp
    }


def create_mask_kmeans_improved(gray_f, features_dict):
    """Improved K-Means segmentation with more features."""
    h, w = gray_f.shape
    
    # Stack features for clustering
    feat_list = [
        gray_f.reshape(-1, 1),
        features_dict['blur'].reshape(-1, 1),
        features_dict['grad_mag'].reshape(-1, 1),
        features_dict['std_local'].reshape(-1, 1),
        features_dict['std_large'].reshape(-1, 1),
        features_dict['laplacian'].reshape(-1, 1),
        features_dict['lbp'].reshape(-1, 1)
    ]
    
    feats = np.hstack(feat_list)
    
    # Standardize features
    scaler = StandardScaler()
    feats_norm = scaler.fit_transform(feats)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10, max_iter=300)
    labels_k = kmeans.fit_predict(feats_norm)
    labels_img = labels_k.reshape(h, w)
    
    # Determine which cluster is floc
    std_flat = features_dict['std_local'].reshape(-1)
    grad_flat = features_dict['grad_mag'].reshape(-1)
    int_flat = gray_f.reshape(-1)
    
    # Statistics for each cluster
    stats = {}
    for c in [0, 1]:
        mask_c = labels_k == c
        stats[c] = {
            'mean_std': std_flat[mask_c].mean(),
            'mean_grad': grad_flat[mask_c].mean(),
            'mean_int': int_flat[mask_c].mean(),
            'count': mask_c.sum()
        }
    
    # Floc typically has higher texture variation and gradient
    score_0 = (stats[0]['mean_std'] / (stats[1]['mean_std'] + 1e-6) + 
               stats[0]['mean_grad'] / (stats[1]['mean_grad'] + 1e-6))
    score_1 = (stats[1]['mean_std'] / (stats[0]['mean_std'] + 1e-6) + 
               stats[1]['mean_grad'] / (stats[0]['mean_grad'] + 1e-6))
    
    floc_label = 0 if score_0 > score_1 else 1
    
    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[labels_img == floc_label] = 255
    
    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Check if clustering failed (very similar clusters)
    floc_ratio = (mask == 255).sum() / mask.size
    diff_std = abs(stats[0]['mean_std'] - stats[1]['mean_std'])
    
    # If clusters are too similar, treat entire image as floc (will be penalized later)
    if diff_std < 1.0 or floc_ratio > 0.95:
        mask[:, :] = 255
    
    return mask, stats


def analyze_floc_aggregation(mask_clean, gray_f, features_dict, min_area_ratio=0.0003):
    """
    Analyze floc aggregation - ปรับให้จับ homogeneous images เป็น POOR
    """
    mask_bin = (mask_clean == 255).astype(np.uint8)
    roi_area = mask_bin.size
    total_floc_area = mask_bin.sum()
    min_area = max(1, int(min_area_ratio * roi_area))
    
    # Check homogeneity FIRST
    homog = analyze_image_homogeneity(gray_f, features_dict)
    
    # If highly homogeneous -> automatic POOR rating
    if homog['is_homogeneous']:
        print(f"  WARNING: HOMOGENEOUS IMAGE DETECTED (score: {homog['homogeneity_score']:.3f})")
        print(f"      int_std={homog['int_std']:.1f}, texture_var={homog['texture_var']:.1f}")
        return {
            "num_flocs": 0,
            "num_large_flocs": 0,
            "num_xlarge_flocs": 0,
            "mean_area": 0.0,
            "median_area": 0.0,
            "p90_area": 0.0,
            "max_area": 0.0,
            "large_floc_ratio": 0.0,
            "large_floc_count_ratio": 0.0,
            "mean_circularity": 0.0,
            "mean_solidity": 0.0,
            "mean_aspect_ratio": 0.0,
            "mean_edge_strength": 0.0,
            "aggregation_index": 0.0,
            "density_index": 0.0,
            "homogeneity_score": homog['homogeneity_score'],
            "is_homogeneous": True,
            "is_dispersed": False,
            "dispersion_penalty": 0.0,
            "dispersion_reason": "Homogeneous",
            "few_flocs_reason": "Homogeneous"
        }
    
    # Continue with normal analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_bin, connectivity=8
    )
    
    floc_areas = []
    circularities = []
    solidities = []
    aspect_ratios = []
    edge_strengths = []
    
    grad_mag = features_dict['grad_mag']
    
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        
        floc_areas.append(area)
        aspect_ratios.append(max(w, h) / (min(w, h) + 1e-6))
        
        # Measure edge strength for this floc
        component_mask = (labels == label)
        floc_edges = grad_mag[component_mask]
        edge_strength = floc_edges.mean()
        edge_strengths.append(edge_strength)
        
        component_mask_uint8 = component_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) > 0:
            cnt = contours[0]
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4.0 * np.pi * area / (perimeter**2 + 1e-6)
            circularities.append(circularity)
            
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / (hull_area + 1e-6)
            solidities.append(solidity)
    
    if len(floc_areas) == 0:
        return {
            "num_flocs": 0,
            "num_large_flocs": 0,
            "num_xlarge_flocs": 0,
            "mean_area": 0.0,
            "median_area": 0.0,
            "p90_area": 0.0,
            "max_area": 0.0,
            "large_floc_ratio": 0.0,
            "large_floc_count_ratio": 0.0,
            "mean_circularity": 0.0,
            "mean_solidity": 0.0,
            "mean_aspect_ratio": 0.0,
            "mean_edge_strength": 0.0,
            "aggregation_index": 0.0,
            "density_index": 0.0,
            "homogeneity_score": homog['homogeneity_score'],
            "is_homogeneous": False,
            "is_dispersed": False,
            "dispersion_penalty": 0.0,
            "dispersion_reason": "No flocs",
            "few_flocs_reason": "No flocs"
        }
    
    floc_areas = np.array(floc_areas, dtype=np.float32)
    circularities = np.array(circularities, dtype=np.float32)
    solidities = np.array(solidities, dtype=np.float32)
    aspect_ratios = np.array(aspect_ratios, dtype=np.float32)
    edge_strengths = np.array(edge_strengths, dtype=np.float32)
    
    mean_area = floc_areas.mean()
    median_area = np.median(floc_areas)
    p90_area = np.percentile(floc_areas, 90)
    max_area = floc_areas.max()
    min_area = floc_areas.min()
    mean_area_norm = mean_area / (roi_area + 1e-6)

    # จำนวน floc ทั้งหมด ต้องประกาศตรงนี้ก่อนคำนวณ ratio ต่าง ๆ
    num_flocs = len(floc_areas)

    # Define "large" flocs - multiple thresholds
    threshold_large = 2.0 * median_area   # flocs > 2x median
    threshold_xlarge = 5.0 * median_area  # very large flocs > 5x median

    large_mask = floc_areas > threshold_large
    xlarge_mask = floc_areas > threshold_xlarge

    num_large_flocs = large_mask.sum()
    num_xlarge_flocs = xlarge_mask.sum()

    large_area_sum = floc_areas[large_mask].sum()
    large_floc_ratio = large_area_sum / (total_floc_area + 1e-6)

    # % ของจำนวน floc ที่เป็นก้อนใหญ่ (ไม่ใช่แค่พื้นที่)
    large_floc_count_ratio = num_large_flocs / (num_flocs + 1e-6)
    xlarge_floc_count_ratio = num_xlarge_flocs / (num_flocs + 1e-6)

    mean_circularity = float(circularities.mean())
    mean_solidity = float(solidities.mean())
    mean_aspect_ratio = float(aspect_ratios.mean())
    mean_edge_strength = float(edge_strengths.mean())

    # === KEY METRIC: ตรวจสอบว่ามีก้อนใหญ่ชัดเจนหรือไม่ ===
    # Good floc = มีก้อนใหญ่จริงๆ หลายก้อน หรือ มีก้อนยักษ์สักก้อน
    has_good_large_flocs = (num_large_flocs >= 3) or (num_xlarge_flocs >= 1)
    
    # === DISPERSION CHECK ===
    # Bad cases:
    # 1. มี floc เยอะมาก (>50) แต่เล็กๆ
    # 2. มี floc ปานกลาง (>30) แต่ขนาดเฉลี่ยเล็กมาก
    # 3. มี floc เยอะ (>20) แต่ไม่มีก้อนใหญ่เลย
    
    if num_flocs > 50 and mean_area_norm < 0.001:
        dispersion_penalty = 0.2
        is_dispersed = True
        dispersion_reason = "Too many tiny flocs"
    elif num_flocs > 30 and mean_area_norm < 0.002:
        dispersion_penalty = 0.4
        is_dispersed = True
        dispersion_reason = "Many small flocs"
    elif num_flocs > 20 and not has_good_large_flocs:
        dispersion_penalty = 0.5
        is_dispersed = True
        dispersion_reason = "No distinct large flocs"
    elif num_flocs > 15 and large_floc_count_ratio < 0.15:
        dispersion_penalty = 0.7
        is_dispersed = True
        dispersion_reason = "Very few large flocs"
    else:
        dispersion_penalty = 1.0
        is_dispersed = False
        dispersion_reason = "OK"
    
    # === HOMOGENEOUS CHECK (too few flocs) ===
    if num_flocs <= 2:
        floc_count_penalty = 0.2
        few_flocs_reason = "Only 1-2 large blobs (likely homogeneous)"
    elif num_flocs <= 5:
        floc_count_penalty = 0.5
        few_flocs_reason = "Very few flocs"
    else:
        floc_count_penalty = 1.0
        few_flocs_reason = "OK"
    
    # === AGGREGATION INDEX (แก้ใหม่) ===
    # Good floc = ก้อนใหญ่ + มีจำนวนพอสมควร (5-30 ก้อน) + มีความหนาแน่น
    
    # 1. Size component: ยิ่งก้อนใหญ่ยิ่งดี
    size_score = 0.4 * np.tanh(mean_area_norm * 400.0) + 0.2 * np.tanh(max_area / (roi_area * 0.1))
    
    # 2. Count component: Sweet spot = 5-30 flocs, too many = bad, too few = bad
    if 5 <= num_flocs <= 30:
        count_score = 0.3
    elif num_flocs < 5:
        count_score = 0.3 * (num_flocs / 5.0)  # linear penalty
    else:  # > 30
        count_score = 0.3 * np.exp(-(num_flocs - 30) / 20.0)  # exponential decay
    
    # 3. Large floc component: มีก้อนใหญ่ชัดเจนกี่ %
    large_score = 0.2 * large_floc_count_ratio + 0.1 * large_floc_ratio
    
    base_agg = size_score + count_score + large_score
    base_agg = float(np.clip(base_agg, 0.0, 1.0))
    
    # Apply penalties
    base_agg *= floc_count_penalty * dispersion_penalty
    
    # Density index (compactness) - includes edge strength
    edge_factor = np.tanh(mean_edge_strength / 15.0)  # strong edges = good
    density_index = (
        0.5 * mean_solidity + 
        0.25 * mean_circularity + 
        0.15 * (1.0 / (mean_aspect_ratio + 0.5)) +
        0.1 * edge_factor
    )
    density_index = float(np.clip(density_index, 0.0, 1.0))
    
    # Final aggregation: penalized by homogeneity
    aggregation_index = base_agg * (0.6 + 0.4 * density_index)
    homogeneity_penalty = 1.0 - (0.5 * homog['homogeneity_score'])  # reduce up to 50%
    aggregation_index *= homogeneity_penalty
    aggregation_index = float(np.clip(aggregation_index, 0.0, 1.0))
    
    return {
        "num_flocs": int(num_flocs),
        "num_large_flocs": int(num_large_flocs),
        "num_xlarge_flocs": int(num_xlarge_flocs),
        "mean_area": float(mean_area),
        "median_area": float(median_area),
        "p90_area": float(p90_area),
        "max_area": float(max_area),
        "large_floc_ratio": float(large_floc_ratio),
        "large_floc_count_ratio": float(large_floc_count_ratio),
        "mean_circularity": mean_circularity,
        "mean_solidity": mean_solidity,
        "mean_aspect_ratio": mean_aspect_ratio,
        "mean_edge_strength": mean_edge_strength,
        "aggregation_index": aggregation_index,
        "density_index": density_index,
        "homogeneity_score": homog['homogeneity_score'],
        "is_homogeneous": False,
        "is_dispersed": is_dispersed,
        "dispersion_penalty": dispersion_penalty,
        "dispersion_reason": dispersion_reason,
        "few_flocs_reason": few_flocs_reason
    }


# -----------------------------
# Training loop
# -----------------------------
for fname in image_files:
    print(f"\n{'='*60}")
    print(f"Processing: {fname}")
    print('='*60)
    
    img = cv2.imread(fname)
    if img is None:
        print(f"  Cannot read {fname}. Skipping.")
        continue
    
    # Preprocess
    roi = preprocess_image(img, target_size=(512, 512), crop_border_ratio=0.1)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32)
    
    # Extract features
    print("  Extracting features...")
    features_dict = build_features(gray_f)
    
    # Create mask
    print("  Creating floc mask...")
    mask_clean, cluster_stats = create_mask_kmeans_improved(gray_f, features_dict)
    
    print(f"  Cluster 0: std={cluster_stats[0]['mean_std']:.2f}, grad={cluster_stats[0]['mean_grad']:.2f}")
    print(f"  Cluster 1: std={cluster_stats[1]['mean_std']:.2f}, grad={cluster_stats[1]['mean_grad']:.2f}")
    
    # Analyze floc
    metrics = analyze_floc_aggregation(mask_clean, gray_f, features_dict)
    print(f"\n  Floc Metrics:")
    print(f"    Floc coverage: {(mask_clean==255).sum()/mask_clean.size*100:.1f}%")
    print(f"    Homogeneity: {metrics['homogeneity_score']:.3f} {'[TOO UNIFORM]' if metrics.get('is_homogeneous', False) else '[OK]'}")
    print(f"    Total flocs: {metrics['num_flocs']} | Large: {metrics.get('num_large_flocs', 0)} | XLarge: {metrics.get('num_xlarge_flocs', 0)}")
    print(f"    Mean area: {metrics['mean_area']:.1f} | Max: {metrics.get('max_area', 0):.1f} pixels")
    print(f"    Large floc count ratio: {metrics.get('large_floc_count_ratio', 0):.3f}")
    
    if metrics.get('is_dispersed', False):
        print(f"    [DISPERSED]: {metrics.get('dispersion_reason', '')} (penalty: {metrics.get('dispersion_penalty', 1.0):.2f})")
    
    if metrics.get('few_flocs_reason', 'OK') != 'OK':
        print(f"    [FEW FLOCS]: {metrics.get('few_flocs_reason', '')}")
    
    print(f"    Mean edge strength: {metrics['mean_edge_strength']:.2f}")
    print(f"    Mean solidity: {metrics['mean_solidity']:.3f}")
    print(f"    Mean circularity: {metrics['mean_circularity']:.3f}")
    print(f"    Mean solidity: {metrics['mean_solidity']:.3f}")
    print(f"    Aggregation index: {metrics['aggregation_index']:.3f}")
    print(f"    Density index: {metrics['density_index']:.3f}")
    
    agg = metrics['aggregation_index']
    dens = metrics['density_index']
    homog = metrics['homogeneity_score']
    
    # Revised quality assessment
    if metrics.get('is_homogeneous', False) or homog > 0.55:
        quality = "VERY POOR - Homogeneous/liquid (no distinct flocs)"
    elif metrics.get('is_dispersed', False) or agg < 0.25:
        quality = "POOR - Dispersed/weak flocs (many tiny pieces)"
    elif agg < 0.45 or metrics['num_flocs'] > 40:
        quality = "FAIR - Some aggregation but dispersed"
    elif agg < 0.65:
        quality = "GOOD - Moderate floc formation"
    else:
        quality = "EXCELLENT - Strong, well-formed flocs"
    
    print(f"\n  => Quality: {quality}")
    
    # Build feature vectors for RF training
    X_full = np.hstack([
        gray_f.reshape(-1, 1),
        features_dict['blur'].reshape(-1, 1),
        features_dict['grad_mag'].reshape(-1, 1),
        features_dict['std_local'].reshape(-1, 1),
        features_dict['std_large'].reshape(-1, 1),
        features_dict['laplacian'].reshape(-1, 1),
        features_dict['lbp'].reshape(-1, 1),
        features_dict['mean_local'].reshape(-1, 1),
    ])
    
    y_full = (mask_clean.reshape(-1) == 255).astype(np.uint8)
    
    # Stratified sampling
    n_samples = min(SAMPLES_PER_IMAGE, X_full.shape[0])
    
    if STRATIFIED_SAMPLING:
        floc_idx = np.where(y_full == 1)[0]
        water_idx = np.where(y_full == 0)[0]
        
        n_per_class = n_samples // 2
        
        if len(floc_idx) > n_per_class:
            floc_sample = np.random.choice(floc_idx, size=n_per_class, replace=False)
        else:
            floc_sample = floc_idx
        
        if len(water_idx) > n_per_class:
            water_sample = np.random.choice(water_idx, size=n_per_class, replace=False)
        else:
            water_sample = water_idx
        
        final_idx = np.concatenate([floc_sample, water_sample])
    else:
        final_idx = np.random.choice(X_full.shape[0], size=n_samples, replace=False)
    
    X_sub = X_full[final_idx]
    y_sub = y_full[final_idx]
    
    all_features.append(X_sub)
    all_labels.append(y_sub)
    
    print(f"  Sampled {len(final_idx)} pixels (floc: {y_sub.sum()}, water: {len(y_sub)-y_sub.sum()})")
    
    # Visualization
    preview = roi.copy()
    preview[mask_clean == 255] = [0, 255, 0]  # floc = green
    preview[mask_clean == 0] = [0, 0, 255]    # water = red
    
    cv2.imshow("Original ROI", roi)
    cv2.imshow("Segmentation (Green=Floc, Red=Water)", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Stack training data
if len(all_features) == 0:
    raise ValueError("No training data collected!")

X_all = np.vstack(all_features)
y_all = np.concatenate(all_labels)

print(f"\n{'='*60}")
print(f"Total training samples: {X_all.shape[0]}")
print(f"  Floc pixels: {y_all.sum()} ({100*y_all.sum()/len(y_all):.1f}%)")
print(f"  Water pixels: {len(y_all)-y_all.sum()} ({100*(len(y_all)-y_all.sum())/len(y_all):.1f}%)")
print('='*60)

# Train Random Forest
print("\nTraining Random Forest classifier...")
clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42,
    class_weight='balanced'
)

clf.fit(X_all, y_all)

# Feature importance
feature_names = ['intensity', 'blur', 'grad_mag', 'std_local', 
                 'std_large', 'laplacian', 'lbp', 'mean_local']
importances = clf.feature_importances_

print("\nFeature Importances:")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {imp:.4f}")

# Save model
model_filename = "floc_model_improved.pkl"
joblib.dump(clf, model_filename)
print(f"\nModel saved to: {model_filename}")
print("Training complete!")