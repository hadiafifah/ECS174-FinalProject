import numpy as np

# Landmark indices
FACE_LEFT_CHEEK = 234
FACE_RIGHT_CHEEK = 454

MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP_INNER = 13
MOUTH_BOTTOM_INNER = 14
NOSE_TIP = 1

RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145

LEFT_EYE_OUTER = 263
LEFT_EYE_INNER = 362
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374

LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466,
                    388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173,
                     157, 158, 159, 160, 161, 246]

LEFT_EYEBROW_INDICES = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
RIGHT_EYEBROW_INDICES = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]


def make_feature_array(feature_dict):
    """
    Takes a dict {name: value} and returns:
      - values as np.float32 array
      - list of names in matching order
    """
    names = list(feature_dict.keys())
    values = np.array(list(feature_dict.values()), dtype=np.float32)
    return values, names


def _dist(p1, p2):
    return np.linalg.norm(p1 - p2)


def _slope(p_from, p_to):
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    return dy / (dx + 1e-6)


def _signed_point_line_distance_2d(p, a, b):
    """
    Signed distance from point p to the line through a and b in xy.
    Uses 2D cross product for sign.
    """
    p2 = p[:2]
    a2 = a[:2]
    b2 = b[:2]
    v = b2 - a2
    w = p2 - a2
    cross = v[0] * w[1] - v[1] * w[0]
    denom = np.linalg.norm(v) + 1e-6
    return cross / denom


def _get_inner_outer_brow_points(pts, indices, side):
    """
    side: 'left' or 'right' from subject perspective.
    Uses x coordinate to choose inner vs outer point.
    """
    brow_pts = pts[indices]          # shape (N, 3)
    xs = brow_pts[:, 0]

    if side == "left":
        # Subject left brow is on viewer right
        # inner is closer to center (smaller x), outer larger x
        inner_local = int(xs.argmin())
        outer_local = int(xs.argmax())
    else:
        # Subject right brow is on viewer left
        # inner is closer to center (larger x), outer smaller x
        inner_local = int(xs.argmax())
        outer_local = int(xs.argmin())

    inner_pt = brow_pts[inner_local]
    outer_pt = brow_pts[outer_local]
    return inner_pt, outer_pt


def extract_features(landmarks):
    """
    landmarks: list of 468 mediapipe landmarks (with .x, .y, .z)
    Returns:
        features: 1D np.array of floats
        feature_names: list of strings in matching order
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    # basic reference distances
    left_cheek = pts[FACE_LEFT_CHEEK]
    right_cheek = pts[FACE_RIGHT_CHEEK]
    face_width = _dist(left_cheek, right_cheek) + 1e-6

    forehead = pts[10]
    chin = pts[152]
    face_height = _dist(forehead, chin) + 1e-6

    # mouth points
    mouth_left = pts[MOUTH_LEFT]
    mouth_right = pts[MOUTH_RIGHT]
    mouth_top = pts[MOUTH_TOP_INNER]
    mouth_bottom = pts[MOUTH_BOTTOM_INNER]
    nose_tip = pts[NOSE_TIP]
    mouth_center = 0.5 * (mouth_left + mouth_right)

    # base mouth features
    mouth_width = _dist(mouth_left, mouth_right) / face_width
    mouth_height = _dist(mouth_top, mouth_bottom) / face_width
    mouth_ar = mouth_height / (mouth_width + 1e-6)

    mouth_corner_asym = (mouth_left[1] - mouth_right[1])
    mouth_center_nose_dist = _dist(mouth_center, nose_tip) / face_height

    slope_left = _slope(mouth_center, mouth_left)
    slope_right = _slope(mouth_center, mouth_right)
    slope_mean = 0.5 * (slope_left + slope_right)
    slope_diff = slope_left - slope_right

    # extra mouth geometry
    lip_thickness_mid = _dist(mouth_top, mouth_bottom) / face_height

    mouth_corner_top_left = _dist(mouth_left, mouth_top) / face_height
    mouth_corner_top_right = _dist(mouth_right, mouth_top) / face_height
    mouth_corner_bottom_left = _dist(mouth_left, mouth_bottom) / face_height
    mouth_corner_bottom_right = _dist(mouth_right, mouth_bottom) / face_height

    mouth_corner_nose_left = _dist(mouth_left, nose_tip) / face_height
    mouth_corner_nose_right = _dist(mouth_right, nose_tip) / face_height

    upper_lip_curvature = _signed_point_line_distance_2d(
        mouth_top, mouth_left, mouth_right
    ) / (face_height + 1e-6)
    lower_lip_curvature = _signed_point_line_distance_2d(
        mouth_bottom, mouth_left, mouth_right
    ) / (face_height + 1e-6)

    # eyes
    left_eye_outer = pts[LEFT_EYE_OUTER]
    left_eye_inner = pts[LEFT_EYE_INNER]
    left_eye_top = pts[LEFT_EYE_TOP]
    left_eye_bottom = pts[LEFT_EYE_BOTTOM]

    right_eye_outer = pts[RIGHT_EYE_OUTER]
    right_eye_inner = pts[RIGHT_EYE_INNER]
    right_eye_top = pts[RIGHT_EYE_TOP]
    right_eye_bottom = pts[RIGHT_EYE_BOTTOM]

    left_eye_width = _dist(left_eye_outer, left_eye_inner) / face_width
    left_eye_height = _dist(left_eye_top, left_eye_bottom) / face_width

    right_eye_width = _dist(right_eye_outer, right_eye_inner) / face_width
    right_eye_height = _dist(right_eye_top, right_eye_bottom) / face_width

    left_ear = left_eye_height / (left_eye_width + 1e-6)
    right_ear = right_eye_height / (right_eye_width + 1e-6)
    mean_ear = 0.5 * (left_ear + right_ear)
    ear_diff = left_ear - right_ear

    # eye centers and cheek distances
    left_eye_center = pts[LEFT_EYE_INDICES].mean(axis=0)
    right_eye_center = pts[RIGHT_EYE_INDICES].mean(axis=0)

    left_cheek_eye = _dist(left_cheek, left_eye_center) / face_width
    right_cheek_eye = _dist(right_cheek, right_eye_center) / face_width
    cheek_eye_mean = 0.5 * (left_cheek_eye + right_cheek_eye)
    cheek_eye_diff = left_cheek_eye - right_cheek_eye

    # eyebrow to eye distances (existing mean based)
    left_eye_mean_y = pts[LEFT_EYE_INDICES][:, 1].mean()
    right_eye_mean_y = pts[RIGHT_EYE_INDICES][:, 1].mean()

    left_brow_mean_y = pts[LEFT_EYEBROW_INDICES][:, 1].mean()
    right_brow_mean_y = pts[RIGHT_EYEBROW_INDICES][:, 1].mean()

    left_brow_eye = (left_brow_mean_y - left_eye_mean_y) / (face_height + 1e-6)
    right_brow_eye = (right_brow_mean_y - right_eye_mean_y) / (face_height + 1e-6)
    brow_eye_mean = 0.5 * (left_brow_eye + right_brow_eye)
    brow_eye_diff = left_brow_eye - right_brow_eye

    # inner vs outer brow points and richer brow geometry
    left_inner_brow, left_outer_brow = _get_inner_outer_brow_points(
        pts, LEFT_EYEBROW_INDICES, side="left"
    )
    right_inner_brow, right_outer_brow = _get_inner_outer_brow_points(
        pts, RIGHT_EYEBROW_INDICES, side="right"
    )

    inner_brow_dist = _dist(left_inner_brow, right_inner_brow) / face_width

    left_brow_tilt = _slope(left_inner_brow, left_outer_brow)
    right_brow_tilt = _slope(right_inner_brow, right_outer_brow)
    brow_tilt_diff = left_brow_tilt - right_brow_tilt

    left_brow_mid = pts[LEFT_EYEBROW_INDICES].mean(axis=0)
    right_brow_mid = pts[RIGHT_EYEBROW_INDICES].mean(axis=0)

    left_brow_curv = _signed_point_line_distance_2d(
        left_brow_mid, left_inner_brow, left_outer_brow
    ) / (face_height + 1e-6)
    right_brow_curv = _signed_point_line_distance_2d(
        right_brow_mid, right_inner_brow, right_outer_brow
    ) / (face_height + 1e-6)
    brow_curv_mean = 0.5 * (left_brow_curv + right_brow_curv)
    brow_curv_diff = left_brow_curv - right_brow_curv

    left_inner_brow_eye = (left_inner_brow[1] - left_eye_center[1]) / (face_height + 1e-6)
    left_outer_brow_eye = (left_outer_brow[1] - left_eye_center[1]) / (face_height + 1e-6)
    right_inner_brow_eye = (right_inner_brow[1] - right_eye_center[1]) / (face_height + 1e-6)
    right_outer_brow_eye = (right_outer_brow[1] - right_eye_center[1]) / (face_height + 1e-6)

    inner_brow_eye_mean = 0.5 * (left_inner_brow_eye + right_inner_brow_eye)
    outer_brow_eye_mean = 0.5 * (left_outer_brow_eye + right_outer_brow_eye)
    inner_outer_brow_eye_diff = inner_brow_eye_mean - outer_brow_eye_mean

    # global geometry
    face_aspect = face_height / face_width

    v = right_cheek - left_cheek
    head_tilt_angle = np.arctan2(v[1], v[0])

    mouth_corner_asym_norm = mouth_corner_asym / (face_height + 1e-6)

    head_tilt_sin = np.sin(head_tilt_angle)
    head_tilt_cos = np.cos(head_tilt_angle)

    # build dictionary of all features so names and values stay aligned
    feature_dict = {
        # mouth
        "mouth_width": mouth_width,
        "mouth_height": mouth_height,
        "mouth_ar": mouth_ar,
        "mouth_corner_asym_norm": mouth_corner_asym_norm,
        "mouth_center_nose_dist": mouth_center_nose_dist,
        "mouth_slope_left": slope_left,
        "mouth_slope_right": slope_right,
        "mouth_slope_mean": slope_mean,
        "mouth_slope_diff": slope_diff,

        # extra mouth
        "lip_thickness_mid": lip_thickness_mid,
        "mouth_corner_top_left": mouth_corner_top_left,
        "mouth_corner_top_right": mouth_corner_top_right,
        "mouth_corner_bottom_left": mouth_corner_bottom_left,
        "mouth_corner_bottom_right": mouth_corner_bottom_right,
        "mouth_corner_nose_left": mouth_corner_nose_left,
        "mouth_corner_nose_right": mouth_corner_nose_right,
        "upper_lip_curvature": upper_lip_curvature,
        "lower_lip_curvature": lower_lip_curvature,

        # eyes
        "left_eye_width": left_eye_width,
        "left_eye_height": left_eye_height,
        "right_eye_width": right_eye_width,
        "right_eye_height": right_eye_height,
        "left_ear": left_ear,
        "right_ear": right_ear,
        "mean_ear": mean_ear,
        "ear_diff": ear_diff,

        # eye-cheek
        "left_cheek_eye": left_cheek_eye,
        "right_cheek_eye": right_cheek_eye,
        "cheek_eye_mean": cheek_eye_mean,
        "cheek_eye_diff": cheek_eye_diff,

        # brows
        "left_brow_eye": left_brow_eye,
        "right_brow_eye": right_brow_eye,
        "brow_eye_mean": brow_eye_mean,
        "brow_eye_diff": brow_eye_diff,

        # brow geometry
        "inner_brow_dist": inner_brow_dist,
        "left_brow_tilt": left_brow_tilt,
        "right_brow_tilt": right_brow_tilt,
        "brow_tilt_diff": brow_tilt_diff,
        "left_brow_curv": left_brow_curv,
        "right_brow_curv": right_brow_curv,
        "brow_curv_mean": brow_curv_mean,
        "brow_curv_diff": brow_curv_diff,
        "left_inner_brow_eye": left_inner_brow_eye,
        "left_outer_brow_eye": left_outer_brow_eye,
        "right_inner_brow_eye": right_inner_brow_eye,
        "right_outer_brow_eye": right_outer_brow_eye,
        "inner_brow_eye_mean": inner_brow_eye_mean,
        "outer_brow_eye_mean": outer_brow_eye_mean,
        "inner_outer_brow_eye_diff": inner_outer_brow_eye_diff,

        # global
        "face_aspect": face_aspect,
        "head_tilt_angle": head_tilt_angle,
        "head_tilt_sin": head_tilt_sin,
        "head_tilt_cos": head_tilt_cos,
    }

    features, feature_names = make_feature_array(feature_dict)
    return features, feature_names
