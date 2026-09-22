from ml.inference.motion_gate import mean_hand_motion


def test_zero_motion_for_a_held_still_window():
    frame = [0.1, 0.2, 0.3, 0.4]
    window = [frame] * 32
    assert mean_hand_motion(window, hand_feature_size=4) == 0.0


def test_positive_motion_for_a_moving_window():
    window = [[float(i), 0.0, 0.0, 0.0] for i in range(32)]
    assert mean_hand_motion(window, hand_feature_size=4) > 0.0


def test_ignores_pose_face_motion_outside_hand_feature_size():
    # Hand portion (first 2 dims) never changes; only the pose/face tail does.
    window = [[0.1, 0.2, float(i), float(i) * 2] for i in range(32)]
    assert mean_hand_motion(window, hand_feature_size=2) == 0.0


def test_returns_zero_for_a_too_short_window():
    assert mean_hand_motion([[0.1, 0.2]], hand_feature_size=2) == 0.0
    assert mean_hand_motion([], hand_feature_size=2) == 0.0
