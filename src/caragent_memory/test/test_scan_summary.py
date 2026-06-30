from caragent_memory.scan_summary import summarize_scan_arrays


def test_scan_summary_accepts_angle_max_from_scan_payload():
    summary = summarize_scan_arrays(
        ranges=[1.0, 2.0, 3.0],
        angle_min=0.0,
        angle_max=0.2,
        angle_increment=0.1,
        range_min=0.05,
        range_max=10.0,
    )

    assert summary["available"] is True
    assert summary["valid_count"] == 3
    assert summary["front_min_m"] == 1.0
