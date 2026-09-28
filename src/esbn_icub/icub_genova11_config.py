"""Pinned iCubGenova11 right-arm hardware configuration.

Values below are transcribed from robotology/robots-configuration at commit
b9dee4946a53a670aeaf0b1b7a8b20196fae4332.

They are kept separate from the URDF because the physical URDF uses placeholder
effort/velocity limits. Position limits and drivetrain couplings come from the
real robot configuration; rigid-body inertias come from icub-models.
"""

from __future__ import annotations

import numpy as np

ROBOTS_CONFIGURATION_COMMIT = "b9dee4946a53a670aeaf0b1b7a8b20196fae4332"

RIGHT_ARM_JOINTS = (
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_shoulder_yaw",
    "r_elbow",
    "r_wrist_prosup",
    "r_wrist_pitch",
    "r_wrist_yaw",
)

# Hardware joint position limits, degrees, in RIGHT_ARM_JOINTS order.
# Sources:
# iCubGenova11/hardware/mechanicals/right_arm-eb3-j0_3-mec.xml
# iCubGenova11/hardware/mechanicals/right_arm-eb27-j4_7-mec.xml
HARDWARE_POSITION_MIN_DEG = np.array(
    [-95.5, 0.0, -32.0, 15.0, -90.0, -80.0, -15.0],
    dtype=float,
)
HARDWARE_POSITION_MAX_DEG = np.array(
    [8.0, 160.0, 80.0, 106.0, 90.0, 30.0, 35.0],
    dtype=float,
)
HARDWARE_POSITION_MIN = np.deg2rad(HARDWARE_POSITION_MIN_DEG)
HARDWARE_POSITION_MAX = np.deg2rad(HARDWARE_POSITION_MAX_DEG)

# Motor-to-joint gearbox values reported by the hardware configuration.
GEARBOX_M2J = np.array(
    [-100.0, -100.0, -100.0, -100.0, 100.0, 159.0, 159.0],
    dtype=float,
)

# Coupling blocks in the hardware configuration. The first block corresponds
# to shoulder pitch/roll/yaw + elbow; the second to prosup/pitch/yaw + hand.
SHOULDER_J2M = np.array(
    [
        [1.000, 0.000, 0.000, 0.000],
        [-1.625, 1.625, 0.000, 0.000],
        [0.000, 0.000, 1.625, 0.000],
        [0.000, 0.000, 0.000, 1.000],
    ],
    dtype=float,
)
SHOULDER_M2J = np.array(
    [
        [1.000, 0.000, 0.000, 0.000],
        [1.000, 0.615, 0.000, 0.000],
        [0.000, -0.615, 0.615, 0.000],
        [0.000, 0.000, 0.000, 1.000],
    ],
    dtype=float,
)
WRIST_J2M = np.array(
    [
        [1.000, 0.000, 0.000, 0.000],
        [0.000, 1.000, 0.000, 0.000],
        [0.000, -1.000, 1.000, 0.000],
        [0.000, 0.000, 0.000, 1.000],
    ],
    dtype=float,
)
WRIST_M2J = np.array(
    [
        [1.000, 0.000, 0.000, 0.000],
        [0.000, 1.000, 1.000, 0.000],
        [0.000, 0.000, 1.000, 0.000],
        [0.000, 0.000, 0.000, 1.000],
    ],
    dtype=float,
)
