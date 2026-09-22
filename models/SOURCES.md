# Model sources

## Baseline

The first teacher uses the official iCubGenova11 URDF distributed by
robotology/icub-models. The upstream README identifies iCubGenova11 as an
iCub v2.7 physical-robot model.

We intentionally do not use iCubGazeboV2_7 for the dynamics baseline because
the upstream documentation states that some Gazebo-model link inertias are
increased in a non-realistic way for simulator stability.

At runtime ICubTeacher locates the installed iCubGenova11/model.urdf, resolves
its package://iCub mesh paths to local absolute paths, and loads it in PyBullet
with URDF_USE_INERTIA_FROM_FILE.

## Implemented scope

The current teacher exposes these seven right-arm/wrist coordinates:

- r_shoulder_pitch
- r_shoulder_roll
- r_shoulder_yaw
- r_elbow
- r_wrist_prosup
- r_wrist_pitch
- r_wrist_yaw

All other movable joints are held fixed during this first phase.

## Hand status

The articulated hand is not yet merged into the model. This is deliberate:
the official physical Genova11 URDF and detailed articulated-hand simulation
assets do not share identical modeling assumptions. The hand phase will add
the finger tree only after the joint mapping and inertia provenance are
documented explicitly.
