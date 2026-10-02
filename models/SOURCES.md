# Model sources

## Rigid-body model

The arm teacher uses the official **iCubGenova11** physical-robot URDF from
robotology/icub-models, pinned at commit:

    dca8c618c264be70ce285bad38040411725e3551

The complete URDF is loaded by Pinocchio. We do not generate a truncated arm
URDF. Instead, all joints except the seven right-arm/wrist joints are locked at
a legal reference configuration with Pinocchio's reduced-model construction.
Their rigid-body inertias are therefore retained through composite inertias.

We intentionally do not use iCubGazeboV2_7 as the dynamics source because the
upstream icub-models documentation states that some Gazebo-model inertias are
artificially increased for simulator stability.

## Physical robot configuration

Joint position limits and drivetrain metadata are taken from the official
robotology/robots-configuration repository, pinned at commit:

    b9dee4946a53a670aeaf0b1b7a8b20196fae4332

Source files:

- iCubGenova11/hardware/mechanicals/right_arm-eb3-j0_3-mec.xml
- iCubGenova11/hardware/mechanicals/right_arm-eb27-j4_7-mec.xml

The values are transcribed into src/esbn_icub/icub_genova11_config.py.

The physical URDF uses placeholder values of 50000 for several effort and
velocity limits. Those values are deliberately not used for excitation,
normalization, or initial-velocity sampling.

The current experiment still applies generalized joint torques directly. The
documented motor/joint coupling matrices are preserved for a later drivetrain
model but are not silently folded into the rigid-body baseline.

## Active coordinates

- r_shoulder_pitch
- r_shoulder_roll
- r_shoulder_yaw
- r_elbow
- r_wrist_prosup
- r_wrist_pitch
- r_wrist_yaw

## Hand status

The hand remains excluded from the dynamics baseline. Available articulated
simulation hands contain placeholder finger inertias/limits, so they will not
be mixed into the physical Genova11 model without explicit provenance and
validation.
