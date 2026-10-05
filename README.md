# ESBN-iCub

Minimal implementation of the nonlinear efficient balanced spiking network from
Alemi et al. (2018), trained directly on the seven-DOF right-arm dynamics of the
official iCubGenova11 model.

## What this repository does

1. Load the official iCubGenova11 URDF.
2. Reduce it to the seven right-arm joints with Pinocchio.
3. Generate smooth torque-driven arm trajectories.
4. Train the paper EBN online with local slow-weight plasticity.
5. Disable teacher feedback and learning.
6. Run the trained spiking network on an unseen iCub trajectory.
7. Save simulator-vs-network joint trajectories, error, and spike raster.

The learned state is

```
x = [q, qdot, tau]
```

with torque dynamics

```
tau_dot = -alpha*tau + c_tau(t)
```

so the teacher remains in the paper's additive-input form

```
x_dot = f(x) + c(t).
```

## Install

```bash
pip install -e ".[robot,plot]"
```

## Run

```bash
python -m esbn_icub.paper_on_icub
```

Outputs are written to:

```
results/paper_on_icub.png
results/paper_on_icub.npz
results/paper_on_icub.json
```

For a quick smoke run:

```bash
python -m esbn_icub.paper_on_icub \
  --neurons 128 \
  --train-episodes 3 \
  --train-steps 100 \
  --test-steps 100
```

## Core files

```
src/esbn_icub/paper_network.py   paper EBN
src/esbn_icub/icub_teacher.py    iCubGenova11 rigid-body teacher
src/esbn_icub/paper_on_icub.py   complete experiment
```

`models/SOURCES.md` records the exact robot-model provenance.

## Scope

This is currently the seven-DOF right arm. The articulated hand is intentionally
not included until the arm experiment is working end-to-end.
