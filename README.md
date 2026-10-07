# ESBN-iCub

This repository applies the efficient balanced spiking network from Alemi et al.
(2018), *Learning Nonlinear Dynamics in Efficient, Balanced Spiking Networks
Using Local Plasticity Rules*, to the seven-DOF right arm of iCubGenova11.

The implementation now follows the paper's formulation directly:

```
teacher:  x_dot = f(x) + c(t)
student:  u_dot = -lambda u + F c - W_fast s + W_slow Psi(r) + k D^T e
readout:  x_hat = D r
learning: dW_slow/dt = eta (D^T e) Psi(r)^T
test:     k_test = 0
```

For the iCub mechanical system,

```
x = [q, qdot]
c = [0, a_command]
```

so the teacher is

```
q_dot    = qdot
qddot    = qddot_passive(q, qdot) + a_command
```

This is the exact additive-input form required by the paper. The command is
therefore an additive generalized-acceleration command, not raw joint torque.

The paper's mechanical example explicitly uses both position and velocity for
a second-order system. The iCub task does the same, giving a 14-dimensional
state for the seven arm joints.

## Install

```bash
pip install -e ".[robot,plot]"
```

## Run

```bash
python -m esbn_icub.paper_on_icub
```

The default experiment uses:

```
N = 200 neurons
500 learning iterations
filtered random command input
high feedback initially, reduced through learning
k_test = 0
```

Outputs:

```
results/paper_on_icub.png
results/paper_on_icub.npz
results/paper_on_icub.json
```

## Core files

```
src/esbn_icub/paper_network.py   EBN equations and local plasticity rule
src/esbn_icub/icub_teacher.py    reduced iCubGenova11 rigid-body teacher
src/esbn_icub/paper_on_icub.py   paper-form iCub experiment
```

`models/SOURCES.md` records the exact robot-model provenance.

## Important scope note

The paper does not provide a torque-controlled iCub model. To remain faithful to
its required `x_dot=f(x)+c(t)` formulation, the iCub command is defined as an
additive generalized acceleration. Using raw torque would require a
control-affine extension because `M(q)^-1 tau` is state dependent, which would
no longer be the paper's model unchanged.
