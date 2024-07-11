import torch
import torch.nn as nn
import torchdiffeq
import robotsystem

factory_kwargs = {'device': 'cuda', 'dtype': torch.float32}

state = torch.randn(5, 6, **factory_kwargs)
control = torch.randn(5, 3, **factory_kwargs)

model = robotsystem.MecanumSystemModel(control, **factory_kwargs)

new_state = model(0, state)

print(state, control, new_state, sep="\n", end="\n\n")
print(torchdiffeq.odeint_adjoint(model, state, torch.linspace(0, robotsystem.delta_t, 2, **factory_kwargs)))
