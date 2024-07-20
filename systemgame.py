import math
import pygame
import robotsystem
import torch
import time
import torchdiffeq

device = 'cpu'
dtype = torch.double
param2 = {'device': device, 'dtype': dtype}

torch.set_num_threads(1)

max_res = 1000
field_size = torch.tensor((2.362, 1.143), **param2)
screen_resolution = torch.tensor((max_res, max_res * (field_size[1] / field_size[0])), **param2).to(dtype=torch.int)
pixels_per_meter = screen_resolution / field_size
screen_center = screen_resolution / 2

def game_to_screen(coords, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    to_screen = torch.tensor(((1,  0),
                              (0, -1)), **factory_kwargs) * pixels_per_meter
    return screen_center + (to_screen @ coords.unsqueeze(-1)).squeeze(-1)

framerate = 30
delta_t = 1 / framerate

pygame.init()
screen_surface = pygame.display.set_mode(tuple(screen_resolution.cpu().numpy()))
pygame.display.set_caption('Mecanum Simulator')

state = torch.tensor((0, 0, 0, 0, 0, 0), **param2)
control = torch.tensor((0, 0, 0), **param2)

keys = (pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d, pygame.K_LEFT, pygame.K_RIGHT)
key_behavior = torch.tensor(((1,  0,  -1,  0,  0,  0),
                             (0,  1, 0,  -1,  0,  0),
                             (0,  0,  0,  0,  1, -1)), **param2)
key = torch.full((6,), 0, **param2)

model = robotsystem.MecanumSystemModel(**param2)

rect_size = pixels_per_meter * 0.0254 * 12
rect_surface = pygame.Surface(tuple(rect_size.cpu().to(dtype=torch.int).numpy()), pygame.SRCALPHA)
rect_surface.fill((255, 0, 0))
pygame.draw.circle(rect_surface, (0, 255, 0), (rect_size[0].item(), rect_size[1].item() / 2), rect_size[0].item() / 4)

frametime = delta_t
running = True
while running:
    begin = time.time()

    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE: running = False
            else:
                try:
                    idx = keys.index(event.key)
                    key[idx] = 1
                except ValueError: pass
        elif event.type == pygame.KEYUP:
            try:
                idx = keys.index(event.key)
                key[idx] = 0
            except ValueError: pass

    control = (key_behavior @ key.unsqueeze(-1)).squeeze(-1)
    control[:2] = torch.nan_to_num(control[:2] / torch.sqrt(torch.sum(torch.square(control[:2]))))
    control /= math.sqrt(2) / 2 * 2 + 1

    model.set_control_duty(control)

    state = torchdiffeq.odeint_adjoint(model,
                                       state,
                                       torch.tensor((0, delta_t), **param2),
                                       method='rk4',
                                       options=dict(step_size=delta_t/10))[-1]
    state[2] = ((state[2] + torch.pi) % (2 * torch.pi)) - torch.pi

    c, s = torch.cos(state[2]), torch.sin(state[2])
    rot_mat = torch.tensor(((c, -s),
                            (s,  c)))
    
    geo_pos = state[:2] - (rot_mat @ robotsystem.center_of_mass.to(**param2).unsqueeze(-1)).squeeze(-1)
    
    rotated_surface = pygame.transform.rotate(rect_surface, state[2] / torch.pi * 180)
    rotated_rect = rotated_surface.get_rect(center=tuple(game_to_screen(geo_pos, **param2).cpu().to(dtype=torch.int).numpy()))

    screen_surface.fill((255, 255, 255))
    screen_surface.blit(rotated_surface, rotated_rect.topleft)

    print(frametime, control, state, sep='\n', end='\n\n')
    
    pygame.display.flip()

    frametime = time.time() - begin
    time_left = 1 / framerate - frametime
    if time_left > 0:
        time.sleep(time_left)
