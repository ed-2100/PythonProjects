import torch
import drivemodel
import math
import pygame
import time

device = 'cpu'
dtype = torch.float32

param1 = {'device': device}
param2 = {'device': device, 'dtype': dtype}

torch.set_num_threads(1)

model = drivemodel.MecanumDriveModel(**param2)
model.load_state_dict(torch.load('state_dict.pt'))
model.eval()

inputs_mean, inputs_std, outputs_mean, outputs_std = (i.to(**param2) for i in torch.load('scale_factors.pt'))

in_to_m = 0.0254

max_res = 1000

dim = torch.tensor((2.362, 1.143), **param2)

res = torch.tensor((max_res, max_res * (dim[1] / dim[0])), **param2).to(dtype=torch.int)

unit = res / dim

center = res / 2

def game_to_screen(coords, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    to_screen = torch.tensor(((1,  0),
                              (0, -1)), **factory_kwargs) * unit
    return center + (to_screen @ coords.unsqueeze(-1)).squeeze(-1)

framerate = 30
delta_t = 1 / framerate # initial

pygame.init()

screen_surface = pygame.display.set_mode(tuple(res.cpu().numpy()))
pygame.display.set_caption('Mecanum Simulator')

state = torch.tensor((0, 0, torch.pi / 2, 0, 0, 0), **param2) #initial
control = torch.tensor((0, 0, 0), **param2) # initial

keys = (pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d, pygame.K_LEFT, pygame.K_RIGHT)
key_behavior = torch.tensor(((0, -1,  0,  1,  0,  0),
                             (1,  0, -1,  0,  0,  0),
                             (0,  0,  0,  0,  1, -1)), **param2)
key = torch.full((6,), 0, **param2)

rect_size = unit * in_to_m * 12
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
                except: pass
        elif event.type == pygame.KEYUP:
            try:
                idx = keys.index(event.key)
                key[idx] = 0
            except: pass
    control = (key_behavior @ key.unsqueeze(-1)).squeeze(-1)
    control[:2] = torch.nan_to_num(control[:2] / torch.sqrt(torch.sum(torch.square(control[:2]))))
    control /= math.sqrt(2) / 2 * 2 + 1

    inputs = torch.cat((state[3:], control))

    c = torch.cos(-state[2])
    s = torch.sin(-state[2])
    tmat = torch.tensor(((c, -s),
                         (s,  c)))
    inputs[:2] = (tmat @ inputs[:2].unsqueeze(-1)).squeeze(-1)
    
    now = time.time()
    with torch.no_grad():
        outputs = (model.forward(((inputs - inputs_mean) / inputs_std).unsqueeze(0)).squeeze(0) * outputs_std) + outputs_mean
    print(time.time() - now)

    tmat = torch.pinverse(tmat)
    outputs[:2] = (tmat @ outputs[:2].unsqueeze(-1)).squeeze(-1)
    outputs[3:5] = (tmat @ outputs[3:5].unsqueeze(-1)).squeeze(-1)

    state += outputs
    state[2] = ((state[2] + torch.pi) % (2 * torch.pi)) - torch.pi
    
    rotated_surface = pygame.transform.rotate(rect_surface, state[2] / torch.pi * 180)
    rotated_rect = rotated_surface.get_rect(center=tuple(game_to_screen(state[:2], **param2).cpu().to(dtype=torch.int).numpy()))

    screen_surface.fill((255, 255, 255))
    screen_surface.blit(rotated_surface, rotated_rect.topleft)

    print(delta_t, frametime, control, state, sep='\n', end='\n\n')
    
    pygame.display.flip()

    frametime = time.time() - begin
    time_left = 1 / framerate - frametime
    if time_left > 0:
        time.sleep(time_left)
