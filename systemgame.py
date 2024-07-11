import pygame
import sys
import pygame.gfxdraw
import robotsystem
import torch
import time
import torchdiffeq

pygame.init()

width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Mecanum Simulator')

running = True

framerate_max = 30

factory_kwargs = {'device': 'cpu', 'dtype': torch.float32}

state = torch.tensor((0, 0, 0, 0, 0, 0), **factory_kwargs)
control = torch.tensor((0, 0, 0), **factory_kwargs)

keys = torch.tensor((0, 0, 0, 0, 0, 0), **factory_kwargs)

delta_t = 1 / framerate_max
frametime = delta_t

model = robotsystem.MecanumSystemModel(control, **factory_kwargs)
# model.compile()


rect_center = (400, 300)
rect_size = (100, 50)
angle = 0

rect_surface = pygame.Surface(rect_size, pygame.SRCALPHA)
rect_surface.fill((255, 0, 0))

key_map = (pygame.K_w, pygame.K_a, pygame.K_LEFT, pygame.K_s, pygame.K_d, pygame.K_RIGHT)

while running:
    begin = time.time()

    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE: running = False
            else:
                try:
                    idx = key_map.index(event.key)
                    if idx != -1: keys[idx] = 1
                except: pass
        elif event.type == pygame.KEYUP:
            try:
                idx = key_map.index(event.key)
                if idx != -1: keys[idx] = 0
            except: pass

    control = (keys[:3] - keys[3:])
    control[:2] = torch.nan_to_num(control[:2] / torch.sum(torch.abs(control[:2])), 0, 0, 0) * 2
    control = control / 3
    
    model.set_control_duty(control)

    state = torchdiffeq.odeint_adjoint(model, state, torch.tensor((0, delta_t), **factory_kwargs))[-1] # , method='rk4', options=dict(step_size=delta_t/10)
    state[2] %= 2 * torch.pi

    rotated_surface = pygame.transform.rotate(rect_surface, state[2] / torch.pi * 180)
    rotated_rect = rotated_surface.get_rect(center=((state[0] * (width / 2.)) + (width / 2.), (-state[1] * (width / 2.)) + (height / 2.)))

    screen.fill((255, 255, 255))
    screen.blit(rotated_surface, rotated_rect.topleft)

    print(delta_t, frametime, control, state, sep='\n', end='\n\n')
    
    pygame.display.flip()
    
    frametime = time.time() - begin
    time_left = 1 / framerate_max - frametime
    if time_left > 0:
        time.sleep(time_left)
    delta_t = time.time() - begin

pygame.quit()
sys.exit()
