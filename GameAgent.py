import pygame, sys
from pygame.locals import *
import random
from LineEnv import LineEnv
import numpy as np

pygame.init()

font = pygame.font.SysFont('Courier New', 30)
 
FPS = 60
FramePerSec = pygame.time.Clock()
 
BLUE  = (0, 0, 255)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 1000
surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
surface.fill(WHITE)
pygame.display.set_caption("Can you FLY")    
         
env = LineEnv('GameAgent')
player_pos = [(0, SCREEN_HEIGHT/2)]
x_scale, y_scale = 10, 5000

player_score = 0


ynet = -y_scale*0.015 + SCREEN_HEIGHT/2

net_history = np.load('NetAgentModels/Smith_v7.history.npy', allow_pickle = True)
net_pos = np.array(net_history[-1]['positions'])

net_scores = [env.CalculateReward(p) for p in net_pos]

net_pos[:, 1] = -y_scale*net_pos[:, 1] + SCREEN_HEIGHT/2
net_pos[:, 0] = net_pos[:, 0]*SCREEN_WIDTH/x_scale



cframe = 2
while True:     
    for event in pygame.event.get():              
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    
    accelerating = False
    keys = pygame.key.get_pressed()
    if keys[K_UP]:
        accelerating = True
    obs, reward, _ = env.Step(accelerating)
    
    if cframe < len(net_pos[:, 0]):
        player_score += reward
    xo, yo = obs[0:2]
    # Shift y to avoid negative values
    #yo += 2
    
    xo = int(xo*SCREEN_WIDTH/x_scale)
    
    # Shift y to screen coordinates
    yo = -y_scale*yo + SCREEN_HEIGHT/2
    player_pos.append((xo, yo))
         
    surface.fill(WHITE)
    
    pygame.draw.lines(surface, RED, False, net_pos[0:cframe, :], width = 3)
    
    pygame.draw.lines(surface, GREEN, False, player_pos, width = 5)
    
    pygame.draw.line(surface, BLACK, (0, SCREEN_HEIGHT/2), (SCREEN_WIDTH, SCREEN_HEIGHT/2), width = 2)
    pygame.draw.line(surface, BLACK, (0, ynet), (SCREEN_WIDTH, ynet), width = 1)
    pygame.draw.line(surface, BLACK, (0, -ynet), (SCREEN_WIDTH, -ynet), width = 1)
    
    reward_surf = font.render(f'Your score: {int(player_score)}', False, (0, 0, 0))
    surface.blit(reward_surf, (0,0))
    
    net_surf = font.render(f' Net score: {int(sum(net_scores[0:cframe]))}', False, (0, 0, 0))
    surface.blit(net_surf, (0, 25))
         
    pygame.display.update()
    FramePerSec.tick(FPS)
    
    cframe += 1
    if cframe >= len(net_pos[:, 0]):
        cframe = len(net_pos[:, 0])