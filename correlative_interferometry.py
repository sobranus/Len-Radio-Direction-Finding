import pygame
import numpy as np
import time
import math

# ---------------------------
# Window Config
# ---------------------------
WIDTH, HEIGHT = 900, 650
GRAPH_HEIGHT = 250
MAIN_HEIGHT = HEIGHT - GRAPH_HEIGHT

# ---------------------------
# Signal Parameters
# ---------------------------
C = 200.0        # wave propagation speed (pixels/s)
FREQ = 2.0       # Hz
AMPLITUDE = 1.0
SAMPLES = WIDTH
WAVELENGTH = C / FREQ

# ---------------------------
# Setup
# ---------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 26)

# Radio source
SOURCE_POS = np.array([200.0, MAIN_HEIGHT / 2])

# ---------------------------
# Pentagon detector positions
# ---------------------------
center_x = WIDTH * 0.70
center_y = MAIN_HEIGHT * 0.55
RADIUS = 50
N = 5

DETECTORS = []
for i in range(N):
    angle = -90 + i * (360 / N)
    rad = math.radians(angle)
    x = center_x + RADIUS * math.cos(rad)
    y = center_y + RADIUS * math.sin(rad)
    DETECTORS.append(np.array([x, y]))

buffers = [[0.0] * SAMPLES for _ in range(N)]

colors = [
    (255, 80, 80),   # D1 - red
    (0, 255, 0),     # D2 - green
    (255, 255, 0),   # D3 - yellow
    (0, 255, 255),   # D4 - cyan
    (255, 0, 255)    # D5 - magenta
]

labels = ["D1", "D2", "D3", "D4", "D5"]

start_time = time.time()


def received_signal(dist, t):
    delay = dist / C
    return AMPLITUDE * np.sin(2*np.pi * FREQ * (t - delay))


running = True

# ---------------------------
# Main Loop
# ---------------------------
while running:
    dt = clock.tick(60) / 1000.0
    t = time.time() - start_time

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = pygame.mouse.get_pos()
            if my < MAIN_HEIGHT:
                SOURCE_POS = np.array([mx, my], dtype=float)

    # Distances
    distances = [np.linalg.norm(SOURCE_POS - d) for d in DETECTORS]

    # Signals
    signals = [received_signal(dist, t) for dist in distances]

    # Update buffers
    for i in range(N):
        buffers[i].pop(0)
        buffers[i].append(signals[i])

    # ---------------------------
    # PHASE DIFFERENCE CALC
    # ---------------------------
    dist_ref = distances[0]                # Detector 1 reference
    phase_diffs = []

    for dist in distances:
        delta_dist = dist - dist_ref
        phase = (delta_dist / WAVELENGTH) * 2 * math.pi
        phase_diffs.append(phase)

    # ---------------------------
    # DRAW MAIN SCREEN
    # ---------------------------
    screen.fill((25, 25, 25))

    pygame.draw.circle(screen, (255, 100, 100), SOURCE_POS.astype(int), 10)

    for d in DETECTORS:
        pygame.draw.circle(screen, (80, 150, 255), d.astype(int), 10)

    text = font.render("Pentagon Detector Array | Real-time Signals + Phase Differences",
                       True, (220, 220, 220))
    screen.blit(text, (20, 20))

    # ---------------------------
    # GRAPH BACKGROUND
    # ---------------------------
    graph_rect = pygame.Rect(0, MAIN_HEIGHT, WIDTH, GRAPH_HEIGHT)
    pygame.draw.rect(screen, (30, 30, 30), graph_rect)

    for x in range(0, WIDTH, 50):
        pygame.draw.line(screen, (50, 50, 50), (x, MAIN_HEIGHT), (x, HEIGHT))
    for y in range(MAIN_HEIGHT, HEIGHT, 50):
        pygame.draw.line(screen, (50, 50, 50), (0, y), (WIDTH, y))

    midline = MAIN_HEIGHT + GRAPH_HEIGHT // 2
    pygame.draw.line(screen, (100, 100, 100), (0, midline), (WIDTH, midline))

    # Plot signals
    for i in range(N):
        pts = []
        buf = buffers[i]
        for x, v in enumerate(buf):
            y = midline - int(v * (GRAPH_HEIGHT / 2 - 10))
            pts.append((x, y))
        pygame.draw.lines(screen, colors[i], False, pts, 2)

    # ---------------------------
    # LEGEND WITH PHASE DIFFERENCES
    # ---------------------------
    for i in range(N):
        label_text = f"{labels[i]}   ({phase_diffs[i]:+.3f} rad)"
        screen.blit(font.render(label_text, True, colors[i]),
                    (20, MAIN_HEIGHT + 10 + i * 25))

    pygame.display.flip()

pygame.quit()
