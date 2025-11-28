import pygame
import numpy as np
import time
import math
from screeninfo import get_monitors

# ---------------------------
# Window Config
# ---------------------------
monitor = get_monitors()[0]   # first monitor
WIDTH, HEIGHT = monitor.width, monitor.height
GRAPH_HEIGHT = 250
MAIN_HEIGHT = HEIGHT - GRAPH_HEIGHT

# ---------------------------
# Signal Parameters
# ---------------------------
C = 40.0        # wave propagation speed (pixels/s)
FREQ = 0.5       # Hz
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
center_y = MAIN_HEIGHT * 0.50
RADIUS = 30
N = 5

DETECTORS = []
dataset = []

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

def add_noise(phases, aoa,
              sigma_phase=0.1,
              amp_noise=0.05,
              bias_range=0.1,
              sigma_aoa=np.deg2rad(2)):

    # 1. Random Gaussian phase noise
    phases = phases + np.random.normal(0, sigma_phase, size=5)

    # 2. Global drift
    phases += np.random.uniform(-0.01, 0.01)

    # 3. Hardware phase biases
    biases = np.random.uniform(-bias_range, bias_range, size=5)
    phases += biases

    # 4. Amp noise => reconstruct phase
    I = np.cos(phases) + np.random.normal(0, amp_noise, size=5)
    Q = np.sin(phases) + np.random.normal(0, amp_noise, size=5)
    phases = np.arctan2(Q, I)

    # 5. Multipath nonlinear distortion
    A = np.random.uniform(0.02, 0.1)
    theta = np.random.uniform(0, 2*np.pi)
    phases += A * np.sin(2 * phases + theta)

    # 6. AoA label noise
    aoa = aoa + np.random.normal(0, sigma_aoa)

    return phases, aoa

def received_signal(dist, t):
    delay = dist / C
    return AMPLITUDE * np.sin(2*np.pi * FREQ * (t - delay))

def get_angle(p1, p2):
    # ---- Compute angle ----
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    angle_rad = math.atan2(-dy, dx)            # negative dy (pygame y-axis is flipped)
    angle_deg = (math.degrees(angle_rad) + 360) % 360
    
    return angle_rad, angle_deg

angle_rad, angle_deg = None, None
dataset = []
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
            
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
                
            if event.key == pygame.K_s:
                data_sample = []
                for phase in phase_diffs:
                    data_sample.append(float(phase))
                data_sample.append(angle_rad)
                dataset.append(data_sample)
                print(f"saved: {data_sample}, total: {len(dataset)}")
            
            if event.key == pygame.K_k:
                dataset = np.array(dataset)
                np.save("df_dataset.npy", dataset)
                print("Dataset saved! Total samples:", len(dataset))
                
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = pygame.mouse.get_pos()
            if my < MAIN_HEIGHT:
                SOURCE_POS = np.array([mx, my], dtype=float)
                
    phase_diffs = []

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

    for dist in distances:
        delta_dist = dist - dist_ref
        phase = (delta_dist / WAVELENGTH) * 2 * math.pi
        phase_diffs.append(phase)

    # ---------------------------
    # DRAW MAIN SCREEN
    # ---------------------------
    screen.fill((25, 25, 25))

    pygame.draw.circle(screen, (255, 100, 100), SOURCE_POS.astype(int), 5)

    for d in DETECTORS:
        pygame.draw.circle(screen, (80, 150, 255), d.astype(int), 5)

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
    angle_rad, angle_deg = get_angle([center_x, center_y], SOURCE_POS)
    phase_diffs, angle_rad = add_noise(phase_diffs, angle_rad)
    
    # ---- Draw line ----
    x_aoa = center_x + math.cos(angle_rad) * 2000
    y_aoa = center_y - math.sin(angle_rad) * 2000
    pygame.draw.line(screen, (255, 255, 255), [center_x, center_y], [x_aoa, y_aoa], 2)

    # ---- Draw angle text ON the line ----
    mid_x = (center_x + SOURCE_POS[0]) / 2
    mid_y = (center_y + SOURCE_POS[1]) / 2
    text = font.render(f"{angle_rad:.1f} rad", True, (255, 255, 0))
    screen.blit(text, (mid_x, mid_y))
    
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
