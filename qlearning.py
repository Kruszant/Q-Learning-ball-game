import numpy as np
import pygame
import time
import sys
import os
from collections import deque
import matplotlib.pyplot as plt  # <--- ADDED: For plotting graphs

# ====================================================================
# A. LOGIC & MAP (Random Generator)
# ====================================================================

DIMENSIONS = (3, 6, 6)  # (Layers, Rows, Columns)
START_POS = (0, 0, 0)
GOAL_POS = (2, 5, 5)

ACTIONS = {
    0: (0, -1, 0),  # North
    1: (0, 1, 0),  # South
    2: (0, 0, -1),  # West
    3: (0, 0, 1),  # East
    4: (1, 0, 0),  # Up (Layer +1)
    5: (-1, 0, 0)  # Down (Layer -1)
}
NUM_ACTIONS = len(ACTIONS)


def check_path_exists(test_map):
    queue = deque([START_POS])
    visited = set([START_POS])
    target = GOAL_POS

    while queue:
        z, r, c = queue.popleft()
        if (z, r, c) == target: return True
        for _, (dz, dr, dc) in ACTIONS.items():
            nz, nr, nc = z + dz, r + dr, c + dc
            if (0 <= nz < DIMENSIONS[0] and 0 <= nr < DIMENSIONS[1] and 0 <= nc < DIMENSIONS[2]):
                if test_map[nz, nr, nc] != 1 and (nz, nr, nc) not in visited:
                    visited.add((nz, nr, nc))
                    queue.append((nz, nr, nc))
    return False


def generate_random_map():
    attempt = 0
    while True:
        attempt += 1
        new_map = np.random.choice([0, 1], size=DIMENSIONS, p=[0.75, 0.25])
        new_map[START_POS] = 0
        new_map[GOAL_POS] = 2
        if check_path_exists(new_map):
            print(f"Map generated successfully on attempt {attempt}.")
            return new_map


MAP = generate_random_map()
MAP_SIZE = MAP.shape
NUM_STATES = MAP_SIZE[0] * MAP_SIZE[1] * MAP_SIZE[2]


def pos_to_state(pos):
    z, r, c = pos
    return z * (MAP_SIZE[1] * MAP_SIZE[2]) + r * MAP_SIZE[2] + c


def state_to_pos(state):
    layer_size = MAP_SIZE[1] * MAP_SIZE[2]
    row_size = MAP_SIZE[2]
    z = state // layer_size
    r = (state % layer_size) // row_size
    c = state % row_size
    return (z, r, c)


def get_reward(pos):
    z, r, c = pos
    val = MAP[z, r, c]
    if val == 2:
        return 100  # Goal!
    elif val == 1:
        return -10  # Hit a wall
    else:
        return -1  # Step cost


def next_state(current_state, action):
    z_start, r_start, c_start = state_to_pos(current_state)
    dz, dr, dc = ACTIONS[action]
    z_new, r_new, c_new = z_start + dz, r_start + dr, c_start + dc

    if (0 <= z_new < MAP_SIZE[0] and
            0 <= r_new < MAP_SIZE[1] and
            0 <= c_new < MAP_SIZE[2]):
        pos_new = (z_new, r_new, c_new)
        if MAP[z_new, r_new, c_new] == 1:
            return current_state, get_reward(pos_new)  # Hit wall, stay put
        return pos_to_state(pos_new), get_reward(pos_new)
    else:
        return current_state, -1  # Out of bounds, stay put


# ====================================================================
# B. Q-LEARNING AGENT (IMPROVED with Analytics)
# ====================================================================

ALPHA = 0.1
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_DECAY = 0.9995  # Slightly slower decay for better exploration
NUM_EPISODES = 10000

Q_TABLE = np.zeros((NUM_STATES, NUM_ACTIONS))


def choose_action(state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(NUM_ACTIONS)
    else:
        return np.argmax(Q_TABLE[state, :])


def plot_learning_curve(rewards, steps):
    """
    ADDED FUNCTION: Visualizes the training process
    """
    window_size = 100
    avg_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
    avg_steps = np.convolve(steps, np.ones(window_size) / window_size, mode='valid')

    plt.figure(figsize=(12, 5))

    # Plot 1: Rewards
    plt.subplot(1, 2, 1)
    plt.plot(avg_rewards, label='Moving Avg (100 eps)')
    plt.title("Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.legend()

    # Plot 2: Steps
    plt.subplot(1, 2, 2)
    plt.plot(avg_steps, color='orange', label='Moving Avg (100 eps)')
    plt.title("Steps per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps to Goal")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    print("Close the graph window to continue to the simulation...")
    plt.show()


def train_agent():
    global Q_TABLE
    epsilon = EPSILON_START
    goal_state = pos_to_state(GOAL_POS)

    # --- ANALYTICS STORAGE ---
    history_rewards = []
    history_steps = []

    print(f"--- START TRAINING ({NUM_EPISODES} episodes) ---")

    for episode in range(NUM_EPISODES):
        current_state = pos_to_state(START_POS)
        epsilon = max(0.01, epsilon * EPSILON_DECAY)
        steps = 0
        total_reward = 0  # Track reward for this episode

        while current_state != goal_state and steps < 200:
            steps += 1
            action = choose_action(current_state, epsilon)
            next_s, reward = next_state(current_state, action)

            # Bellman Equation Update
            max_q_next = np.max(Q_TABLE[next_s, :])
            Q_TABLE[current_state, action] = (1 - ALPHA) * Q_TABLE[current_state, action] + \
                                             ALPHA * (reward + GAMMA * max_q_next)

            current_state = next_s
            total_reward += reward

        # Save metrics
        history_rewards.append(total_reward)
        history_steps.append(steps)

        if episode % 1000 == 0:
            avg_r = np.mean(history_rewards[-1000:])
            print(f"Episode {episode}: Epsilon={epsilon:.4f} | Avg Reward (last 1k): {avg_r:.2f}")

    print("--- TRAINING FINISHED ---")

    # --- SHOW ANALYTICS ---
    plot_learning_curve(history_rewards, history_steps)


# ====================================================================
# C. VISUALIZATION (Pygame)
# ====================================================================

TILE_SIZE = 60
MARGIN = 20
PANEL_GAP = 50

PANEL_WIDTH = DIMENSIONS[2] * TILE_SIZE
PANEL_HEIGHT = DIMENSIONS[1] * TILE_SIZE

WINDOW_WIDTH = (3 * PANEL_WIDTH) + (2 * PANEL_GAP) + (2 * MARGIN)
WINDOW_HEIGHT = PANEL_HEIGHT + 150

BG_COLOR = (30, 30, 30)
C_GRID = (50, 50, 50)
C_WALL = (80, 80, 80)
C_FREE = (200, 200, 200)
C_START = (50, 50, 200)
C_GOAL = (50, 200, 50)
C_AGENT = (255, 100, 0)
C_ACTIVE_BORDER = (255, 255, 0)
C_INACTIVE_OVERLAY = (0, 0, 0, 150)

os.environ['SDL_AUDIODRIVER'] = 'dsp'  # Fix for some systems, can be removed if audio errors occur


def draw_panel(z_index, offset_x, offset_y, is_active, agent_pos_local):
    """Draws a single layer (2D grid) at a specific position"""

    title = f"LAYER {z_index}"
    text_color = (255, 255, 255) if is_active else (100, 100, 100)
    img_text = FONT_BIG.render(title, True, text_color)
    text_rect = img_text.get_rect(center=(offset_x + PANEL_WIDTH // 2, offset_y - 30))
    SCREEN.blit(img_text, text_rect)

    rect_panel = pygame.Rect(offset_x - 5, offset_y - 5, PANEL_WIDTH + 10, PANEL_HEIGHT + 10)
    border_color = C_ACTIVE_BORDER if is_active else (50, 50, 50)
    pygame.draw.rect(SCREEN, border_color, rect_panel, 3)

    for r in range(DIMENSIONS[1]):
        for c in range(DIMENSIONS[2]):
            x = offset_x + c * TILE_SIZE
            y = offset_y + r * TILE_SIZE
            rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)

            cell_type = MAP[z_index, r, c]

            if cell_type == 1:
                color = C_WALL
            elif (z_index, r, c) == START_POS:
                color = C_START
            elif (z_index, r, c) == GOAL_POS:
                color = C_GOAL
            else:
                color = C_FREE

            pygame.draw.rect(SCREEN, color, rect)
            pygame.draw.rect(SCREEN, C_GRID, rect, 1)

            if is_active and (r, c) == agent_pos_local:
                pygame.draw.circle(SCREEN, C_AGENT, rect.center, TILE_SIZE // 2.5)
                pygame.draw.circle(SCREEN, (255, 255, 255), (rect.centerx - 5, rect.centery - 5), 5)

    if not is_active:
        s = pygame.Surface((PANEL_WIDTH, PANEL_HEIGHT), pygame.SRCALPHA)
        s.fill(C_INACTIVE_OVERLAY)
        SCREEN.blit(s, (offset_x, offset_y))


def draw_scene(agent_pos):
    SCREEN.fill(BG_COLOR)

    agent_z, agent_r, agent_c = agent_pos
    info = f"Pos: (L:{agent_z}, R:{agent_r}, C:{agent_c}) | Goal: {GOAL_POS}"
    SCREEN.blit(FONT.render(info, True, (200, 200, 200)), (20, WINDOW_HEIGHT - 40))

    for z in range(DIMENSIONS[0]):
        offset_x = MARGIN + (z * (PANEL_WIDTH + PANEL_GAP))
        offset_y = 80
        is_active = (z == agent_z)
        draw_panel(z, offset_x, offset_y, is_active, (agent_r, agent_c))

    pygame.display.flip()


def test_agent():
    # Initialize Pygame only when needed for visualization
    pygame.init()
    pygame.font.init()
    global FONT, FONT_BIG, SCREEN
    FONT = pygame.font.SysFont('Arial', 20)
    FONT_BIG = pygame.font.SysFont('Arial', 28, bold=True)
    SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("3D Drone Navigation - Analysis Mode")

    current_state = pos_to_state(START_POS)
    goal_state = pos_to_state(GOAL_POS)

    print("\n--- Testing Mode (Watch the agent) ---")

    running = True
    while running and current_state != goal_state:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                sys.exit()

        pos_3d = state_to_pos(current_state)
        draw_scene(pos_3d)

        optimal_action = np.argmax(Q_TABLE[current_state, :])
        next_s, _ = next_state(current_state, optimal_action)

        current_state = next_s
        time.sleep(0.5)

    if current_state == goal_state:
        draw_scene(state_to_pos(goal_state))
        print("SUCCESS! Target reached.")
        time.sleep(3)

    pygame.quit()


if __name__ == "__main__":
    train_agent()  # Computes logic & shows graph
    test_agent()  # Shows Pygame animation