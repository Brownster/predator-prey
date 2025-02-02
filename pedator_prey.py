import pygame
import random
import math
import numpy as np

# ----------------------------
# Simulation Parameters
# ----------------------------
WIDTH, HEIGHT = 800, 600
FPS = 30

# Colors
PREDATOR_COLOR = (255, 0, 0)    # Red
PREY_COLOR = (0, 255, 0)        # Green
PLANT_COLOR = (0, 128, 0)       # Dark Green
BACKGROUND_COLOR = (0, 0, 0)    # Black

# Agent sizes and speeds
PREDATOR_SIZE, PREY_SIZE, PLANT_SIZE = 10, 8, 6
PREDATOR_SPEED, PREY_SPEED = 2, 1.5

# Energy and reproduction parameters
ENERGY_LOSS_RATE = 0.1
REPRODUCTION_ENERGY = 20
LIFESPAN_RANDOMNESS = 50

# Q-learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 1.0
EXPLORATION_DECAY = 0.995
MIN_EXPLORATION_RATE = 0.01

# ----------------------------
# Spatial Hashing Parameters
# ----------------------------
CELL_SIZE = 50  # Each grid cell is CELL_SIZE x CELL_SIZE
GRID_WIDTH = WIDTH // CELL_SIZE + 1
GRID_HEIGHT = HEIGHT // CELL_SIZE + 1

# Initialize the grid as a global variable
grid = [[[] for _ in range(GRID_HEIGHT)] for _ in range(GRID_WIDTH)]

def get_cell(x, y):
    """Convert screen coordinates to grid cell indices."""
    return (int(x // CELL_SIZE), int(y // CELL_SIZE))

def update_grid(agents):
    """Clear and update the grid with the current positions of agents."""
    global grid
    grid = [[[] for _ in range(GRID_HEIGHT)] for _ in range(GRID_WIDTH)]
    for agent in agents:
        cell_x, cell_y = get_cell(agent.x, agent.y)
        if 0 <= cell_x < GRID_WIDTH and 0 <= cell_y < GRID_HEIGHT:
            grid[cell_x][cell_y].append(agent)

def get_nearby_agents(agent, agent_type):
    """Return a list of agents of a specific type from the current and adjacent cells."""
    nearby_agents = []
    cell_x, cell_y = get_cell(agent.x, agent.y)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            x = cell_x + dx
            y = cell_y + dy
            if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                nearby_agents.extend([a for a in grid[x][y] if isinstance(a, agent_type)])
    return nearby_agents

# ----------------------------
# Agent Classes
# ----------------------------
class Agent:
    def __init__(self, x, y, color, size, speed, is_predator=False):
        self.x = x
        self.y = y
        self.color = color
        self.size = size
        self.speed = speed
        self.energy = 100
        self.age = 0
        self.lifespan = random.randint(200, 200 + LIFESPAN_RANDOMNESS)
        self.is_predator = is_predator

    def move(self, direction=None):
        """Move in one of four directions; if no action provided, move randomly."""
        if direction == "up":
            self.y -= self.speed
        elif direction == "down":
            self.y += self.speed
        elif direction == "left":
            self.x -= self.speed
        elif direction == "right":
            self.x += self.speed
        else:
            self.x += random.choice([-self.speed, self.speed])
            self.y += random.choice([-self.speed, self.speed])
        # Keep within bounds
        self.x = max(0, min(WIDTH, self.x))
        self.y = max(0, min(HEIGHT, self.y))
        self.energy -= ENERGY_LOSS_RATE
        self.age += 1

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.size)

    def is_dead(self):
        return self.energy <= 0 or self.age >= self.lifespan

    def should_reproduce(self):
        # Adaptive reproduction probability based on energy and age
        reproduction_prob = min(1.0, self.energy / 100 + self.age / self.lifespan)
        return random.random() < reproduction_prob

    def reproduce(self):
        if self.energy >= REPRODUCTION_ENERGY:
            self.energy /= 2
            offspring = type(self)(
                self.x + random.randint(-20, 20),
                self.y + random.randint(-20, 20)
            )
            # Apply a small mutation in speed
            offspring.speed += random.uniform(-0.2, 0.2)
            # Mutate Q-table if it exists (for learning agents)
            if hasattr(self, 'q_table'):
                offspring.q_table = {state: {action: val + random.uniform(-0.1, 0.1)
                                             for action, val in actions.items()}
                                     for state, actions in self.q_table.items()}
            return offspring
        return None

class Predator(Agent):
    def __init__(self, x, y):
        super().__init__(x, y, PREDATOR_COLOR, PREDATOR_SIZE, PREDATOR_SPEED, is_predator=True)
        self.q_table = {}  # Dictionary for Q-values
        self.exploration_rate = EXPLORATION_RATE

    def get_state(self, prey_list):
        if not prey_list:
            return None
        # Choose nearest prey from the provided list (using spatial hashing)
        nearest_prey = min(prey_list, key=lambda p: math.hypot(p.x - self.x, p.y - self.y))
        dx = nearest_prey.x - self.x
        dy = nearest_prey.y - self.y
        distance = math.hypot(dx, dy)
        # Adaptive binning: fine when close (<50 px), coarser otherwise.
        if distance < 50:
            state = (round(dx / 10), round(dy / 10))
        else:
            state = (round(dx / 50), round(dy / 50))
        return state

    def choose_action(self, state):
        actions = ["up", "down", "left", "right"]
        if random.random() < self.exploration_rate:
            return random.choice(actions)
        else:
            if state in self.q_table:
                return max(self.q_table[state], key=self.q_table[state].get)
            else:
                return random.choice(actions)

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in actions_list()}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in actions_list()}
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())
        new_value = old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max - old_value)
        self.q_table[state][action] = new_value

class Prey(Agent):
    def __init__(self, x, y):
        super().__init__(x, y, PREY_COLOR, PREY_SIZE, PREY_SPEED)
        self.q_table = {}  # Q-values for evasion strategy
        self.exploration_rate = EXPLORATION_RATE

    def get_state(self, predators):
        if not predators:
            return None
        # Find the nearest predator among those provided by spatial hash
        nearest_predator = min(predators, key=lambda pr: math.hypot(pr.x - self.x, pr.y - self.y))
        dx = nearest_predator.x - self.x
        dy = nearest_predator.y - self.y
        distance = math.hypot(dx, dy)
        # Adaptive binning for prey as well
        if distance < 50:
            state = (round(dx / 10), round(dy / 10))
        else:
            state = (round(dx / 50), round(dy / 50))
        return state

    def choose_action(self, state):
        actions = ["up", "down", "left", "right"]
        if random.random() < self.exploration_rate:
            return random.choice(actions)
        else:
            if state in self.q_table:
                return max(self.q_table[state], key=self.q_table[state].get)
            else:
                return random.choice(actions)

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in actions_list()}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in actions_list()}
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())
        new_value = old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max - old_value)
        self.q_table[state][action] = new_value

class Plant(Agent):
    def __init__(self, x, y):
        super().__init__(x, y, PLANT_COLOR, PLANT_SIZE, 0)
        self.last_reproduced = pygame.time.get_ticks()

    def reproduce(self, plants):
        if pygame.time.get_ticks() - self.last_reproduced > 500:
            self.last_reproduced = pygame.time.get_ticks()
            return Plant(self.x + random.randint(-20, 20), self.y + random.randint(-20, 20))
        return None

def actions_list():
    """Helper function to provide the list of possible actions."""
    return ["up", "down", "left", "right"]

# ----------------------------
# Create initial populations
# ----------------------------
predators = [Predator(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(5)]
prey = [Prey(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(20)]
plants = [Plant(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(50)]

# ----------------------------
# Main Simulation Loop
# ----------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Predator-Prey Simulation with Spatial Hashing")
clock = pygame.time.Clock()

running = True
while running:
    clock.tick(FPS)
    screen.fill(BACKGROUND_COLOR)

    # Process events (e.g., quit)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update the spatial grid with current positions of all agents
    update_grid(predators + prey + plants)

    # ----------------------------
    # Update Predators
    # ----------------------------
    for predator in predators[:]:
        nearby_prey = get_nearby_agents(predator, Prey)
        state = predator.get_state(nearby_prey)
        if state:
            action = predator.choose_action(state)
            predator.move(action)
            reward = -1  # default penalty
            # Check collision with nearby prey (only in nearby cells)
            for p in nearby_prey:
                if math.hypot(p.x - predator.x, p.y - predator.y) < predator.size:
                    predator.energy += 20
                    if p in prey:
                        prey.remove(p)
                    reward = 10  # reward for catching prey
                    break
            next_state = predator.get_state(nearby_prey)
            predator.update_q_table(state, action, reward, next_state)
            predator.exploration_rate = max(MIN_EXPLORATION_RATE, predator.exploration_rate * EXPLORATION_DECAY)
        else:
            predator.move()
        predator.draw(screen)
        if predator.is_dead():
            predators.remove(predator)
        if predator.should_reproduce():
            new_pred = predator.reproduce()
            if new_pred:
                predators.append(new_pred)

    # ----------------------------
    # Update Prey
    # ----------------------------
    for p in prey[:]:
        nearby_predators = get_nearby_agents(p, Predator)
        state = p.get_state(nearby_predators)
        if state:
            action = p.choose_action(state)
            p.move(action)
            reward = 1  # reward for surviving
            # Check if any nearby predator catches the prey
            for predator in nearby_predators:
                if math.hypot(predator.x - p.x, predator.y - p.y) < predator.size:
                    p.energy = 0  # prey dies
                    reward = -10
                    break
            next_state = p.get_state(nearby_predators)
            p.update_q_table(state, action, reward, next_state)
            p.exploration_rate = max(MIN_EXPLORATION_RATE, p.exploration_rate * EXPLORATION_DECAY)
        else:
            p.move()
        p.draw(screen)
        if p.is_dead():
            prey.remove(p)
        if p.should_reproduce():
            new_prey = p.reproduce()
            if new_prey:
                prey.append(new_prey)

    # ----------------------------
    # Update Plants
    # ----------------------------
    for plant in plants[:]:
        plant.draw(screen)
        new_plant = plant.reproduce(plants)
        if new_plant:
            plants.append(new_plant)
    if random.random() < 0.02:
        plants.append(Plant(random.randint(0, WIDTH), random.randint(0, HEIGHT)))

    # ----------------------------
    # Display Simulation Stats
    # ----------------------------
    font = pygame.font.SysFont("Arial", 18)
    stats = f"Predators: {len(predators)} | Prey: {len(prey)} | Plants: {len(plants)} | Expl: {predators[0].exploration_rate:.2f}" if predators else ""
    text = font.render(stats, True, (255, 255, 255))
    screen.blit(text, (10, 10))

    pygame.display.flip()

pygame.quit()
