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

# Agent sizes (radius) and speed boundaries (agents will have random speed within these ranges)
PREDATOR_SIZE, PREY_SIZE, PLANT_SIZE = 10, 8, 6
PREDATOR_SPEED_RANGE = (1.5, 3.0)
PREY_SPEED_RANGE = (1.0, 2.0)

# Sight and FOV boundaries (in pixels and degrees)
# Predators: typically have moderate sight and FOV.
PREDATOR_SIGHT_RANGE = (80, 150)
PREDATOR_FOV_RANGE = (60, 120)  # degrees
# Prey: generally need further sight and wider FOV.
PREY_SIGHT_RANGE = (150, 250)
PREY_FOV_RANGE = (120, 180)     # degrees

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

# Global grid initialization
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

def is_colliding(agent1, agent2):
    """Optimized collision detection using squared distance and radius sum."""
    dx = agent1.x - agent2.x
    dy = agent1.y - agent2.y
    distance_sq = dx * dx + dy * dy
    radius_sum = agent1.size + agent2.size
    return distance_sq < radius_sum * radius_sum

def angle_difference(angle1, angle2):
    """Return the absolute smallest difference between two angles (in degrees)."""
    diff = (angle2 - angle1 + 180) % 360 - 180
    return abs(diff)

def angle_to_target(source, target):
    """Compute the angle (in degrees) from source to target."""
    dx = target.x - source.x
    dy = target.y - source.y
    angle = math.degrees(math.atan2(dy, dx))
    return angle % 360

def actions_list():
    """Helper: List of possible actions."""
    return ["up", "down", "left", "right"]

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
        # Default orientation (in degrees); updated on moves.
        self.orientation = random.uniform(0, 360)

    def move(self, direction=None):
        """Move in one of four directions; update orientation accordingly."""
        if direction == "up":
            self.y -= self.speed
            self.orientation = 90
        elif direction == "down":
            self.y += self.speed
            self.orientation = 270
        elif direction == "left":
            self.x -= self.speed
            self.orientation = 180
        elif direction == "right":
            self.x += self.speed
            self.orientation = 0
        else:
            # Random move if no direction specified.
            direction = random.choice(actions_list())
            self.move(direction)
            return  # Already handled in recursive call.
        # Keep within screen bounds.
        self.x = max(0, min(WIDTH, self.x))
        self.y = max(0, min(HEIGHT, self.y))
        self.energy -= ENERGY_LOSS_RATE
        self.age += 1

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.size)

    def is_dead(self):
        return self.energy <= 0 or self.age >= self.lifespan

    def should_reproduce(self):
        reproduction_prob = min(1.0, self.energy / 100 + self.age / self.lifespan)
        return random.random() < reproduction_prob

    def reproduce(self):
        if self.energy >= REPRODUCTION_ENERGY:
            self.energy /= 2
            offspring = type(self)(
                self.x + random.randint(-20, 20),
                self.y + random.randint(-20, 20)
            )
            # Mutate speed, sight_range, and fov if they exist.
            offspring.speed = mutate_value(self.speed, -0.2, 0.2, self.__class__.speed_bounds)
            if hasattr(self, 'sight_range'):
                offspring.sight_range = mutate_value(self.sight_range, -5, 5, self.__class__.sight_bounds)
            if hasattr(self, 'fov'):
                offspring.fov = mutate_value(self.fov, -5, 5, self.__class__.fov_bounds)
            if hasattr(self, 'q_table'):
                offspring.q_table = {state: {action: val + random.uniform(-0.1, 0.1)
                                             for action, val in actions.items()}
                                     for state, actions in self.q_table.items()}
            # Inherit orientation from parent
            offspring.orientation = self.orientation
            return offspring
        return None

def mutate_value(value, delta_min, delta_max, bounds):
    """Mutate a value by a random delta, keeping it within specified bounds (min, max)."""
    new_value = value + random.uniform(delta_min, delta_max)
    return max(bounds[0], min(bounds[1], new_value))

class Predator(Agent):
    # Define class-level bounds for speed, sight_range, and fov.
    speed_bounds = PREDATOR_SPEED_RANGE
    sight_bounds = PREDATOR_SIGHT_RANGE
    fov_bounds = PREDATOR_FOV_RANGE

    def __init__(self, x, y):
        speed = random.uniform(*Predator.speed_bounds)
        super().__init__(x, y, PREDATOR_COLOR, PREDATOR_SIZE, speed, is_predator=True)
        # Initialize Q-learning table and exploration rate.
        self.q_table = {}
        self.exploration_rate = EXPLORATION_RATE
        # Set random sight range and field-of-view (in degrees)
        self.sight_range = random.uniform(*Predator.sight_bounds)
        self.fov = random.uniform(*Predator.fov_bounds)

    def get_state(self, prey_list):
        # Filter prey that are within sight range and within field-of-view.
        visible_prey = []
        for p in prey_list:
            dx = p.x - self.x
            dy = p.y - self.y
            distance = math.hypot(dx, dy)
            if distance <= self.sight_range:
                # Compute angle to prey and compare with orientation.
                angle_to_p = angle_to_target(self, p)
                if angle_difference(self.orientation, angle_to_p) <= self.fov / 2:
                    visible_prey.append(p)
        if not visible_prey:
            return None
        # Choose the nearest visible prey.
        nearest_prey = min(visible_prey, key=lambda p: math.hypot(p.x - self.x, p.y - self.y))
        dx = nearest_prey.x - self.x
        dy = nearest_prey.y - self.y
        distance = math.hypot(dx, dy)
        # Adaptive binning: fine when close (< sight_range/2), coarser otherwise.
        if distance < self.sight_range / 2:
            state = (round(dx / 10), round(dy / 10))
        else:
            state = (round(dx / 50), round(dy / 50))
        return state

    def choose_action(self, state):
        actions = actions_list()
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
    # Define class-level bounds for speed, sight_range, and fov.
    speed_bounds = PREY_SPEED_RANGE
    sight_bounds = PREY_SIGHT_RANGE
    fov_bounds = PREY_FOV_RANGE

    def __init__(self, x, y):
        speed = random.uniform(*Prey.speed_bounds)
        super().__init__(x, y, PREY_COLOR, PREY_SIZE, speed)
        self.q_table = {}
        self.exploration_rate = EXPLORATION_RATE
        self.sight_range = random.uniform(*Prey.sight_bounds)
        self.fov = random.uniform(*Prey.fov_bounds)

    def get_state(self, predators):
        visible_preds = []
        for pr in predators:
            dx = pr.x - self.x
            dy = pr.y - self.y
            distance = math.hypot(dx, dy)
            if distance <= self.sight_range:
                angle_to_pr = angle_to_target(self, pr)
                if angle_difference(self.orientation, angle_to_pr) <= self.fov / 2:
                    visible_preds.append(pr)
        if not visible_preds:
            return None
        nearest_pred = min(visible_preds, key=lambda pr: math.hypot(pr.x - self.x, pr.y - self.y))
        dx = nearest_pred.x - self.x
        dy = nearest_pred.y - self.y
        distance = math.hypot(dx, dy)
        if distance < self.sight_range / 2:
            state = (round(dx / 10), round(dy / 10))
        else:
            state = (round(dx / 50), round(dy / 50))
        return state

    def choose_action(self, state):
        actions = actions_list()
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

# ----------------------------
# Create Initial Populations
# ----------------------------
predators = [Predator(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(5)]
prey = [Prey(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(20)]
plants = [Plant(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(50)]

# ----------------------------
# Main Simulation Loop
# ----------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Predator-Prey Simulation with Mutating Vision and Speed")
clock = pygame.time.Clock()

running = True
while running:
    clock.tick(FPS)
    screen.fill(BACKGROUND_COLOR)

    # Process events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update spatial grid with all agents
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
            reward = -1  # Default penalty
            # Check collision with nearby prey using optimized collision detection.
            for p in nearby_prey:
                if is_colliding(predator, p):
                    predator.energy += 20
                    if p in prey:
                        prey.remove(p)
                    reward = 10  # Reward for catching prey
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
            reward = 1  # Reward for survival
            for predator in nearby_predators:
                if is_colliding(p, predator):
                    p.energy = 0  # Prey dies
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
    stats = (f"Predators: {len(predators)} | Prey: {len(prey)} | Plants: {len(plants)} | "
             f"Expl: {predators[0].exploration_rate:.2f}") if predators else ""
    text = font.render(stats, True, (255, 255, 255))
    screen.blit(text, (10, 10))

    pygame.display.flip()

pygame.quit()
