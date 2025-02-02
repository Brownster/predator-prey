import pygame
import random
import math
import numpy as np
# Uncomment these if you want to experiment with a DQN-based approach.
# import torch
# import torch.nn as nn

# Constants and parameters
WIDTH, HEIGHT = 800, 600
FPS = 30

# Colors
PREDATOR_COLOR = (255, 0, 0)    # Red
PREY_COLOR = (0, 255, 0)        # Green
PLANT_COLOR = (0, 128, 0)       # Dark Green
BACKGROUND_COLOR = (0, 0, 0)    # Black

# Sizes and speeds
PREDATOR_SIZE, PREY_SIZE, PLANT_SIZE = 10, 8, 6
PREDATOR_SPEED, PREY_SPEED = 2, 1.5

# Energy and reproduction
ENERGY_LOSS_RATE = 0.1
REPRODUCTION_ENERGY = 20
LIFESPAN_RANDOMNESS = 50

# Q-learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 1.0
EXPLORATION_DECAY = 0.995
MIN_EXPLORATION_RATE = 0.01

# Initialize PyGame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Predator-Prey Simulation with RL & Evolution")
clock = pygame.time.Clock()

# Base Agent class
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
        # Move agent in one of four directions (or randomly if no action provided)
        if direction == "up":
            self.y -= self.speed
        elif direction == "down":
            self.y += self.speed
        elif direction == "left":
            self.x -= self.speed
        elif direction == "right":
            self.x += self.speed
        else:
            # Move randomly if no direction is provided
            self.x += random.choice([-self.speed, self.speed])
            self.y += random.choice([-self.speed, self.speed])
        # Keep within screen bounds
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
            # Mutate offspring parameters slightly
            offspring.speed += random.uniform(-0.2, 0.2)
            # If using Q-table, mutate its entries if present
            if hasattr(self, 'q_table'):
                # Create a shallow copy with small random changes
                offspring.q_table = {state: {action: val + random.uniform(-0.1, 0.1)
                                             for action, val in actions.items()}
                                     for state, actions in self.q_table.items()}
            return offspring
        return None

# Predator class using Q-learning
class Predator(Agent):
    def __init__(self, x, y):
        super().__init__(x, y, PREDATOR_COLOR, PREDATOR_SIZE, PREDATOR_SPEED, is_predator=True)
        self.q_table = {}  # Q-table for state-action values
        self.exploration_rate = EXPLORATION_RATE

    def get_state(self, prey_list):
        # If no prey exists, return None
        if not prey_list:
            return None
        # Choose nearest prey
        nearest_prey = min(prey_list, key=lambda p: math.hypot(p.x - self.x, p.y - self.y))
        dx = nearest_prey.x - self.x
        dy = nearest_prey.y - self.y
        distance = math.hypot(dx, dy)
        # Adaptive binning: higher resolution if close (<50 pixels)
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
                # Exploit: choose the action with the highest Q-value
                return max(self.q_table[state], key=self.q_table[state].get)
            else:
                return random.choice(actions)

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in ["up", "down", "left", "right"]}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in ["up", "down", "left", "right"]}
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())
        new_value = old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max - old_value)
        self.q_table[state][action] = new_value

# Prey class using Q-learning for evasion
class Prey(Agent):
    def __init__(self, x, y):
        super().__init__(x, y, PREY_COLOR, PREY_SIZE, PREY_SPEED)
        self.q_table = {}  # Q-table for evasion
        self.exploration_rate = EXPLORATION_RATE

    def get_state(self, predators):
        if not predators:
            return None
        # Find nearest predator
        nearest_predator = min(predators, key=lambda pr: math.hypot(pr.x - self.x, pr.y - self.y))
        dx = nearest_predator.x - self.x
        dy = nearest_predator.y - self.y
        distance = math.hypot(dx, dy)
        # You can also use adaptive binning for prey state;
        # here we use fixed binning for simplicity:
        state = (round(dx / 20), round(dy / 20))
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
            self.q_table[state] = {a: 0 for a in ["up", "down", "left", "right"]}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in ["up", "down", "left", "right"]}
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())
        new_value = old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max - old_value)
        self.q_table[state][action] = new_value

# Plant class (non-learning, static behavior)
class Plant(Agent):
    def __init__(self, x, y):
        super().__init__(x, y, PLANT_COLOR, PLANT_SIZE, 0)
        # Plants do not move, so energy and age are not critical here.
        self.last_reproduced = pygame.time.get_ticks()

    def reproduce(self, plants):
        # Plants reproduce every fixed amount of time
        if pygame.time.get_ticks() - self.last_reproduced > 500:
            self.last_reproduced = pygame.time.get_ticks()
            return Plant(self.x + random.randint(-20, 20), self.y + random.randint(-20, 20))
        return None

# Create initial populations
predators = [Predator(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(5)]
prey = [Prey(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(20)]
plants = [Plant(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(50)]

# Main simulation loop
running = True
while running:
    clock.tick(FPS)
    screen.fill(BACKGROUND_COLOR)

    # Handle quit events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update predators (RL to hunt prey)
    for predator in predators[:]:
        state = predator.get_state(prey)
        if state:
            action = predator.choose_action(state)
            predator.move(action)
            # Check for collision with prey (simple circle collision)
            nearest_prey = min(prey, key=lambda p: math.hypot(p.x - predator.x, p.y - predator.y))
            if math.hypot(nearest_prey.x - predator.x, nearest_prey.y - predator.y) < predator.size:
                predator.energy += 20
                prey.remove(nearest_prey)
                reward = 10  # reward for eating prey
            else:
                reward = -1  # small penalty for moving without eating
            next_state = predator.get_state(prey)
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

    # Update prey (RL to evade predators)
    for p in prey[:]:
        state = p.get_state(predators)
        if state:
            action = p.choose_action(state)
            p.move(action)
            # If a predator is too close, prey gets eaten (dies)
            nearest_predator = min(predators, key=lambda pr: math.hypot(pr.x - p.x, pr.y - p.y))
            if math.hypot(nearest_predator.x - p.x, nearest_predator.y - p.y) < nearest_predator.size:
                p.energy = 0   # Prey dies
                reward = -10  # penalty for being caught
            else:
                reward = 1    # reward for surviving
            next_state = p.get_state(predators)
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

    # Update plants (static behavior)
    for plant in plants[:]:
        plant.draw(screen)
        new_plant = plant.reproduce(plants)
        if new_plant:
            plants.append(new_plant)
    # Occasionally add new plants randomly
    if random.random() < 0.02:
        plants.append(Plant(random.randint(0, WIDTH), random.randint(0, HEIGHT)))

    # Display stats on screen
    font = pygame.font.SysFont("Arial", 18)
    stats = f"Predators: {len(predators)} | Prey: {len(prey)} | Plants: {len(plants)} | Expl: {predators[0].exploration_rate:.2f}"
    text = font.render(stats, True, (255, 255, 255))
    screen.blit(text, (10, 10))

    pygame.display.flip()

pygame.quit()
