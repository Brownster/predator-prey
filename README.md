# Predator-Prey Simulation with Reinforcement Learning & Spatial Hashing

This repository contains a predator–prey simulation implemented in Python using PyGame. The simulation features:
- **Reinforcement Learning (Q-Learning)** for both predators (hunting) and prey (evasion).
- **Adaptive state discretization** for improved decision-making when agents are close or far apart.
- **Spatial Hashing** to efficiently manage collision detection and nearby agent queries, which makes the simulation scalable to larger populations.
- **Reproduction with mutation** to simulate evolutionary dynamics.

## Features

- **Multi-Agent Q-Learning:**  
  Both predator and prey agents learn from their interactions using a simple dictionary-based Q-learning algorithm.
  
- **Adaptive Binning:**  
  The state space is discretized adaptively. When agents (predators or prey) are close to each other, the simulation uses a finer resolution to capture critical information.

- **Spatial Hashing for Performance:**  
  The screen is divided into a grid (using a configurable cell size), and each agent is mapped into a grid cell. Collision detection and nearby agent queries only consider agents in the same or neighboring cells, significantly reducing computation time.

- **Evolutionary Dynamics:**  
  Agents reproduce when they have sufficient energy, and offspring inherit mutated parameters, allowing evolutionary dynamics to emerge.

- **Visualization:**  
  The simulation displays agents in real time using PyGame. The statistics (number of predators, prey, plants, and current exploration rate) are shown on screen.

## Installation

### Prerequisites
- Python 3.x
- [Pygame](https://www.pygame.org/news)  
  Install via pip:
  ```bash
  pip install pygame

Clone the Repository

git clone https://github.com/yourusername/predator-prey-simulation.git
cd predator-prey-simulation

Usage

Run the simulation by executing the main Python file:

python predator_prey.py

The simulation window will open. Agents are represented as colored circles:

    Predators: Red
    Prey: Green
    Plants: Dark Green

You can adjust simulation parameters (e.g., screen size, number of agents, cell size for spatial hashing) by modifying the constants at the top of predator_prey.py.
Code Overview

    predator_prey.py:
    The main simulation file that defines the agent classes (Agent, Predator, Prey, Plant), spatial hashing helper functions (get_cell, update_grid, get_nearby_agents), and the main loop for updating and drawing agents.

    Spatial Hashing:
    The simulation divides the screen into a grid of cells (with a configurable CELL_SIZE). Agents are updated in the grid using the update_grid() function, and only nearby agents are considered during collision detection and state updates.

    Q-Learning:
    Both predator and prey agents maintain a dictionary-based Q-table. The Q-values are updated using standard Q-learning rules. Agents choose actions based on an ε-greedy policy, and the exploration rate decays over time.

Future Improvements

    Deep Q-Network (DQN) Integration:
    Replace the dictionary-based Q-table with a neural network-based approximator using libraries such as PyTorch for handling continuous or high-dimensional state spaces.

    Enhanced Spatial Data Structures:
    For even larger simulations, consider implementing more advanced spatial data structures (e.g., quadtrees or k-d trees) for improved performance.

    Extended Agent Behaviors:
    Add more sophisticated behaviors, communication between agents, and environmental features.

Contributing

Contributions are welcome! Please fork the repository and open a pull request with your improvements or bug fixes.
License

This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

    The spatial hashing technique helps reduce collision detection complexity from O(n²) to O(n) see discussions on StackOverflow.
    Inspired by previous works in predator–prey dynamics and multi-agent reinforcement learning 1.
