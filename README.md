# 🐜 AntColony Pathfinder

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.0+-green.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.20+-red.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-orange.svg)

**AntColony Pathfinder** is an interactive visualization of Ant Colony Optimization (ACO) algorithms for pathfinding problems. This application demonstrates how swarm intelligence can efficiently find optimal paths through complex environments.

![AntColony Pathfinder Demo](https://via.placeholder.com/800x450?text=AntColony+Pathfinder+Screenshot)

## ✨ Features

- **Interactive Environment Building**: Create and modify obstacle patterns with mouse clicks
- **Real-time Visualization**: Watch ants explore and converge on optimal paths
- **Dynamic Parameter Adjustment**: Experiment with algorithm parameters to see immediate effects
- **Live Performance Analytics**: Track optimization progress with real-time charts
- **Pre-built Obstacle Patterns**: Test algorithm performance on different environment types
- **Visual Pheromone Trails**: Observe how pheromone intensities guide the collective intelligence
- **Polished, Modern UI**: Enjoy a visually appealing interface with intuitive controls

## 🧠 About Ant Colony Optimization

Ant Colony Optimization (ACO) is a nature-inspired AI technique based on the foraging behavior of ants. In nature, ants find efficient paths to food sources through collective intelligence and pheromone communication:

1. **Exploration**: Ants initially explore randomly
2. **Pheromone Deposition**: When finding food, ants lay down pheromone trails on their return journey
3. **Attraction**: Other ants prefer paths with stronger pheromone concentrations
4. **Reinforcement**: Shorter paths accumulate more pheromone as ants travel them more frequently
5. **Evaporation**: Pheromones naturally evaporate, causing less optimal paths to fade away
6. **Convergence**: The colony gradually converges on the most efficient route

This simulation visualizes this process in real-time, allowing you to see swarm intelligence in action.

## 🔧 Installation

### Prerequisites
- Python 3.8+
- Pygame 2.0+
- NumPy
- Matplotlib

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/antcolony-pathfinder.git
   cd antcolony-pathfinder
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install pygame numpy matplotlib
   ```

4. Run the application:
   ```bash
   python ant_colony_pathfinder.py
   ```

## 🎮 How to Use

### Controls

- **Left Click**: Add obstacles
- **Right Click**: Remove obstacles
- **Space**: Start/pause simulation
- **P**: Toggle pheromone visibility
- **R**: Reset algorithm (clear pheromones)
- **C**: Clear all obstacles
- **S**: Run a single iteration
- **ESC**: Exit application

### UI Elements

The interface is divided into three main sections:

1. **Main Grid**: Where ants navigate from start (green) to end (red) points
2. **Control Panel**: Buttons and sliders to adjust simulation parameters
3. **Information Panel**: Performance charts and instructions

### Algorithm Parameters

- **Number of Ants**: Controls population size (more ants = more exploration)
- **Alpha (α)**: Pheromone importance factor (higher = more pheromone following)
- **Beta (β)**: Heuristic importance factor (higher = more greedy path selection)
- **Evaporation Rate**: How quickly pheromone trails fade (higher = faster adaptation)
- **Simulation Speed**: Controls iterations per second

## 🔬 Technical Implementation

The core algorithm is implemented in the `AntColonyOptimization` class, which uses:

```python
# Formula for path selection probability
P(i,j) = [τ(i,j)]^α * [η(i,j)]^β / Σ [τ(i,k)]^α * [η(i,k)]^β

Where:
- τ(i,j) is the pheromone value on path segment (i,j)
- η(i,j) is the heuristic value (1/distance)
- α is the pheromone importance factor
- β is the heuristic importance factor
```

Key technical features:

- **Efficient Grid Representation**: NumPy arrays for fast operations
- **Probabilistic Path Selection**: Weighted random choice based on pheromone and heuristic values
- **Dynamic Visualization**: Real-time rendering of pheromone intensity changes
- **Convergence Detection**: Algorithm detects when an optimal path has been found
- **Performance Metrics**: Tracks and visualizes path lengths across iterations

## 🛠️ Customization

The application is designed to be easily customizable:

- **Grid Resolution**: Adjust `GRID_CELLS` constant for different detail levels
- **Color Scheme**: Modify color constants to change the visual appearance
- **Custom Obstacle Patterns**: Add new patterns to the `presets` dictionary
- **Heuristic Function**: Experiment with different distance calculations
- **Visualization Style**: Change how ants, pheromones, and optimal paths are displayed

## 📚 Educational Value

This project serves as an excellent educational tool for:

- Learning about swarm intelligence algorithms
- Understanding emergent behavior in multi-agent systems
- Visualizing optimization processes
- Exploring the effects of different parameters on convergence
- Demonstrating nature-inspired computing techniques

## 📋 Future Enhancements

Potential future improvements:

- Multiple ant colonies with different parameters
- Additional environment challenges (moving obstacles, changing terrain)
- Algorithm comparison (ACO vs. other pathfinding algorithms)
- Export/import of custom environments
- 3D visualization option

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Created with 🐜 by [HAKAN AYDIN] - [haklan.aydinpl@gmail.com]

If you find this project useful, please consider starring the repository!
