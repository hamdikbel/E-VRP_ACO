# 2E-EVRP Ant Colony Optimization Solver – Streamlit App

[![Streamlit App](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](#)

A **Streamlit-based interactive web application** that solves the **Two-Echelon Electric Vehicle Routing Problem (2E-EVRP)** using an **Ant Colony Optimization (ACO)**. Visualize routes, recharge stations, satellites, and clients with real-time optimization metrics.

---

## 📋 Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Algorithm Details](#algorithm-details)
- [Usage Guide](#usage-guide)
- [Customization](#customization)
- [References](#references)

---

## ✨ Features

| Feature                 | Details                                                              |
| ----------------------- | -------------------------------------------------------------------- |
| **ACO Optimization**    |Population-based metaheuristic with pheromone trails and heuristic guidance|
| **Battery Management**  | Realistic energy consumption & recharging at stations                |
| **Satellite Routing**   | Smart satellite visits based on load thresholds                      |
| **Time Windows**        | Constraint handling with penalties for late arrivals                 |
| **Interactive UI**      | Real-time parameter tuning (population, generations, mutation rates) |
| **Graph Visualization** | Node-edge graphs with cost labels via `streamlit-agraph`             |
| **Route Analytics**     | Battery percentage & load tracking per visit                         |

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Installation & Run

```bash
# 1. Clone the repository
git clone <https://github.com/hamdikbel/E-VRP_ACO.git>
cd E-VRP_ACO

# 2. Install dependencies
pip install streamlit streamlit-agraph numpy

# 3. Ensure dataset folder exists
# → Place "2E-EVRP-Instances-v2/" in the project root

# 4. Launch the app
streamlit run ACO_evrp.py
or
python -m streamlit run ACO_evrp.py
```

Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
E-VRP_ACO/
├── ACO_evrp.py                 # Main Streamlit application
├── 2E-EVRP-Instances-v2/          # Dataset (Type_x, Type_y folders)
├── requirements.txt               # Dependencies
├── .gitignore
└── README.md
```

---

## 🧬 Algorithm Details

### Problem Overview

The **2E-EVRP** extends VRP with:

- **Two-level delivery**: Depot → Satellites → Clients
- **Electric vehicles** with battery constraints
- **Recharging stations (f)** and **transfer satellites (s)**
- **Time windows** and **vehicle capacity limits**

### Node Types

| Type      | Notation | Color  | Purpose               |
| --------- | -------- | ------ | --------------------- |
| Depot     | `d`      | Gray   | Distribution hub      |
| Client    | `c`      | Blue   | Demand point          |
| Satellite | `s`      | Purple | Transfer/refill point |
| Station   | `f`      | Green  | Charging point        |

### Optimization Objective

```
Minimize: total_distance + penalty_cost + time_delay_penalty
Maximize: clients_served
```

### ACO Parameters

- **Number of Ants**: Colony size
- **Iterations**: Evolutionary cycles
- **Alpha**: Pheromone importance
- **Beta**: Heuristic (inverse distance) importance
- **Rho**: Pheromone evaporation rate


---

## 💻 Usage Guide

### Input Controls (Sidebar)

1. **Dataset Type**: Select `Type_x` or `Type_y`
2. **Customer Count**: Choose `5`, `10`, `15`, `50`, or `100`
3. **Instance File**: Select a `.txt` instance
4. **ACO Hyperparameters**: Tune ants, iterations, alpha, beta, and rho
5. **Run**: Click **"Lancer ACO"** to optimize

### Output Metrics

- **Total Score**: Fitness value of best solution
- **Distance**: Sum of edge costs
- **Penalties**: Constraint violations
- **Clients Served**: Coverage percentage

### Visualization

- **Interactive Graph**: Drag nodes, hover for full details
- **Route Breakdown**: Battery % and load after each stop
- **Metrics Dashboard**: Real-time performance tracking

---

## 🛠️ Customization

Enhance the algorithm by:

- Implementing elitism or local search (e.g., 2-opt)
- Adding advanced pheromone update rules
- Tuning penalty coefficients
- Supporting partial recharging
- Exporting solutions to CSV/JSON

---

## 📚 References

- [2E-EVRP Dataset Repository](https://github.com/manilakbay/2E-EVRP-Instances)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit-Agraph](https://github.com/ChrisDelClea/streamlit-agraph)
- Jie, W., et al. (2019). _The electric vehicle routing problem with time windows and recharging stations._

---

## 📝 License

MIT License – Feel free to use and modify.

---

**Made with ❤️**  
_December 2025_
