# Autonomous-Pac-Man-AI

# Pac-Man AI System — Reinforcement Learning & Machine Learning Agents (Python)

An autonomous AI system built to play the classic game Pac-Man, featuring a novel hybrid decision architecture that combines Tabular Q-Learning with a custom, Gated Ensemble Machine Learning model for robust real-time action selection.

This project was developed *from scratch*, with **all core algorithms implemented manually in Python** (no use of Scikit-learn, PyTorch, or other major ML frameworks), demonstrating a deep, foundational understanding of both Reinforcement Learning and Supervised Machine Learning principles.

## 🚀 Key Features and Architecture

The AI is structured into two main components that work in tandem to control the agent:

### 1. Reinforcement Learning Agent (Tabular Q-Learning)

A classic **Tabular Q-Learning** agent was developed to handle general state exploration and optimal long-term pathfinding across the maze environment.

| Feature | Detail |
| :--- | :--- |
| **Algorithm** | Q-Learning for off-policy control and value function approximation. |
| **State Abstraction** | Engineered a compact, custom state representation (feature-based state space) to ensure generalizability across different maze layouts, resulting in an effective state space of **~40,000 to 80,000 entries**. |
| **Exploration Policy** | Optimized $\epsilon$-greedy exploration policy. |
| **Hyperparameters** | Optimized using grid search over multiple training episodes: $\epsilon=0.12, \alpha=0.22, \gamma=0.90$. |

### 2. Hybrid ML Ensemble Decision Model (Action Selection)

A unique **Machine Learning ensemble** was trained offline to specialize in immediate, critical action selection, especially in danger or uncertainty zones. The model performs real-time inference via successor-state evaluation.

| Component | Detail |
| :--- | :--- |
| **Architecture** | A Gated Hybrid Ensemble combining three distinct models: **Decision Tree**, **Multilayer Perceptron (MLP)** (23-32-1), and **Logistic Regression**. |
| **Implementation** | All models (DT, MLP, LogReg with SGD & Momentum) were **implemented manually** in Python. |
| **Custom Gating Logic** | Implemented uncertainty gating ($\tau_{\text{low}}=0.45, \tau_{\text{high}}=0.55$) to selectively switch between model predictions or fall back to safety rules based on model confidence. |
| **Danger-Aware Learning** | Applied **sample weighting** and **cost-sensitive learning** during training to prioritize safe actions and heavily penalize moves that lead to 'danger states.' |
| **Real-time Integration** | Trained model was saved/loaded offline and hooked into the live simulator, demonstrating seamless real-time action selection for the Pac-Man agent. |

## 📊 Experimental Results and Engineering

The project involved rigorous experimental engineering to ensure performance stability and generalisation.

* **Ablation Studies:** Conducted to quantify the performance gain of the ML ensemble and the custom state encoding over simpler baselines.
* **Generalisation Testing:** Evaluated the trained agent on unseen maze layouts of small, medium, and large size to confirm robustness against overfitting.
* **Hyperparameter Tuning:** Performed grid search over the core RL parameters ($\epsilon$, $\alpha$, $\gamma$) to maximize learning speed and final score across training episodes.

**(Self-Correction/Placeholder: You would replace the following with one of your actual charts or tables and a brief caption. For example, a chart showing the Learning Curve or Ablation Results.)**

> **[INSERT CHART/TABLE IMAGE HERE: e.g., A learning curve showing average score vs. episode count]**
>
> **_Caption Example:_** *Performance stability and score improvement over 2,000 training episodes, demonstrating the effectiveness of the optimized $\gamma$ and $\alpha$ values.*

## 💻 Tech Stack & Setup

* **Language:** Python
* **Libraries:** NumPy (for optimized numerical operations)
* **Simulator:** Custom Pac-Man simulator environment (based on a university-provided scaffold)

### Getting Started

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YourUsername/pacman-ai-system-rl-ml-agents.git](https://github.com/YourUsername/pacman-ai-system-rl-ml-agents.git)
    cd pacman-ai-system-rl-ml-agents
    ```
2.  Install dependencies:
    ```bash
    pip install numpy
    ```
3.  To run the trained agent on a sample layout:
    ```bash
    python src/main_simulator.py --layout mediumClassic --mode test
    ```
4.  To retrain the Q-Learning model:
    ```bash
    python src/rl_agent.py --train --episodes 4000
    ```