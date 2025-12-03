# Q-Learning State Encoding (Q2)

This document explains the compact state representation used in the Q-Learning agent for Pac-Man.  
The goal of the encoding is to capture the core game information needed for effective decision-making while keeping the state space **small enough for tabular Q-learning**.

---

# 1. Design Goals
The state encoding is designed to satisfy:

### ✔ **Compactness**  
Tabular Q-learning scales poorly with large state spaces. The encoding compresses the environment into a small set of discrete features.

### ✔ **Task-relevant features only**  
Only information that directly affects survival and scoring is included.

### ✔ **Generalisation across layouts**  
The state space does not depend on the absolute size of the maze — only on **relative conditions** around Pac-Man.

---

# 2. State Representation Components

The full state vector is composed of the following parts:

---

## **1. Pac-Man Position (Tile-based)**
- The (x, y) tile location of Pac-Man.
- Encoded as `(pacman_x, pacman_y)`.

Reason:  
Needed to evaluate which food, ghosts and walls are relevant at the next move.

---

## **2. Ghost Proximity / Danger Zone**
For each ghost, we encode whether it is:

- In the **adjacent tiles**, or  
- Within a **Manhattan distance ≤ 2** of Pac-Man.

This is a compact discretisation:
ghost_nearby = 0   # no ghost within distance 2
ghost_nearby = 1   # ghost within distance 2 (danger)

Reason:
- Q-learning must learn to avoid risky moves.
- Exact ghost coordinates are unnecessary; danger/no-danger is enough for decisions.

---

## **3. Local Food Mask (4 Directions)**
A 4-bit mask:

| Bit | Direction | Meaning |
|-----|------------|---------|
| 0 | North | Food exists in the next tile up |
| 1 | South | Food exists in the next tile down |
| 2 | East | Food exists in the next tile right |
| 3 | West | Food exists in the next tile left |

Example:  
`0101` → food to the South and East.

Reason:  
Pac-Man tends to make most decisions based on **nearby pellets**, not global maze layout.

---

## **4. Wall Mask (4 Directions)**
Same 4-bit representation as the food mask:

| Bit | Dir | Meaning |
|-----|------|---------|
| 0 | N | Wall directly north |
| 1 | S | Wall directly south |
| 2 | E | Wall directly east |
| 3 | W | Wall directly west |

This ensures the agent never tries learning from illegal moves.

---

# 3. Final State Tuple

The final encoded state is:
(
pacman_x, pacman_y,
ghost_nearby,
food_mask_4bit,
wall_mask_4bit
)

This yields a very compact state space, approximately **40,000–80,000** distinct states depending on layout size.

---

# 4. Why This Encoding Works Well
### ✔ Compact enough for fast learning  
Tabular Q-learning converges faster when the state count is small.

### ✔ Focuses on local decision-making  
Pac-Man’s optimal behaviour is highly dependent on local food + ghost conditions.

### ✔ Layout-independent  
Works on small, medium and large layouts without modification.

---

# 5. Notes
- More granular ghost distance (e.g., 0,1,2,3…) was avoided to reduce state explosion.
- No capsule or global food count was encoded to keep the state small.
- This encoding performed robustly in experiments across all training budgets (1000–4000 episodes).

---

# 6. Reference Implementation
The encoding is implemented in `q2_agent.py` inside the method that extracts game state information for the Q-table.

def getState(self, gameState):
# compute pacman position, ghost danger, food mask, wall mask
return (pac_x, pac_y, ghost_nearby, food_mask, wall_mask)

---

This file accompanies the Q2 agent in `q_learning_agent/`.