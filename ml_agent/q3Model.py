import util
from pacman import GameState
import random
import numpy as np
from pacman import Directions
import matplotlib.pyplot as plt
import math
import numpy as np
from featureExtractors import FEATURE_NAMES
import pickle


# -------------------- Decision Tree Node --------------------

class DecisionTreeNode:
    __slots__ = ("feature_idx", "threshold", "left", "right", "is_leaf",
                 "prediction", "pos", "neg", "samples")
    
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.prediction = None
        self.samples = 0
        self.pos = 0
        self.neg = 0


# -------------------- Q3Model --------------------

class Q3Model:

    # *** YOUR CODE HERE ***
    def __init__(self, **kwargs):
        """
        You can add whatever parameters you need for your model here.
        You can specify any parameter your model needs by passing it via a keyword when you call trainModel.py at the command line
        
        Make sure you add default values as you will not get to specify values when this is called in the Q3Agent. 
        Then you can safely ignore them because your Q3Agent will use a trained model.
        """
        
        # A list of which features to include by name. To exclude a feature comment out the line with that feature
        # DO NOT MODIFY this other than commenting out lines
        feature_names_to_use = [
            'closestFoodNext', 
            'closestFoodNow',
            'closestCapsuleNext', 
            'closestCapsuleNow',
            'closestGhostNext',
            'closestGhostNow',
            'closestScaredGhostNext',
            'closestScaredGhostNow',
            'eatenByGhost',
            'eatsCapsule',
            'eatsFood',
            'foodCount',
            'foodWithinFiveSpaces',
            'foodWithinNineSpaces',
            'foodWithinThreeSpaces',  
            'furthestFoodNext',
            'furthestFoodNow', 
            'numberAvailableActions',
            'numberWallsSurroundingPacman',
            'ratioCapsuleDistance',
            'ratioFoodDistance',
            'ratioGhostDistance',
            'ratioScaredGhostDistance'
            ]
        
        # DON'T MODIFY THESE TWO LINES OF CODE
        feature_name_to_idx = dict(zip(FEATURE_NAMES, np.arange(len(FEATURE_NAMES))))

        # a list of the indexs for the features that should be used. 
        self.features_to_use = [feature_name_to_idx[feature_name] for feature_name in feature_names_to_use]

        # *** YOUR CODE HERE ***
        
        self.all_feature_count = len(FEATURE_NAMES)

        # Map used feature names -> column indices in the filtered matrix (for weighting/safety)
        used_names = [FEATURE_NAMES[i] for i in self.features_to_use]
        self.used_name_to_col = dict(zip(used_names, np.arange(len(used_names))))

        # ---- Generic helpers ----
        g_int   = lambda k,d: int(kwargs.get(k, d))
        g_flt   = lambda k,d: float(kwargs.get(k, d))
        g_str   = lambda k,d: str(kwargs.get(k, d))
        g_bool  = lambda k,d: bool(kwargs.get(k, d))

        self.verbose = bool(kwargs.get("verbose", False))

        # ---- Model selection ----
        self.model_type = g_str('model_type', 'ensemble')  # 'tree' | 'logreg' | 'mlp' | 'ensemble'

        # ---- Normalization (legacy global; keep but don't use for tree) ----
        self.use_rescale = g_bool('use_rescale', True)
        self._min_ = None
        self._ptp_ = None
        self._zerovar_mask_ = None

        # ---- Decision Tree hyperparams ----
        self.max_depth         = g_int('max_depth', 8)
        self.min_samples_split = g_int('min_samples_split', 20)
        self.min_samples_leaf  = g_int('min_samples_leaf', 10)
        self.min_info_gain     = g_flt('min_info_gain', 1e-4)
        # Tree scaling flag: keep tree crisp by default
        self.tree_use_rescale  = g_bool('tree_use_rescale', False)

        # ---- Logistic Regression hyperparams ----
        self.lr            = g_flt('lr', 0.2)
        self.epochs        = g_int('epochs', 40)
        self.l2            = g_flt('l2', 5e-4)
        self.class_weight  = g_str('class_weight', 'balanced')  # 'balanced' or 'none'

        # ---- MLP hyperparams (tiny net) ----
        self.hidden        = g_int('hidden', 32)
        self.mlp_lr        = g_flt('mlp_lr', 1e-2)
        self.mlp_epochs    = g_int('mlp_epochs', 100)
        self.mlp_l2        = g_flt('mlp_l2', 1e-4)
        self.mlp_momentum  = g_flt('mlp_momentum', 0.9)
        self.mlp_batch     = g_int('mlp_batch', 128)

        # ---- Per-head scaling: tree OFF, MLP ON (its own scaler) ----
        self._mlp_min = None
        self._mlp_ptp = None

        # ---- Tree-first MLP gating (conservative) ----
        self.gate_enable   = g_bool('gate_enable', True)
        self.gate_tau_low  = g_flt('gate_tau_low', 0.45)   # tighter band than before
        self.gate_tau_high = g_flt('gate_tau_high', 0.55)
        self.gate_margin   = g_flt('gate_margin', 0.18)    # require stronger disagreement
        self.gate_mlp_conf = g_flt('gate_mlp_conf', 0.65)  # require higher confidence

        # ---- Mild calibration shrink (0=no shrink, 1=original). Use 0.9 by default.
        self.calib_alpha   = g_flt('calib_alpha', 0.9)

        # ---- Safety veto strength (subtract from prob in dangerous deltas)
        self.veto_penalty  = g_flt('veto_penalty', 0.25)

        # ---- Legacy ensemble weight (logreg vs tree)
        self.ens_w_logreg  = g_flt('ens_w_logreg', 0.6)  # used only if no MLP

        # ---- Deterministic tie-break order ----
        self._action_order = ["North", "East", "South", "West", "Stop"]

        # ---- Trained params ----
        self.tree = None
        self.w = None  # logistic weights incl. bias at index 0

        # ---- MLP trained params ----
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    # ------------------------------
    # Scaling (legacy global; kept for compat)
    # ------------------------------
    def _scale_fit(self, X):
        if not self.use_rescale:
            self._min_ = self._ptp_ = self._zerovar_mask_ = None
            return
        X = np.asarray(X, dtype=np.float64)
        self._min_ = X.min(axis=0)
        self._ptp_ = X.ptp(axis=0)
        self._zerovar_mask_ = self._ptp_ < 1e-8

    def _scale_apply(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self._min_ is None or self._ptp_ is None:
            return X
        ptp_safe = np.where(self._ptp_ < 1e-8, 1.0, self._ptp_)
        Z = (X - self._min_) / ptp_safe
        # set zero-variance features to 0.0 deterministically
        if np.ndim(Z) == 1:
            Z[self._zerovar_mask_] = 0.0
        else:
            Z[:, self._zerovar_mask_] = 0.0
        return Z

    # ------------------------------
    # Per-head MLP scaler
    # ------------------------------
    def _mlp_scale_fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self._mlp_min = X.min(axis=0)
        ptp = X.ptp(axis=0)
        self._mlp_ptp = np.where(ptp < 1e-8, 1.0, ptp)

    def _mlp_scale_apply(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self._mlp_min is None or self._mlp_ptp is None:
            return X
        return (X - self._mlp_min) / self._mlp_ptp

    # ------------------------------
    # Logistic Regression
    # ------------------------------
    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _fit_logreg(self, X, y, extra_weight=None):
        # Add bias column
        Xb = np.hstack([np.ones((X.shape[0], 1), dtype=np.float64), X])
        y = np.asarray(y, dtype=np.float64)

        # Sample weights
        if self.class_weight == 'balanced':
            n_pos = max(1.0, float(y.sum()))
            n_neg = max(1.0, float(len(y) - y.sum()))
            w1 = len(y) / (2.0 * n_pos)
            w0 = len(y) / (2.0 * n_neg)
            sample_w = np.where(y == 1.0, w1, w0)
        else:
            sample_w = np.ones_like(y)

        if extra_weight is not None:
            sample_w = sample_w * np.asarray(extra_weight, dtype=np.float64)

        self.w = np.zeros(Xb.shape[1], dtype=np.float64)
        w_sum = max(float(sample_w.sum()), 1e-9)

        for _ in range(int(self.epochs)):
            p = self._sigmoid(Xb @ self.w)
            p = np.clip(p, 1e-6, 1-1e-6)
            err = (p - y) * sample_w
            grad = (Xb.T @ err) / w_sum
            # L2 (no penalty on bias)
            grad[1:] += self.l2 * self.w[1:]
            # Optional small gradient clip for stability
            np.clip(grad, -1.0, 1.0, out=grad)
            self.w -= self.lr * grad

    def _predict_logreg_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        xb = np.hstack([np.ones((X.shape[0], 1), dtype=np.float64), X])
        p = self._sigmoid(xb @ self.w)
        return np.clip(p, 1e-6, 1-1e-6)

    # ------------------------------
    # MLP (23 -> ReLU(hidden) -> 1 sigmoid)
    # ------------------------------
    @staticmethod
    def _relu(z):
        return np.maximum(z, 0.0)

    def _mlp_forward(self, X):
        Z1 = X @ self.W1 + self.b1
        H1 = self._relu(Z1)
        logits = H1 @ self.W2 + self.b2
        logits = np.clip(logits, -30.0, 30.0)
        P = 1.0 / (1.0 + np.exp(-logits))
        return Z1, H1, logits, np.clip(P, 1e-6, 1-1e-6)

    def _predict_mlp_proba(self, X):
        if self.W1 is None:
            return np.full((X.shape[0],), 0.5, dtype=np.float64)
        _, _, _, P = self._mlp_forward(X.astype(np.float64))
        return P.ravel()

    def _fit_mlp(self, X, y, extra_weight=None):
        """
        Train a tiny MLP:  (d -> ReLU(hidden) -> 1 sigmoid)
        X must already be scaled with the MLP scaler (see _mlp_scale_fit/_mlp_scale_apply).
        Logs per-epoch train accuracy/loss to runs/mlp_train_acc.npy and runs/mlp_train_loss.npy.
        """
        import os
        X = X.astype(np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, d = X.shape
        h = int(self.hidden)

        # ----- sample weights: class-balanced * optional danger weights -----
        if self.class_weight == 'balanced':
            n_pos = max(1.0, float(y.sum()))
            n_neg = max(1.0, float(len(y) - y.sum()))
            w1 = len(y) / (2.0 * n_pos)
            w0 = len(y) / (2.0 * n_neg)
            sw = np.where(y == 1.0, w1, w0).astype(np.float64)
        else:
            sw = np.ones_like(y, dtype=np.float64)

        if extra_weight is not None:
            sw = sw * np.asarray(extra_weight, dtype=np.float64)

        w_sum = max(float(sw.sum()), 1e-9)

        # ----- init (He for hidden, small for output) -----
        rng = np.random.default_rng(12345)  # keep deterministic
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / max(1, d)), size=(d, h))
        self.b1 = np.zeros((1, h), dtype=np.float64)
        self.W2 = rng.normal(0.0, 1e-2, size=(h, 1))
        self.b2 = np.zeros((1, 1), dtype=np.float64)

        # momentum buffers
        vW1 = np.zeros_like(self.W1); vB1 = np.zeros_like(self.b1)
        vW2 = np.zeros_like(self.W2); vB2 = np.zeros_like(self.b2)

        lr  = float(self.mlp_lr)
        l2  = float(self.mlp_l2)
        mom = float(self.mlp_momentum)
        bs  = max(1, int(self.mlp_batch))
        E   = int(self.mlp_epochs)

        # ----- history for plots -----
        train_acc_hist = []
        train_loss_hist = []

        idx = np.arange(n)
        for _ in range(E):
            rng.shuffle(idx)

            # mini-batch SGD
            for s in range(0, n, bs):
                j = idx[s:s+bs]
                Xb, yb, swb = X[j], y[j], sw[j]

                # forward
                Z1 = Xb @ self.W1 + self.b1        # (B,h)
                H1 = np.maximum(Z1, 0.0)           # ReLU
                logits = H1 @ self.W2 + self.b2     # (B,1)
                logits = np.clip(logits, -30.0, 30.0)
                P = 1.0 / (1.0 + np.exp(-logits))   # sigmoid
                P = np.clip(P, 1e-6, 1-1e-6)

                # weighted BCE gradient wrt logits: (p - y) * weight / sum(weight)
                dlogit = ((P.ravel() - yb) * swb / w_sum).reshape(-1, 1)

                # grads
                gW2 = H1.T @ dlogit + l2 * self.W2
                gB2 = dlogit.sum(axis=0, keepdims=True)
                dH1 = dlogit @ self.W2.T
                dZ1 = dH1 * (Z1 > 0.0)
                gW1 = Xb.T @ dZ1 + l2 * self.W1
                gB1 = dZ1.sum(axis=0, keepdims=True)

                # clip for stability
                for g in (gW1, gB1, gW2, gB2):
                    np.clip(g, -1.0, 1.0, out=g)

                # momentum update
                vW2 = mom * vW2 + (1 - mom) * gW2; self.W2 -= lr * vW2
                vB2 = mom * vB2 + (1 - mom) * gB2; self.b2 -= lr * vB2
                vW1 = mom * vW1 + (1 - mom) * gW1; self.W1 -= lr * vW1
                vB1 = mom * vB1 + (1 - mom) * gB1; self.b1 -= lr * vB1

            # ----- epoch-end metrics (on full train) -----
            # forward on all X for metrics
            Z1_all = X @ self.W1 + self.b1
            H1_all = np.maximum(Z1_all, 0.0)
            logits_all = H1_all @ self.W2 + self.b2
            logits_all = np.clip(logits_all, -30.0, 30.0)
            P_all = 1.0 / (1.0 + np.exp(-logits_all))
            P_all = np.clip(P_all, 1e-6, 1-1e-6).ravel()

            # weighted BCE loss (with L2 terms)
            bce = -(sw * (y * np.log(P_all) + (1 - y) * np.log(1 - P_all))).sum() / w_sum
            reg = 0.5 * l2 * (np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2))
            train_loss_hist.append(float(bce + reg))

            preds = (P_all >= 0.5).astype(np.float64)
            train_acc_hist.append(float(np.mean(preds == y)))

        # ----- persist history for plotting -----
        os.makedirs("runs", exist_ok=True)
        np.save("runs/mlp_train_acc.npy", np.array(train_acc_hist, dtype=np.float64))
        np.save("runs/mlp_train_loss.npy", np.array(train_loss_hist, dtype=np.float64))


    # ------------------------------
    # Decision Tree (entropy + Laplace leaves)
    # ------------------------------
    @staticmethod
    def _entropy(labels):
        if len(labels) == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        p = counts.astype(np.float64) / float(len(labels))
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    def _information_gain(self, y, yL, yR):
        if len(yL) == 0 or len(yR) == 0:
            return 0.0
        H = self._entropy(y)
        wL = len(yL) / float(len(y))
        wR = 1.0 - wL
        return H - (wL * self._entropy(yL) + wR * self._entropy(yR))

    def _best_split(self, X, y):
        best_gain, best_f, best_t = -1.0, None, None
        nF = X.shape[1]
        for f in range(nF):
            vals = X[:, f]
            uniq = np.unique(vals)
            if len(uniq) < 2:
                continue
            # midpoints
            thr = (uniq[:-1] + uniq[1:]) * 0.5
            for t in thr:
                L = vals <= t
                R = ~L
                gain = self._information_gain(y, y[L], y[R])
                if gain > best_gain:
                    best_gain, best_f, best_t = gain, f, t
        return best_f, best_t, best_gain

    def _build_tree(self, X, y, depth=0):
        node = DecisionTreeNode()
        node.samples = len(y)

        # Stopping conditions
        unique = np.unique(y)
        if len(unique) == 1:
            node.is_leaf = True
            node.pos = int(y.sum())
            node.neg = int(len(y) - node.pos)
            node.prediction = (node.pos + 1) / (node.pos + node.neg + 2)  # Laplace
            return node
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            node.is_leaf = True
            node.pos = int(y.sum())
            node.neg = int(len(y) - node.pos)
            node.prediction = (node.pos + 1) / (node.pos + node.neg + 2)
            return node

        # Choose split
        f, t, gain = self._best_split(X, y)
        if f is None or gain < self.min_info_gain:
            node.is_leaf = True
            node.pos = int(y.sum())
            node.neg = int(len(y) - node.pos)
            node.prediction = (node.pos + 1) / (node.pos + node.neg + 2)
            return node

        node.feature_idx = f
        node.threshold = t
        Lmask = X[:, f] <= t
        Rmask = ~Lmask

        # Enforce min leaf size
        if Lmask.sum() < self.min_samples_leaf or Rmask.sum() < self.min_samples_leaf:
            node.is_leaf = True
            node.pos = int(y.sum())
            node.neg = int(len(y) - node.pos)
            node.prediction = (node.pos + 1) / (node.pos + node.neg + 2)
            return node

        node.left = self._build_tree(X[Lmask], y[Lmask], depth + 1)
        node.right = self._build_tree(X[Rmask], y[Rmask], depth + 1)
        return node

    def _tree_predict_proba_one(self, node, x):
        while not node.is_leaf:
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return float(node.prediction)

    # ------------------------------
    # Public API required by the assignment
    # ------------------------------
    def _maybe_filter(self, v):
        v = np.asarray(v, dtype=np.float64)
        if v.shape[0] == self.all_feature_count:
            return v[self.features_to_use]
        if v.shape[0] == len(self.features_to_use):
            return v
        raise ValueError(f"Unexpected feature length {v.shape[0]} (expected {self.all_feature_count} or {len(self.features_to_use)})")

    # Utility: extract a named feature from a filtered vector
    def _f(self, filtered_vec, name, default=0.0):
        j = self.used_name_to_col.get(name, None)
        if j is None or j >= filtered_vec.shape[-1]:
            return default
        return float(filtered_vec[j])

    # Utility: calibration shrink toward 0.5
    def _shrink(self, p):
        return 0.5 + (float(p) - 0.5) * float(self.calib_alpha)

    # ------------------- Public API -------------------

    def predict(self, feature_vector):
        """
        This function should take a feature vector as a numpy array and pass it through your model
        """
        # filter the data to only include your chosen features. We might not need to do this if we're working with training data that has already been filtered.
        if len(feature_vector) > len(self.features_to_use):
            vector_to_classify = feature_vector[self.features_to_use]
        else:
            vector_to_classify = feature_vector

        # *** YOUR CODE HERE ***
        
        # Filtered, unscaled (tree uses this)
        x_unscaled = self._maybe_filter(feature_vector)

        # Scaled (legacy global; used by logreg if desired)
        x_global_scaled = self._scale_apply(x_unscaled)

        # Per-head MLP scaled
        Xrow_mlp = self._mlp_scale_apply(x_unscaled.reshape(1, -1))

        # Defensive NaNs -> neutral
        if not np.all(np.isfinite(x_unscaled)):
            return 0.5

        # Head probabilities
        p_tree = None
        p_log  = None
        p_mlp  = None

        # TREE: always from UNscaled features (unless tree_use_rescale=True)
        if self.model_type in ('tree', 'ensemble') and self.tree is not None:
            xtree = self._scale_apply(x_unscaled) if self.tree_use_rescale else x_unscaled
            p_tree = self._tree_predict_proba_one(self.tree, xtree)
            p_tree = self._shrink(p_tree)

        # LOGREG: uses legacy global scaling
        if self.model_type in ('logreg', 'ensemble') and self.w is not None:
            p_log = float(self._predict_logreg_proba(x_global_scaled.reshape(1, -1))[0])
            p_log = self._shrink(p_log)

        # MLP: uses its own scaler
        if self.model_type in ('mlp', 'ensemble') and self.W1 is not None:
            p_mlp = float(self._predict_mlp_proba(Xrow_mlp)[0])
            p_mlp = self._shrink(p_mlp)

        # Pure heads (for ablations)
        if self.model_type == 'tree':
            prob = float(p_tree if p_tree is not None else 0.5)
            # soft safety veto (rare lethal deltas)
            prob -= self._safety_veto(x_unscaled)
            return max(0.0, min(1.0, prob))
        if self.model_type == 'logreg':
            prob = float(p_log if p_log is not None else 0.5)
            prob -= self._safety_veto(x_unscaled)
            return max(0.0, min(1.0, prob))
        if self.model_type == 'mlp':
            prob = float(p_mlp if p_mlp is not None else 0.5)
            prob -= self._safety_veto(x_unscaled)
            return max(0.0, min(1.0, prob))

        # ------------------------------
        # TREE-FIRST GATE (only use MLP when necessary & safe)
        # ------------------------------
        # Base prefers tree, then logreg if tree missing
        p_base = None
        if p_tree is not None:
            p_base = p_tree
        elif p_log is not None:
            p_base = p_log
        else:
            if p_mlp is not None:
                p = float(p_mlp) - self._safety_veto(x_unscaled)
                return max(0.0, min(1.0, p))
            return 0.5

        # If no MLP or gate disabled, return base with safety veto
        if (p_mlp is None) or (not self.gate_enable):
            p = float(p_base) - self._safety_veto(x_unscaled)
            return max(0.0, min(1.0, p))

        # Gate parameters
        tau_low  = float(self.gate_tau_low)
        tau_high = float(self.gate_tau_high)
        margin   = float(self.gate_margin)
        mlp_conf = float(self.gate_mlp_conf)

        # Safety predicate from filtered (unscaled) features
        safe_ok = self._safety_ok(x_unscaled)

        # Only consult MLP if tree is uncertain AND safety OK
        if safe_ok and (tau_low <= p_base <= tau_high):
            if (abs(p_mlp - p_base) >= margin) and (abs(p_mlp - 0.5) >= (mlp_conf - 0.5)):
                p_choice = p_mlp
            else:
                p_choice = p_base
        else:
            p_choice = p_base

        p_choice = float(p_choice) - self._safety_veto(x_unscaled)
        return max(0.0, min(1.0, p_choice))


    # Soft safety veto: penalize actions that move closer to a non-scared ghost without eating capsule
    def _safety_veto(self, x_unscaled):
        closestGhostNow  = self._f(x_unscaled, 'closestGhostNow',  999.0)
        closestGhostNext = self._f(x_unscaled, 'closestGhostNext', 999.0)
        eatenByGhost     = self._f(x_unscaled, 'eatenByGhost', 0.0)
        eatsCapsule      = self._f(x_unscaled, 'eatsCapsule', 0.0)
        closestScaredNow = self._f(x_unscaled, 'closestScaredGhostNow', 999.0)

        # If already death-labeled, strong veto (but we're predicting prob, so soft)
        if eatenByGhost >= 0.5:
            return self.veto_penalty

        # Moving closer to active ghost without capsule and with nearby threat
        danger = (closestGhostNext < closestGhostNow) and (eatsCapsule < 0.5) and (closestGhostNow <= 2.5) and (closestScaredNow > 2.5)
        return self.veto_penalty if danger else 0.0

    def _safety_ok(self, x_unscaled):
        closestGhostNow = self._f(x_unscaled, 'closestGhostNow', 999.0)
        wallsAround     = self._f(x_unscaled, 'numberWallsSurroundingPacman', 0.0)
        eatenByGhost    = self._f(x_unscaled, 'eatenByGhost', 0.0)
        if closestGhostNow <= 2.0 or wallsAround >= 2.0 or eatenByGhost >= 0.5:
            return False
        return True


    def selectBestAction(self, features_and_actions):
        """
        Takes a dictionary where the keys are actions and the items are feature vectors which are generated by taking the action key
        This method should return the best action based on what the model outputs for each feature vector
        """
        # *** YOUR CODE HERE ***
        
        # Pick highest prob; on near-ties, choose the safer move (avoid risky NESW bias).
        best_a = None
        best_s = -1e9
        best_safety = -1e9
        eps = 1e-9

        def safety_score(vec):
            v = self._maybe_filter(vec)
            gnow  = self._f(v, 'closestGhostNow',  999.0)
            gnext = self._f(v, 'closestGhostNext', 999.0)
            walls = self._f(v, 'numberWallsSurroundingPacman', 0.0)
            eaten = self._f(v, 'eatenByGhost', 0.0)
            cap   = self._f(v, 'eatsCapsule', 0.0)
            # higher is safer: getting farther from ghost, eating capsule, fewer walls, not dying
            return (gnext - gnow) + 1.5*cap - 0.5*walls - 2.0*eaten

        for a, f in features_and_actions.items():
            s = self.predict(f)
            safe = safety_score(f)
            if (s > best_s + eps) or (abs(s - best_s) <= eps and safe > best_safety + eps):
                best_s, best_a, best_safety = s, a, safe

        # Final deterministic tiebreak if still equal
        if best_a is None:
            for a in self._action_order:
                if a in features_and_actions:
                    return a
            # last resort
            return list(features_and_actions.keys())[0]
        return best_a

    def evaluate(self, data, labels):
        """
        This function should take a data set and corresponding labels and compute the performance of the model.
        You might for example use accuracy for classification, but you can implement whatever performance measure
        you think is suitable. You aren't evaluated what you choose here. 
        This function is just used for you to assess the performance of your training.

        The data should be a 2D numpy array where each row is a feature vector

        The labels should be a list of 1s and 0s, where the value at index i is the
        corresponding label for the feature vector at index i in the appropriate data set. For example, labels[1]
        is the label for the feature at data[1]
        """

        # filter the data to only include your chosen features
        X_eval = data[:, self.features_to_use]

        # *** YOUR CODE HERE ***

        X = np.asarray(data, dtype=np.float64)
        # NOTE: predict() handles its own per-head scaling / safety; we don't rescale here.
        y = np.asarray(labels, dtype=np.float64)
        preds = np.array([1 if self.predict(x) >= 0.5 else 0 for x in X], dtype=np.float64)
        return float(np.mean(preds == y))


    def _danger_weights_from_X(self, X_unscaled, y):
        """
        Build per-sample weights to emphasize dangerous contexts.
        Returns an array aligned with rows in X_unscaled.
        """
        # Extract columns by name if present
        def col(name):
            j = self.used_name_to_col.get(name, None)
            if j is None or j >= X_unscaled.shape[1]:
                return None
            return X_unscaled[:, j]

        gnow  = col('closestGhostNow')
        gnext = col('closestGhostNext')
        cap   = col('eatsCapsule')
        eaten = col('eatenByGhost')

        w = np.ones(X_unscaled.shape[0], dtype=np.float64)

        if gnow is not None:
            # emphasize decisions with nearby ghosts
            w *= (1.0 + 1.5 * (gnow <= 3.0))

        if gnow is not None and gnext is not None and cap is not None:
            # moving closer to ghost without capsule is high risk
            risky_delta = (gnext < gnow) & (cap < 0.5) & (gnow <= 3.0)
            w *= (1.0 + 1.5 * risky_delta.astype(np.float64))

        if eaten is not None:
            # if the sample corresponds to lethal outcome, upweight
            w *= (1.0 + 2.0 * (eaten >= 0.5))

        # Slightly upweight positives so we don't drown rare good actions
        y = np.asarray(y, dtype=np.float64)
        pos_boost = 0.25
        w *= (1.0 + pos_boost * (y >= 0.5))

        return w

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        This function should take training and validation data sets and train the model

        The training and validation data sets should be 2D numpy arrays where each row is a different feature vector

        The training and validation labels should be a list of 1s and 0s, where the value at index i is the
        corresponding label for the feature vector at index i in the appropriate data set. For example, trainingLabels[1]
        is the label for the feature at trainingData[1]
        """

        # filter the data to only include your chosen features. Use the validation data however you like.
        # if validation is provided then also filter that
        X_train = trainingData[:, self.features_to_use]
        if validationData is not None:
            X_validate = validationData[:, self.features_to_use]

        # *** YOUR CODE HERE ***
        
        Xtr_raw_all = np.asarray(trainingData, dtype=np.float64)
        ytr = np.asarray(trainingLabels, dtype=np.float64)
        Xtr_unscaled = Xtr_raw_all[:, self.features_to_use]   # always unscaled base

        # Legacy global scaler fit (for logreg if used)
        self._scale_fit(Xtr_unscaled)

        # Per-head: MLP scaler fit on unscaled
        self._mlp_scale_fit(Xtr_unscaled)
        Xtr_mlp = self._mlp_scale_apply(Xtr_unscaled)

        # Optional validation (unused for training here, but kept)
        Xva_unscaled = None
        yva = None
        if validationData is not None and validationLabels is not None:
            Xva_unscaled = np.asarray(validationData, dtype=np.float64)[:, self.features_to_use]
            yva = np.asarray(validationLabels, dtype=np.float64)

        # Danger-aware weights
        danger_w = self._danger_weights_from_X(Xtr_unscaled, ytr)

        # Train depending on model_type
        want_tree = self.model_type in ('tree', 'ensemble')
        want_log  = self.model_type in ('logreg', 'ensemble')
        want_mlp  = self.model_type in ('mlp', 'ensemble')

        if want_tree:
            # Tree uses unscaled unless explicitly overridden
            X_for_tree = self._scale_apply(Xtr_unscaled) if self.tree_use_rescale else Xtr_unscaled
            self.tree = self._build_tree(X_for_tree, ytr, depth=0)

        if want_log:
            # Logreg uses legacy global scaling
            Xtr_log = self._scale_apply(Xtr_unscaled)
            self._fit_logreg(Xtr_log, ytr, extra_weight=danger_w)

        if want_mlp:
            # MLP uses its own scaler
            self._fit_mlp(Xtr_mlp, ytr, extra_weight=danger_w)

        # --- Verbose report (for report accuracy) ---
        if self.verbose:
            tr = self.evaluate(trainingData, trainingLabels)
            print(f"[Q3] Train acc: {tr:.4f}")
            if validationData is not None and validationLabels is not None:
                va = self.evaluate(validationData, validationLabels)
                print(f"[Q3] Val acc:   {va:.4f}")


    def save(self, weights_path):
        """
        Saves your model to a .model file
        """
        # *** YOUR CODE HERE ***
        
        model_data = {
            'model_type': self.model_type,
            'features_to_use': self.features_to_use,
            'all_feature_count': self.all_feature_count,

            # legacy global scaler
            'use_rescale': self.use_rescale,
            '_min_': self._min_,
            '_ptp_': self._ptp_,
            '_zerovar_mask_': self._zerovar_mask_,

            # per-head: MLP scaler
            '_mlp_min': self._mlp_min,
            '_mlp_ptp': self._mlp_ptp,

            # tree
            'tree': self.tree,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'min_info_gain': self.min_info_gain,
            'tree_use_rescale': self.tree_use_rescale,

            # logreg
            'w': self.w,
            'lr': self.lr,
            'epochs': self.epochs,
            'l2': self.l2,
            'class_weight': self.class_weight,

            # mlp params + hparams
            'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2,
            'hidden': self.hidden,
            'mlp_lr': self.mlp_lr,
            'mlp_epochs': self.mlp_epochs,
            'mlp_l2': self.mlp_l2,
            'mlp_momentum': self.mlp_momentum,
            'mlp_batch': self.mlp_batch,

            # gate & calibration & veto
            'gate_enable': self.gate_enable,
            'gate_tau_low': self.gate_tau_low,
            'gate_tau_high': self.gate_tau_high,
            'gate_margin': self.gate_margin,
            'gate_mlp_conf': self.gate_mlp_conf,
            'calib_alpha': self.calib_alpha,
            'veto_penalty': self.veto_penalty,

            # legacy ensemble
            'ens_w_logreg': self.ens_w_logreg,

            # action order
            '_action_order': self._action_order,
        }
        with open(weights_path, 'wb') as f:
            pickle.dump(model_data, f)
        # print(f"Model saved to {weights_path}")

    def load(self, weights_path):
        """
        Loads your model from a .model file
        """
        # *** YOUR CODE HERE ***
        
        with open(weights_path, 'rb') as f:
            m = pickle.load(f)
        self.model_type = m.get('model_type', self.model_type)
        self.features_to_use = m.get('features_to_use', self.features_to_use)
        self.all_feature_count = m.get('all_feature_count', self.all_feature_count)

        # legacy global scaler
        self.use_rescale = m.get('use_rescale', self.use_rescale)
        self._min_ = m.get('_min_', None)
        self._ptp_ = m.get('_ptp_', None)
        self._zerovar_mask_ = m.get('_zerovar_mask_', None)

        # per-head MLP scaler
        self._mlp_min = m.get('_mlp_min', self._mlp_min)
        self._mlp_ptp = m.get('_mlp_ptp', self._mlp_ptp)

        # tree params
        self.tree = m.get('tree', None)
        self.max_depth = m.get('max_depth', self.max_depth)
        self.min_samples_split = m.get('min_samples_split', self.min_samples_split)
        self.min_samples_leaf = m.get('min_samples_leaf', self.min_samples_leaf)
        self.min_info_gain = m.get('min_info_gain', self.min_info_gain)
        self.tree_use_rescale = m.get('tree_use_rescale', self.tree_use_rescale)

        # logreg params
        self.w = m.get('w', None)
        self.lr = m.get('lr', self.lr)
        self.epochs = m.get('epochs', self.epochs)
        self.l2 = m.get('l2', self.l2)
        self.class_weight = m.get('class_weight', self.class_weight)

        # mlp params
        self.W1 = m.get('W1', None); self.b1 = m.get('b1', None)
        self.W2 = m.get('W2', None); self.b2 = m.get('b2', None)
        self.hidden = m.get('hidden', self.hidden)
        self.mlp_lr = m.get('mlp_lr', self.mlp_lr)
        self.mlp_epochs = m.get('mlp_epochs', self.mlp_epochs)
        self.mlp_l2 = m.get('mlp_l2', self.mlp_l2)
        self.mlp_momentum = m.get('mlp_momentum', self.mlp_momentum)
        self.mlp_batch = m.get('mlp_batch', self.mlp_batch)

        # gate & calibration & veto
        self.gate_enable = m.get('gate_enable', self.gate_enable)
        self.gate_tau_low = m.get('gate_tau_low', self.gate_tau_low)
        self.gate_tau_high = m.get('gate_tau_high', self.gate_tau_high)
        self.gate_margin = m.get('gate_margin', self.gate_margin)
        self.gate_mlp_conf = m.get('gate_mlp_conf', self.gate_mlp_conf)
        self.calib_alpha = m.get('calib_alpha', self.calib_alpha)
        self.veto_penalty = m.get('veto_penalty', self.veto_penalty)

        # legacy ensemble
        self.ens_w_logreg = m.get('ens_w_logreg', self.ens_w_logreg)

        # other
        self._action_order = m.get('_action_order', self._action_order)
        # print(f"Model loaded from {weights_path}")