import random
import math

# --- SPATIAL LAYOUT CONFIGURATION CONSTANTS ---
LAYOUT_SEED = 31                 # Integer for a fixed layout (e.g., 42), or None for random every time
LAYOUT_MODE = "CLUSTERED"        # "CLUSTERED" (sequential targets are grouped) or "RANDOM" (uniform scatter)

CLUSTER_MIN_STEP = 5.0          # Minimum % distance to the previous target (prevents overlapping)
CLUSTER_MAX_STEP = 50.0          # Maximum % distance to the previous target (forces visual clustering)
MIN_GLOBAL_SPACING = 8.0         # Minimum % distance from ALL previously placed targets

class TMTTaskProvider:
    def __init__(self, task_type="A", layout=None, base_layout=None, seed=LAYOUT_SEED, mode=LAYOUT_MODE):
        self.task_type = task_type
        self.seed = seed
        self.mode = mode
        
        if task_type == "A":
            self.targets = [str(i) for i in range(1, 26)]
        else:
            self.targets = [
                "1", "A", "2", "B", "3", "C", "4", "D", "5", "E",
                "6", "F", "7", "G", "8", "H", "9", "I", "10", "J",
                "11", "K", "12", "L", "13",
            ]

        self.current_index = 0
        self.completed = False

        if layout is not None:
            self.nodes = layout
        elif base_layout is not None:
            self.nodes = self._map_positions(base_layout)
        else:
            self.nodes = self._generate_spatial_layout(self.seed)

    def _generate_spatial_layout(self, seed=None):
        rng = random.Random(seed)
        
        if self.mode == "RANDOM":
            # --- OLD BEHAVIOR: UNIFORM SCATTER ---
            base_nodes = {}
            for node_index in range(1, 26):
                placed = False
                while not placed:
                    x, y = rng.uniform(5, 95), rng.uniform(5, 95)
                    if all(math.hypot(x - nx, y - ny) > MIN_GLOBAL_SPACING for nx, ny in base_nodes.values()):
                        base_nodes[node_index] = (x, y)
                        placed = True
            return self._map_positions(base_nodes)
            
        else:
            # --- NEW BEHAVIOR: SERPENTINE CLUSTERING ---
            while True:
                base_nodes = {}
                base_nodes[1] = (rng.uniform(10, 90), rng.uniform(10, 90))
                
                success = True
                for i in range(2, 26):
                    placed = False
                    prev_x, prev_y = base_nodes[i - 1]
                    
                    for _ in range(200):
                        angle = rng.uniform(0, 2 * math.pi)
                        dist = rng.uniform(CLUSTER_MIN_STEP, CLUSTER_MAX_STEP)
                        
                        nx = prev_x + math.cos(angle) * dist
                        ny = prev_y + math.sin(angle) * dist
                        
                        if 5 <= nx <= 95 and 5 <= ny <= 95:
                            if all(math.hypot(nx - ox, ny - oy) > MIN_GLOBAL_SPACING for ox, oy in base_nodes.values()):
                                base_nodes[i] = (nx, ny)
                                placed = True
                                break
                    
                    if not placed:
                        success = False
                        break 
                
                if success:
                    return self._map_positions(base_nodes)

    def _map_positions(self, base_positions):
        if self.task_type == "A":
            return {str(i): base_positions[i] for i in range(1, 26)}
        return {target: base_positions[i + 1] for i, target in enumerate(self.targets)}

    def get_current_target(self):
        if self.current_index < len(self.targets):
            return self.targets[self.current_index]
        return None

    def get_target_coords(self, target):
        return self.nodes.get(target, (50, 50))

    def submit_action(self, target):
        if self.completed:
            return False
        if target == self.targets[self.current_index]:
            self.current_index += 1
            if self.current_index >= len(self.targets):
                self.completed = True
            return True
        return False

    def get_uncompleted_targets(self):
        return self.targets[self.current_index:]

def generate_shared_layout(seed=LAYOUT_SEED, mode=LAYOUT_MODE):
    base_provider = TMTTaskProvider(task_type="A", seed=seed, mode=mode)
    base_layout = {int(k): v for k, v in base_provider.nodes.items()}
    a_layout = base_provider.nodes
    b_layout = TMTTaskProvider(task_type="B", base_layout=base_layout).nodes
    return a_layout, b_layout