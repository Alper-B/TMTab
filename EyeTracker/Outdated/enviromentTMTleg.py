import random
import math

class TMTTaskProvider:
    def __init__(self, task_type="A", layout=None, base_layout=None, seed=None):
        self.task_type = task_type
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
            self.nodes = self._generate_spatial_layout(seed)

    def _generate_spatial_layout(self, seed=None):
        rng = random.Random(seed)
        base_nodes = {}
        for node_index in range(1, 26):
            placed = False
            while not placed:
                x, y = rng.uniform(5, 95), rng.uniform(5, 95)
                if all(math.hypot(x - nx, y - ny) > 8 for nx, ny in base_nodes.values()):
                    base_nodes[node_index] = (x, y)
                    placed = True
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


def generate_shared_layout(seed=None):
    base_provider = TMTTaskProvider(task_type="A", seed=seed)
    base_layout = {int(k): v for k, v in base_provider.nodes.items()}
    a_layout = base_provider.nodes
    b_layout = TMTTaskProvider(task_type="B", base_layout=base_layout).nodes
    return a_layout, b_layout
