from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, "..", "data", "circuit.json")


class Circuit:

    def __init__(self, path=None):
        path = path or DEFAULT_PATH
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.meta = data.get("meta", {})
        self.neurons = data["neurons"]
        self.edges = data["edges"]
        self.n = len(self.neurons)
        self.by_role = defaultdict(list)
        self.by_type = defaultdict(list)
        self.by_side = defaultdict(list)
        for i, nb in enumerate(self.neurons):
            self.by_role[nb.get("role", "")].append(i)
            self.by_type[nb.get("type", "")].append(i)
            self.by_side[nb.get("side", "")].append(i)
        self.roles = sorted(k for k in self.by_role if k)

    def indices(self, role=None, type_=None, side=None):
        if role is not None:
            idx = list(self.by_role.get(role, []))
        elif type_ is not None:
            idx = list(self.by_type.get(type_, []))
        else:
            idx = list(range(self.n))
        if side is not None:
            idx = [i for i in idx if self.neurons[i].get("side") == side]
        return idx

    def edge_arrays(self):
        if not self.edges:
            z = np.zeros(0, dtype=np.int32)
            return z, z, np.zeros(0, dtype=np.float32)
        e = np.asarray(self.edges, dtype=np.float64)
        return (e[:, 0].astype(np.int32),
                e[:, 1].astype(np.int32),
                e[:, 2].astype(np.float32))

    def describe(self):
        lines = [f"神经元 {self.n}  突触 {len(self.edges)}"]
        for role in self.roles:
            idx = self.by_role[role]
            sides = {self.neurons[i].get("side", "") for i in idx}
            lines.append(f"  {role:<14} {len(idx):>4}  {'/'.join(sorted(s for s in sides if s))}")
        return "\n".join(lines)

    def id_to_index(self):
        return {nb["id"]: i for i, nb in enumerate(self.neurons)}
