import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import OPS
from genotypes import PRIMITIVES


class MixedOp(nn.Module):
    def __init__(self, C, stride=1):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False, False)
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class CamoNASCell(nn.Module):
    def __init__(self, C, op_names, steps):
        super().__init__()
        self.C = C
        self.steps = steps
        self.ops = nn.ModuleList()

        for i in range(steps):
            for j in range(1 + i):
                self.ops.append(MixedOp(C, stride=1))

    def forward(self, x, weights):
        states = [x]
        idx = 0

        for step in range(self.steps):
            new_states = []
            for h in states:
                new_states.append(self.ops[idx](h, weights[idx]))
                idx += 1
            states.append(sum(new_states))

        return states[-self.steps:]
