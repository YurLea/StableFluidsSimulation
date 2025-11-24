from enum import Enum

class BoundaryType(Enum):
    Top = 1
    Bottom = 2
    Right = 3
    Left = 4

class DomainBoundaryDirichlet:
    def __init__(self, value, border):
        self.value = value
        self.border = border

    def apply_boundary_conditions(self, field):
        if self.border == BoundaryType.Top:
            field[0, :] = self.value
        elif self.border == BoundaryType.Bottom:
            field[-1, :] = self.value
        elif self.border == BoundaryType.Left:
            field[:, 0] = self.value
        elif self.border == BoundaryType.Right:
            field[:, -1] = self.value

class DomainBoundaryNeumann:
    def __init__(self, border):
        self.border = border

    def apply_boundary_conditions(self, field):
        if self.border == BoundaryType.Top:
            field[0, :] = field[1, :]
        elif self.border == BoundaryType.Bottom:
            field[-1, :] = field[-2, :]
        elif self.border == BoundaryType.Left:
            field[:, 0] = field[:, 1]
        elif self.border == BoundaryType.Right:
            field[:, -1] = field[:, -2]