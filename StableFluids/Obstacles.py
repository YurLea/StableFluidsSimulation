class RectangleObstacle:
    def __init__(self, x_left, y_left, width, height):
        self.x_left = x_left
        self.y_left = y_left
        self.width = width
        self.height = height

    def apply_obstacle(self, field, dx, dy):
        i_left = int(self.x_left / dx)
        j_left = int(self.y_left / dy)
        i_right = int((self.x_left + self.width) / dx)
        j_right = int((self.y_left + self.width) / dy)
        field[i_left:i_right, j_left:j_right] = True
