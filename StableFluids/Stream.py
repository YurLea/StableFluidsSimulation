class RectangleStream:
    def __init__(self, x_left, y_left, x_size, y_size, value):
        self.x_left = x_left
        self.y_left = y_left
        self.x_size = x_size
        self.y_size = y_size
        self.value = value

    def apply_stream(self, field, dx, dy):
        i_left = int(self.x_left / dx)
        j_left = int(self.y_left / dy)
        i_right = int((self.x_left + self.x_size) / dx)
        j_right = int((self.y_left + self.y_size) / dy)
        field[i_left:i_right, j_left:j_right] = self.value
