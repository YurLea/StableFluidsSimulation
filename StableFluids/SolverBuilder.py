from StableFluidsSolver import StableFluidsSolver
import numpy as np

class SolverBuilder:
    def __init__(self):
        self.x_points = None
        self.y_points = None
        self.x_size = None
        self.y_size = None
        self.dt = None
        self.vorticity = None
        self.viscosity = None
        self.u = None
        self.v = None
        self.s = None
        self.obstacle_mask = None
        self.u_velocity_conditions = []
        self.v_velocity_conditions = []
        self.q_pressure_conditions = []
        self.s_density_conditions = []
        self.u_velocity_streams = []
        self.v_velocity_streams = []
        self.s_density_sources = []

    def with_grid_points(self, x_points, y_points):
        self.x_points = x_points
        self.y_points = y_points
        self.u = np.zeros((x_points, y_points))
        self.v = np.zeros((x_points, y_points))
        self.s = np.zeros((x_points, y_points))
        self.obstacle_mask = np.full((x_points, y_points), False)
        return self

    def with_grid_size(self, x_size, y_size):
        self.x_size = x_size
        self.y_size = y_size
        return self

    def with_physics_parameters(self, vorticity, viscosity, dt):
        self.vorticity = vorticity
        self.viscosity = viscosity
        self.dt = dt
        return self

    def with_obstacle(self, obstacle):
        dx = self.x_size / (self.x_points - 1)
        dy = self.y_size / (self.y_points - 1)
        self.obstacle_mask = obstacle.apply_obstacle(self.obstacle_mask, dx, dy)
        return self

    def with_u_velocity_condition(self, condition):
        self.u_velocity_conditions.append(condition)
        return self

    def with_v_velocity_condition(self, condition):
        self.v_velocity_conditions.append(condition)
        return self

    def with_q_pressure_condition(self, condition):
        self.q_pressure_conditions.append(condition)
        return self

    def with_s_density_condition(self, condition):
        self.s_density_conditions.append(condition)
        return self

    def with_u_velocity_stream(self, stream):
        self.u_velocity_streams.append(stream)
        return self

    def with_v_velocity_stream(self, stream):
        self.v_velocity_streams.append(stream)
        return self

    def with_s_density_source(self, source):
        self.s_density_sources.append(source)
        return self

    def build(self):
        return StableFluidsSolver(self.x_points, self.y_points, self.x_size, self.y_size,
                                  self.dt, self.viscosity, self.vorticity,
                                  self.u, self.v, self.s,
                                  self.u_velocity_conditions, self.v_velocity_conditions,
                                  self.q_pressure_conditions, self.s_density_conditions,
                                  self.obstacle_mask, self.u_velocity_streams, self.v_velocity_streams,
                                  self.s_density_sources)