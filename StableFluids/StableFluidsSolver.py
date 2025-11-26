import numpy as np

class StableFluidsSolver:
    def __init__(self, x_points, y_points, x_domain, y_domain,
                 dt, viscosity, vorticity,
                 u, v, s,
                 u_velocity_conditions, v_velocity_conditions,
                 q_pressure_conditions, s_density_conditions,
                 obstacle_mask, u_streams, v_streams, s_sources,
                 gauss_seidel_viscosity_iterations = 45, gauss_seidel_pressure_iterations = 45):

        self.x_points = x_points
        self.y_points = y_points
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.dx = x_domain / (x_points - 1)
        self.dy = y_domain / (y_points - 1)
        self.dt = dt
        self.viscosity = viscosity
        self.vorticity = vorticity
        self.gauss_seidel_pressure_iterations = gauss_seidel_pressure_iterations
        self.gauss_seidel_viscosity_iterations = gauss_seidel_viscosity_iterations

        self.u = u
        self.v = v
        self.s = s
        self.q = np.zeros((x_points, y_points))

        self.u_velocity_conditions = u_velocity_conditions
        self.v_velocity_conditions = v_velocity_conditions
        self.q_pressure_conditions = q_pressure_conditions
        self.s_density_conditions = s_density_conditions

        self.obstacle_mask = obstacle_mask
        self.u_streams = u_streams
        self.v_streams = v_streams
        self.s_sources = s_sources

    def apply_boundary_velocity_conditions(self, u, v):

        for condition in self.u_velocity_conditions:
            condition.apply_boundary_conditions(u)

        for condition in self.v_velocity_conditions:
            condition.apply_boundary_conditions(v)

    def apply_boundary_pressure_conditions(self, q):

        for condition in self.q_pressure_conditions:
            condition.apply_boundary_pressure(q)

    def apply_density_boundary_conditions(self, s):

        for condition in self.s_density_conditions:
            condition.apply_density_boundary_conditions(s)

    def simulation_step(self):

        for u_stream in self.u_streams:
            u_stream.apply_stream(self.u, self.dx, self.dy)

        for v_stream in self.v_streams:
            v_stream.apply_stream(self.v, self.dx, self.dy)

        for s_source in self.s_sources:
            s_source.apply_stream(self.s, self.dx, self.dy)

