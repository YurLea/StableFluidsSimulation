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
        self.source_step()

        self.vorticity_step()

    def source_step(self):

        for u_stream in self.u_streams:
            u_stream.apply_stream(self.u, self.dx, self.dy)

        for v_stream in self.v_streams:
            v_stream.apply_stream(self.v, self.dx, self.dy)

        for s_source in self.s_sources:
            s_source.apply_stream(self.s, self.dx, self.dy)

    def vorticity_step(self):
        """Vorticity confinement using numpy"""
        w = np.zeros_like(self.u)

        interior = slice(1, -1)

        valid_w = (~self.obstacle_mask[1:-1, 1:-1] &
                   ~self.obstacle_mask[2:, 1:-1] &
                   ~self.obstacle_mask[:-2, 1:-1] &
                   ~self.obstacle_mask[1:-1, 2:] &
                   ~self.obstacle_mask[1:-1, :-2])

        w[1:-1, 1:-1] = np.where(valid_w,
                                 (self.v[2:, 1:-1] - self.v[:-2, 1:-1]) / (2 * self.dx) -
                                 (self.u[1:-1, 2:] - self.u[1:-1, :-2]) / (2 * self.dy),
                                 0)

        theta_x = np.zeros_like(self.u)
        theta_y = np.zeros_like(self.u)

        valid_theta_x = ~self.obstacle_mask[1:-1, 1:-1] & ~self.obstacle_mask[2:, 1:-1] & ~self.obstacle_mask[:-2, 1:-1]
        valid_theta_y = ~self.obstacle_mask[1:-1, 1:-1] & ~self.obstacle_mask[1:-1, 2:] & ~self.obstacle_mask[1:-1, :-2]

        theta_x[1:-1, 1:-1] = np.where(valid_theta_x, (np.abs(w[2:, 1:-1]) - np.abs(w[:-2, 1:-1])) / (2 * self.dx), 0)
        theta_y[1:-1, 1:-1] = np.where(valid_theta_y, (np.abs(w[1:-1, 2:]) - np.abs(w[1:-1, :-2])) / (2 * self.dy), 0)

        magnitude = np.sqrt(theta_x ** 2 + theta_y ** 2)
        magnitude = np.where(magnitude < 1e-10, 1e-10, magnitude)

        phi_x = np.where(~self.obstacle_mask, theta_x / magnitude, 0)
        phi_y = np.where(~self.obstacle_mask, theta_y / magnitude, 0)

        tx = np.where(~self.obstacle_mask, w * phi_y, 0)
        ty = np.where(~self.obstacle_mask, -w * phi_x, 0)

        interior_i, interior_j = slice(1, -1), slice(1, -1)
        valid_cells = ~self.obstacle_mask[interior_i, interior_j]

        self.u[interior_i, interior_j] = np.where(valid_cells,
                                                  self.u[interior_i, interior_j] + self.vorticity * tx[
                                                      interior_i, interior_j] * self.dx,
                                                  self.u[interior_i, interior_j])

        self.v[interior_i, interior_j] = np.where(valid_cells,
                                                  self.v[interior_i, interior_j] + self.vorticity * ty[
                                                      interior_i, interior_j] * self.dy,
                                                  self.v[interior_i, interior_j])



