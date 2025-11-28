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

        self.u1 = None
        self.v1 = None
        self.s1 = None
        self.u2 = None
        self.v2 = None
        self.u3 = None
        self.v3 = None

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

        u[self.obstacle_mask] = 0
        v[self.obstacle_mask] = 0

    def apply_boundary_pressure_conditions(self, q):

        for condition in self.q_pressure_conditions:
            condition.apply_boundary_conditions(q)

    def apply_density_boundary_conditions(self, s):

        for condition in self.s_density_conditions:
            condition.apply_boundary_conditions(s)

        mask_bottom = self.obstacle_mask & ~np.roll(self.obstacle_mask, -1, axis=1)

        mask_top = self.obstacle_mask & ~np.roll(self.obstacle_mask, 1, axis=1)

        mask_right = self.obstacle_mask & ~np.roll(self.obstacle_mask, -1, axis=0)

        mask_left = self.obstacle_mask & ~np.roll(self.obstacle_mask, 1, axis=0)

        s[mask_bottom] = np.roll(s, -1, axis=1)[mask_bottom]
        s[mask_top] = np.roll(s, 1, axis=1)[mask_top]
        s[mask_right] = np.roll(s, -1, axis=0)[mask_right]
        s[mask_left] = np.roll(s, 1, axis=0)[mask_left]

    def simulation_step(self):
        self.source_step()

        self.vorticity_step()

        self.advection_step()

        self.diffusion_step()

        self.pressure_poisson_step()

        self.correction_step()

        return self.s, self.u, self.v

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

        self.apply_boundary_velocity_conditions(self.u, self.v)
        self.apply_density_boundary_conditions(self.s)

    def advection_step(self):
        """Semi-Lagrangian advection step using numpy"""
        self.u1 = self.u.copy()
        self.v1 = self.v.copy()
        self.s1 = self.s.copy()

        I, J = np.meshgrid(np.arange(1, self.x_points - 1), np.arange(1, self.y_points - 1), indexing='ij')
        mask = ~self.obstacle_mask[I, J]

        I_val, J_val = I[mask], J[mask]

        x_prev = I_val * self.dx - self.dt * self.u[I_val, J_val]
        y_prev = J_val * self.dy - self.dt * self.v[I_val, J_val]

        x_prev = np.clip(x_prev, self.dx / 2, self.x_domain - self.dx / 2)
        y_prev = np.clip(y_prev, self.dy / 2, self.y_domain - self.dy / 2)

        i1 = (x_prev / self.dx).astype(int)
        i2 = i1 + 1
        j1 = (y_prev / self.dy).astype(int)
        j2 = j1 + 1

        x1, x2 = i1 * self.dx, i2 * self.dx
        y1, y2 = j1 * self.dy, j2 * self.dy

        wx1 = (x2 - x_prev) / self.dx
        wx2 = (x_prev - x1) / self.dx
        wy1 = (y2 - y_prev) / self.dy
        wy2 = (y_prev - y1) / self.dy

        for field_in, field_out in [(self.u, self.u1), (self.v, self.v1), (self.s, self.s1)]:
            field_out[I_val, J_val] = (wx1 * wy1 * field_in[i1, j1] +
                                       wx2 * wy1 * field_in[i2, j1] +
                                       wx1 * wy2 * field_in[i1, j2] +
                                       wx2 * wy2 * field_in[i2, j2])

        self.apply_boundary_velocity_conditions(self.u1, self.v1)
        self.apply_density_boundary_conditions(self.s1)

    def diffusion_step(self):
        """Viscous diffusion using Gauss-Seidel iterations with numpy"""
        self.u2 = self.u1.copy()
        self.v2 = self.v1.copy()

        alpha = self.viscosity * self.dt
        dx2 = self.dx * self.dx
        dy2 = self.dy * self.dy
        denominator = 1 + 2 * alpha * (1 / dx2 + 1 / dy2)

        interior_i = slice(1, self.x_points - 1)
        interior_j = slice(1, self.y_points - 1)
        valid_mask = ~self.obstacle_mask[interior_i, interior_j]

        for _ in range(self.gauss_seidel_viscosity_iterations):
            u_update = (self.u1[interior_i, interior_j] + alpha * (
                    (self.u2[2:, interior_j] + self.u2[:-2, interior_j]) / dx2 +
                    (self.u2[interior_i, 2:] + self.u2[interior_i, :-2]) / dy2
            )) / denominator

            self.u2[interior_i, interior_j] = np.where(valid_mask, u_update, self.u2[interior_i, interior_j])

            v_update = (self.v1[interior_i, interior_j] + alpha * (
                    (self.v2[2:, interior_j] + self.v2[:-2, interior_j]) / dx2 +
                    (self.v2[interior_i, 2:] + self.v2[interior_i, :-2]) / dy2
            )) / denominator

            self.v2[interior_i, interior_j] = np.where(valid_mask, v_update, self.v2[interior_i, interior_j])

            self.apply_boundary_velocity_conditions(self.u2, self.v2)

    def pressure_poisson_step(self):
        """Pressure-Poisson step with numpy"""
        self.u3, self.v3 = self.u2.copy(), self.v2.copy()

        dx2, dy2 = self.dx * self.dx, self.dy * self.dy
        i, j = slice(1, -1), slice(1, -1)

        solid_center = self.obstacle_mask[i, j]
        solid_right = self.obstacle_mask[2:, j]
        solid_left = self.obstacle_mask[:-2, j]
        solid_top = self.obstacle_mask[i, 2:]
        solid_bottom = self.obstacle_mask[i, :-2]

        for _ in range(self.gauss_seidel_pressure_iterations):
            sum_q = (np.where(~solid_right, self.q[2:, j], 0) / dx2 +
                     np.where(~solid_left, self.q[:-2, j], 0) / dx2 +
                     np.where(~solid_top, self.q[i, 2:], 0) / dy2 +
                     np.where(~solid_bottom, self.q[i, :-2], 0) / dy2)

            sum_coefficient = (np.where(~solid_right, 1.0, 0) / dx2 +
                         np.where(~solid_left, 1.0, 0) / dx2 +
                         np.where(~solid_top, 1.0, 0) / dy2 +
                         np.where(~solid_bottom, 1.0, 0) / dy2)

            rhs = ((self.u3[2:, j] - self.u3[:-2, j]) / (2 * self.dx) +
                   (self.v3[i, 2:] - self.v3[i, :-2]) / (2 * self.dy))

            update_mask = ~solid_center & (sum_coefficient > 0)
            self.q[i, j] = np.where(update_mask, (sum_q - rhs) / sum_coefficient, self.q[i, j])

            self.apply_boundary_pressure_conditions(self.q)

    def correction_step(self):
        """Velocity correction step with numpy"""
        u_corrected = self.u3.copy()
        v_corrected = self.v3.copy()

        i, j = slice(1, self.x_points - 1), slice(1, self.y_points - 1)

        valid_u_correction = (~self.obstacle_mask[i, j] &
                              ~self.obstacle_mask[2:, j] &
                              ~self.obstacle_mask[:-2, j])

        valid_v_correction = (~self.obstacle_mask[i, j] &
                              ~self.obstacle_mask[i, 2:] &
                              ~self.obstacle_mask[i, :-2])

        u_gradient = (self.q[2:, j] - self.q[:-2, j]) / (2 * self.dx)
        u_corrected[i, j] = np.where(valid_u_correction,
                                     self.u3[i, j] - u_gradient,
                                     self.u3[i, j])

        v_gradient = (self.q[i, 2:] - self.q[i, :-2]) / (2 * self.dy)
        v_corrected[i, j] = np.where(valid_v_correction,
                                     self.v3[i, j] - v_gradient,
                                     self.v3[i, j])

        self.apply_boundary_velocity_conditions(u_corrected, v_corrected)

        self.u = u_corrected.copy()
        self.v = v_corrected.copy()
        self.s = self.s1.copy()

    def curl_2d(self, u, v):
        """Calculate 2D vorticity"""
        curl = np.zeros((self.x_points, self.y_points))

        inner = np.s_[1:-1, 1:-1]
        right = np.s_[2:, 1:-1]
        left = np.s_[:-2, 1:-1]
        top = np.s_[1:-1, 2:]
        bottom = np.s_[1:-1, :-2]

        dv_dx = (v[right] - v[left]) / (2 * self.dx)
        du_dy = (u[top] - u[bottom]) / (2 * self.dy)
        curl[inner] = dv_dx - du_dy

        return curl

    def get_curl_2d(self):
        return self.curl_2d(self.u, self.v)

    def get_velocity_fields(self):
        return self.u, self.v

    def get_density_field(self):
        return self.s

