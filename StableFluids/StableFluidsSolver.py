import numpy as np

class StableFluidsSolver:
    def __init__(self, x_points, y_points, x_domain, y_domain,
                 dt, viscosity, vorticity,
                 u, v, s,
                 u_velocity_conditions, v_velocity_conditions,
                 q_pressure_conditions, s_density_conditions,
                 obstacle_mask, streams,
                 gauss_seidel_viscosity_iterations = 45, gauss_seidel_pressure_iterations = 45):

        self.x_points = x_points
        self.y_points = y_points
        self.x_domain = x_domain
        self.y_domain = y_domain
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
        self.streams = streams

