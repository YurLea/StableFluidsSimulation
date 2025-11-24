class StableFluidsSolver:
    def __init__(self, x_points, y_points, x_domain, y_domain,
                 dt, viscosity, vorticity,
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

