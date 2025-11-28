from StableFluids.SolverBuilder import SolverBuilder
from StableFluids.BoundariesDomain import DomainBoundaryDirichlet, BoundaryType, DomainBoundaryNeumann
from StableFluids.Obstacles import RectangleObstacle
from StableFluids.Stream import RectangleStream

class DefaultScenario:
    def __init__(self, x_points, y_points, x_domain, y_domain,
                 dt, viscosity, vorticity,
                 v_velocity_stream_parameters,
                 s_density_source_parameters,
                 rectangle_obstacle_parameters):
        self.x_points = x_points
        self.y_points = y_points
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.dt = dt
        self.viscosity = viscosity
        self.vorticity = vorticity
        self.v_velocity_stream_parameters = v_velocity_stream_parameters
        self.s_density_source_parameters = s_density_source_parameters
        self.rectangle_obstacle_parameters = rectangle_obstacle_parameters
        self.solver = self.build_solver()

    def build_solver(self):
        v_velocity_stream = RectangleStream(x_left=self.v_velocity_stream_parameters["x_left"],
                                            y_left=self.v_velocity_stream_parameters["y_left"],
                                            x_size=self.v_velocity_stream_parameters["x_size"],
                                            y_size=self.v_velocity_stream_parameters["y_size"],
                                            value=self.v_velocity_stream_parameters["value"])

        s_density_source = RectangleStream(x_left=self.s_density_source_parameters["x_left"],
                                            y_left=self.s_density_source_parameters["y_left"],
                                            x_size=self.s_density_source_parameters["x_size"],
                                            y_size=self.s_density_source_parameters["y_size"],
                                            value=self.s_density_source_parameters["value"])

        obstacle = RectangleObstacle(x_left=self.rectangle_obstacle_parameters["x_left"],
                                     y_left=self.rectangle_obstacle_parameters["y_left"],
                                     width=self.rectangle_obstacle_parameters["width"],
                                     height=self.rectangle_obstacle_parameters["height"])

        return (SolverBuilder()
          .with_grid_points(x_points=self.x_points, y_points=self.y_points)
          .with_grid_size(x_size=self.x_domain, y_size=self.y_domain)
          .with_physics_parameters(dt=self.dt, viscosity=self.viscosity, vorticity=self.vorticity)
          .with_u_velocity_boundary_conditions([DomainBoundaryDirichlet(value=0, border=BoundaryType.Top),
                                               DomainBoundaryDirichlet(value=0, border=BoundaryType.Bottom),
                                               DomainBoundaryDirichlet(value=0, border=BoundaryType.Right),
                                               DomainBoundaryDirichlet(value=0, border=BoundaryType.Left)])
          .with_v_velocity_boundary_conditions([DomainBoundaryDirichlet(value=0, border=BoundaryType.Top),
                                               DomainBoundaryDirichlet(value=0, border=BoundaryType.Bottom),
                                               DomainBoundaryDirichlet(value=0, border=BoundaryType.Right),
                                               DomainBoundaryDirichlet(value=0, border=BoundaryType.Left)])
          .with_q_pressure_boundary_conditions([DomainBoundaryNeumann(border=BoundaryType.Top),
                                               DomainBoundaryNeumann(border=BoundaryType.Bottom),
                                               DomainBoundaryNeumann(border=BoundaryType.Right),
                                               DomainBoundaryNeumann(border=BoundaryType.Left)])
          .with_s_density_boundary_conditions([DomainBoundaryNeumann(border=BoundaryType.Top),
                                               DomainBoundaryNeumann(border=BoundaryType.Bottom),
                                               DomainBoundaryNeumann(border=BoundaryType.Right),
                                               DomainBoundaryNeumann(border=BoundaryType.Left)])
          .with_v_velocity_stream(v_velocity_stream)
          .with_s_density_source(s_density_source)
          .with_obstacle(obstacle)
          .build()
          )





