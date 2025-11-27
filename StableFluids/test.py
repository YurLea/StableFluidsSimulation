import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cmasher as cmr
from SolverBuilder import SolverBuilder
from BoundariesDomain import DomainBoundaryDirichlet, BoundaryType, DomainBoundaryNeumann
from Obstacles import RectangleObstacle
from Stream import RectangleStream

x_size = 1.0
y_size = 1.0
x_points = 50
y_points = 50

x = np.linspace(0.0, x_size, x_points)
y = np.linspace(0.0, y_size, y_points)
X, Y = np.meshgrid(x, y, indexing="ij")

solver = (SolverBuilder()
          .with_grid_points(x_points=x_points, y_points=y_points)
          .with_grid_size(x_size=x_size, y_size=y_size)
          .with_physics_parameters(dt=0.01, viscosity=0.001, vorticity=0)
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
          .with_v_velocity_stream(RectangleStream(x_left=0.42, y_left=0.1, x_size=0.2, y_size=0.08,
                                                  value=0.1))
          #.with_obstacle(RectangleObstacle(x_left=0.45, y_left=0.5, width=0.1, height=0.05))
          .build()
          )

plt.figure(figsize=(10, 8))

for step in tqdm(range(4000)):

    curl, u0, v0 = solver.simulation_step()

    if step % 10 == 0:
        plt.clf()
        plt.contourf(X, Y, curl, cmap=cmr.redshift, levels=50)
        plt.colorbar(label='Vorticity')
        plt.quiver(X[::2, ::2], Y[::2, ::2], u0[::2, ::2], v0[::2, ::2],
                   scale=10, color='white', alpha=0.7)
        #rect = plt.Rectangle((0.42, 0.5), 0.1, 0.05,
        #                     facecolor='black', alpha=1.0, edgecolor='white')
        #plt.gca().add_patch(rect)
        plt.title(f'Time step: {step}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.draw()
        plt.pause(0.001)

plt.show()