from firedrake import File
import time
import numpy as np
import spyro


#line1 = spyro.create_transect((6.0, 6.0), (6.0, 11.8), 4)
#line2 = spyro.create_transect((6.0, 4.6), (6.0, 0.4), 4)
# line3 = spyro.create_transect((5.59, 2.514), (5.59, 9.514), 4)
# line4 = spyro.create_transect((3.59, 2.514), (3.59, 9.514), 4)
# line5 = spyro.create_transect((1.59, 2.514), (1.59, 9.514), 4)
# line1 = spyro.create_transect((-1.75, 0.25), (-1.75, 7.25), 4)
# line2 = spyro.create_transect((1.75, 0.25), (1.75, 7.25), 4)
# line3 = spyro.create_transect((3.75, 0.25), (3.75, 7.25), 4)
# line4 = spyro.create_transect((5.5, 0.25), (5.25, 7.25), 4)
# line5 = spyro.create_transect((7.25, 0.25), (7.25, 7.25), 4)
# lines = np.concatenate((line1, line2))
# lines = np.concatenate((line1,line2))
# sources = spyro.insert_fixed_value(lines, -0.6455, 0)

# lines = np.concatenate((line1, line2, line3, line4, line5))
lines = spyro.read_coordinate_from_txt('/home/firedrake/project/src_rec/source_points.txt')
sources = spyro.insert_fixed_value(lines, 7.75, 1)
linesr = spyro.read_coordinate_from_txt('/home/firedrake/project/src_rec/ellipse_points.txt')
receivers = spyro.insert_fixed_value(linesr, 7.75, 1)
#line3 = spyro.create_transect((0.5, 6.0), (6.0, 6.0), 200)
#line4 = spyro.create_transect((6.0, 6.0), (11.5, 6.0), 200)
#linesr = np.concatenate((line3, line4))
#receivers = spyro.insert_fixed_value(linesr, -0.15, 0)
# receivers = spyro.create_2d_grid(1.0, 10, 1.0, 10, 10)
# receivers = spyro.insert_fixed_value(receivers, -0.15, 0)

model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadrature": "KMV",  # Equi or KMV
    "degree": 2,  # p order
    "dimension": 3,  # dimension
}
model["parallelism"] = {"type": "spatial"}  # automatic",
model["mesh"] = {
    "Lz": 8.75,  # depth in km - always positive
    "Lx": 12.0,  # width in km - always positive
    "Ly": 12.0,  # thickness in km - always positive
    "meshfile": "/home/firedrake/project/meshes/Brain3D2PML.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "/home/firedrake/project/velocity_models/head_velocity3Doffbrainbeam.hdf5",
}
# model["mesh"] = {
#     "Lz": 5.175,  # depth in km - always positive
#     "Lx": 7.5,  # width in km - always positive
#     "Ly": 7.5,  # thickness in km - always positive
#     "meshfile": "meshes/overthrust3D_guess_model.msh",
#     "initmodel": "velocity_models/overthrust_3D_guess_model.hdf5",
#     "truemodel": "velocity_models/overthrust_3D_true_model.hdf5",
# }
# model["mesh"] = {
#     "Lz": 12.88,  # depth in km - always positive
#     "Lx": 81.25,  # width in km - always positive
#     "Ly": 41.25,  # thickness in km - always positive
#     "meshfile": "meshes/headvolume.msh",
#     "initmodel": "velocity_models/overthrust_3D_guess_model.hdf5",
#     "truemodel": "velocity_models/overthrust_3D_true_model.hdf5",
# }
model["BCs"] = {
    "status": True,  # True or false
    "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 4.5,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": 2.0,  # thickness of the PML in the z-direction (km) - always positive
    "lx": 2.0,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 2.0,  # thickness of the PML in the y-direction (km) - always positive
}
model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": len(sources),
    "source_pos": sources,
    "frequency": 5.0,
    "delay": 1.0,
    "num_receivers": len(receivers),
    "receiver_locations": receivers,
    "num_rec_x_columns": 25,  # number of receivers in x direction
    "num_rec_y_columns": 0,  # number of receivers in y direction
    "num_rec_z_columns": 25,  # number of receivers in z direction
}
model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 4,  # Final time for event
    "dt": 0.000035,
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 99999,  # how frequently to save solution to RAM
}
comm = spyro.utils.mpi_init(model)
mesh, V = spyro.io.read_mesh(model, comm)
if comm.ensemble_comm.rank == 0:
    print(f"The mesh has {V.dim()} degrees of freedom")
vp = spyro.io.interpolate(model, mesh, V, guess=False)
if comm.ensemble_comm.rank == 0:
    File("true_velocity.pvd", comm=comm.comm).write(vp)
sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)
wavelet = spyro.full_ricker_wavelet(
    dt=model["timeaxis"]["dt"],
    tf=model["timeaxis"]["tf"],
    freq=model["acquisition"]["frequency"],
)
t1 = time.time()
p, p_r = spyro.solvers.forward(
    model, mesh, comm, vp, sources, wavelet, receivers, output=True
)
print(time.time() - t1, flush=True)
spyro.plots.plot_shots(model, comm, p_r, itera=1, vmin=-1e-3, vmax=1e-3)
spyro.io.save_shots(model,comm,p_r)

