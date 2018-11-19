from distutils.version import LooseVersion

import nengo
import numpy as np

if LooseVersion(nengo.__version__) > LooseVersion('2.8.0'):
    import nengo.transforms as nengo_transforms

    def conn_solver(solver, activities, targets, rng):
        return solver(activities, targets, rng=rng)

    def transform_array(transform):
        return transform.init

    def sample_transform(conn, rng=np.random):
        return conn.transform.sample(rng=rng)

else:
    nengo_transforms = None
    from nengo.dists import get_samples as _get_samples

    nengo.solvers.Solver.compositional = True
    nengo.solvers.LstsqDrop.compositional = False
    nengo.solvers.LstsqL1.compositional = False
    nengo.solvers.Nnls.compositional = False

    def conn_solver(solver, activities, targets, rng):
        # pass E=1 because solver.weights requires E to not be None, but we've
        # already multiplied targets by encoders, so multiply by 1 does nothing
        return solver(activities, targets, rng=rng,
                      E=1 if solver.weights else None)

    def transform_array(transform):
        return transform

    def sample_transform(conn, rng=np.random):
        return _get_samples(conn.transform, conn.size_out,
                            d=conn.size_mid, rng=rng)
