"""OmnisVox CFD - adaptive rollback-refinement pseudo-spectral Navier-Stokes solver."""
__version__ = "1.1.2"

from .backend import Backend, BackendError, get_backend, gpu_status  # noqa: E402,F401
from .arr import ARRConfig, ARRController, SimulationFailed  # noqa: E402,F401
from .base import SolverFault  # noqa: E402,F401
from .cases import CASES, get_case  # noqa: E402,F401
from .runner import SimConfig, Simulation, resolve_config, json_safe  # noqa: E402,F401
from .solver2d import NavierStokes2D  # noqa: E402,F401
from .solver3d import NavierStokes3D  # noqa: E402,F401
from .spectral import SpectralGrid, resample_hat  # noqa: E402,F401
from .local_arr import LocalPatch3D, PatchRegion, locate_hot_region, local_high_k_activity  # noqa: E402,F401
