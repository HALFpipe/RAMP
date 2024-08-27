try:
    from pytest_cov.embed import cleanup_on_sigterm
except ImportError:
    pass
else:
    cleanup_on_sigterm()

import os

os.environ["JAX_PLATFORMS"] = "cpu"
