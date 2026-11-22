"""Run the lightweight host for the browser-computed CreditView application."""
from __future__ import annotations

import os

# Keep numerical-library worker pools small on memory-constrained web instances.
# Set these before importing the model-serving stack so OpenMP/BLAS allocate
# their runtime state with one worker. Explicit deployment values remain usable.
for _thread_env in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import uvicorn

from frontend_host import app


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
