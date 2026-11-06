"""Pin BLAS/OpenMP threads before numeric libraries spawn workers.

``RandomizedSearchCV`` and ``GridSearchCV`` use ``n_jobs=-1``. Leaving one
BLAS thread per process avoids oversubscribing the machine. This module must
be imported before numpy or scikit-learn.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
