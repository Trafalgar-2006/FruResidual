"""Session-wide test setup.

KMP_DUPLICATE_LIB_OK must be set before torch is imported anywhere in the
process (see fruresidual.models.gnn for why) -- setting it here too,
before pytest imports any test module, is a safety net in case a test
imports torch before fruresidual.models.gnn does.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
