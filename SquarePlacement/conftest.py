"""One random seed for the whole test suite - every test draws from the same
deterministic sequence of the global random module, rather than each test
file seeding its own. RANDOM_SEED is applied once, here, at collection time
(before any test runs), not re-applied per test - a test that needs its own
random sequence isolated from whatever ran before it should seed explicitly
itself; this is deliberately just the one shared baseline the whole run
starts from.
"""

import random

RANDOM_SEED = 0
random.seed(RANDOM_SEED)
