# ============================================================
# ABI CONFIGURATION
# ============================================================

UINT16_SIZE = 2
UINT32_SIZE = 4
PTR_SIZE = 4   # change to 8 for 64-bit targets

# ============================================================
# SIZE COUNTER
# ============================================================

SIZE_TSETLIN = (
    4 * 4 +     # n_class, n_feature, n_clause, n_state
    4 +         # model_type
    (PTR_SIZE + 4) * 4
)

class SizeCounter:
    def __init__(self):
        self.bytes = 0

    def add_u16_array(self, n):
        self.bytes += n * UINT16_SIZE

    def add_u32_array(self, n):
        self.bytes += n * UINT32_SIZE

    def add_struct_array(self, n, struct_size):
        self.bytes += n * struct_size
