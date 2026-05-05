from tsetlin.compiler.size import PTR_SIZE
from tsetlin.compiler.array import emit_uint16_array

# ============================================================
# CLAUSE COMPRESSED
# ============================================================

SIZE_CLAUSE_COMPRESSED = (
    2 + 2 +
    PTR_SIZE +
    PTR_SIZE
)

def emit_clausec_arrays(clauses, size, inference=False):
    out = []
    for i, c in enumerate(clauses):
        out.append(emit_uint16_array(f"clausec_{i}_pos", c.position, const=True))
        out.append(emit_uint16_array(f"clausec_{i}_data", c.data, const=inference))
        size.add_u16_array(len(c.position))
        size.add_u16_array(len(c.data))
    return "\n".join(out)

def emit_clausec_table(clauses, size):
    size.add_struct_array(len(clauses), SIZE_CLAUSE_COMPRESSED)
    out = ["static const FLASH ClauseCompressed clauses_compressed[] = {"]
    for i, c in enumerate(clauses):
        out.append(
            f"    {{ {c.n_pos_literal}, {c.n_neg_literal}, "
            f"clausec_{i}_pos, "
            f"clausec_{i}_data }},"
        )
    out.append("};\n")
    return "\n".join(out)
