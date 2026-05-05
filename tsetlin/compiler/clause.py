from tsetlin.compiler.size import PTR_SIZE
from tsetlin.compiler.array import emit_uint32_array

# ============================================================
# CLAUSE
# ============================================================

SIZE_CLAUSE = PTR_SIZE + 4

def emit_clause_arrays(clauses, size):
    out = []
    for i, c in enumerate(clauses):
        out.append(emit_uint32_array(f"clause_{i}_data", c.data))
        size.add_u32_array(len(c.data))
    return "\n".join(out)

def emit_clause_table(clauses, size):
    size.add_struct_array(len(clauses), SIZE_CLAUSE)
    out = ["static Clause clauses[] = {"]
    for i, c in enumerate(clauses):
        out.append(f"    {{ clause_{i}_data }},")
    out.append("};\n")
    return "\n".join(out)
