import tsetlin.compiler

from tsetlin.compiler.size import SIZE_TSETLIN
from tsetlin.compiler.array import emit_uint16_array, emit_uint32_array
from tsetlin.compiler.clause import emit_clause_arrays, emit_clause_table
from tsetlin.compiler.clause_compressed import emit_clausec_arrays, emit_clausec_table
