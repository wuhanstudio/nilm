from pathlib import Path
from string import Template
from tsetlin.compiler.size import SIZE_TSETLIN
from tsetlin.compiler.size import SizeCounter
from tsetlin.compiler.clause import emit_clause_arrays, emit_clause_table
from tsetlin.compiler.clause_compressed import emit_clausec_arrays, emit_clausec_table

import tsetlin.tsetlin_pb2 as tsetlin_pb2

TEMPLATE_DIR = Path(__file__).parent / "template"

def load_struct_template(path, **types):
    text = Path(path).read_text()
    return Template(text).substitute(**types)

def tsetlin_compile(model_path, out_path):
    target_dir = Path(model_path)

    m_tsetlin = tsetlin_pb2.Tsetlin()
    m_tsetlin.ParseFromString(target_dir.read_bytes())

    size_counter = SizeCounter()

    out = []

    if m_tsetlin.model_type == tsetlin_pb2.ModelType.INFERENCE:
        out.append(load_struct_template(
            TEMPLATE_DIR / "tsetlin_structs.h.in",
            COMPRESS_TYPE="const uint16_t",
            COMPRESS_TYPE_DATA="const uint16_t",
            OFF_TYPE="const uint16_t",
            OFF_TYPE_DATA="const uint16_t",
            BITPACK_TYPE="const uint32_t",
            BITPACK_TYPE_DATA="const uint32_t",
        ))
    else:
        out.append(load_struct_template(
            TEMPLATE_DIR / "tsetlin_structs.h.in",
            COMPRESS_TYPE="const uint16_t",
            COMPRESS_TYPE_DATA="uint16_t",
            OFF_TYPE="const uint16_t",
            OFF_TYPE_DATA="uint16_t",
            BITPACK_TYPE="const uint32_t",
            BITPACK_TYPE_DATA="uint32_t",
        ))

    Path("tsetlin_model.h").write_text("\n".join(out))

    out = []
    out.append(f"""
/* Auto-generated Tsetlin Machine model header file. */
#include "tsetlin_model.h"

#ifdef __AVR__
#include <avr/pgmspace.h>
#define FLASH PROGMEM
#else
#define FLASH
#endif
    """)

    inference = m_tsetlin.model_type == tsetlin_pb2.ModelType.INFERENCE
    out.append(emit_clause_arrays(m_tsetlin.clauses, size_counter))
    out.append(emit_clause_table(m_tsetlin.clauses, size_counter))

    out.append(emit_clausec_arrays(m_tsetlin.clauses_compressed, size_counter, inference=inference))
    out.append(emit_clausec_table(m_tsetlin.clauses_compressed, size_counter))

    size_counter.add_struct_array(1, SIZE_TSETLIN)

    trainable = f"""
#if !defined(TSETLIN_MODEL_TRAINABLE) 
#define TSETLIN_MODEL_TRAINABLE 
#endif
""" if m_tsetlin.model_type != tsetlin_pb2.ModelType.INFERENCE else f"""
#pragma message(\"Inference-Only Model\")
    """

    out.append(f"""
#define TSETLIN_MODEL_TOTAL_BYTES {size_counter.bytes}

{trainable}

static Tsetlin tsetlin_model = {{
    .n_class = {m_tsetlin.n_class},
    .n_feature = {m_tsetlin.n_feature},
    .n_clause = {m_tsetlin.n_clause},
    .n_state = {m_tsetlin.n_state},
    .model_type = (ModelType){m_tsetlin.model_type},

    .clauses = clauses,
    .clauses_compressed = clauses_compressed,
}};
    """)

    Path(out_path).write_text("\n".join(out))

    print(f"Written {out_path}")
    print(f"Total model size: {size_counter.bytes} bytes ({size_counter.bytes / 1024:.2f} KiB)")
