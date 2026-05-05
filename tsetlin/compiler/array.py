# ============================================================
# CODE EMISSION HELPERS
# ============================================================

def emit_uint16_array(name, values, const=False):
    body = ", ".join(str(v) for v in values)
    const_str = "const" if const else ""
    flash_str = "FLASH" if const else ""
    return f"static {const_str} {flash_str} uint16_t {name}[] = {{ {body} }};\n"

def emit_uint32_array(name, values, const=False):
    body = ", ".join(str(v) for v in values)
    const_str = "const" if const else ""
    flash_str = "FLASH" if const else ""
    return f"static {const_str} {flash_str} uint32_t {name}[] = {{ {body} }};\n"
