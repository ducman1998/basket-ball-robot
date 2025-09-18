def clip_int16(v: int) -> int:
    return max(-32768, min(32767, int(v)))

def clip_uint16(v: int) -> int:
    return max(0, min(65535, int(v)))