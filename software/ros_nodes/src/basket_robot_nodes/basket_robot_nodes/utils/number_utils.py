def clip_int16(v: int) -> int:
    return max(-32768, min(32767, int(v)))


def clip_uint16(v: int) -> int:
    return max(0, min(65535, int(v)))


def thrower_clip(v: int) -> int:
    if v < 0:
        return 0
    if v > 100:
        return 100
    return v
