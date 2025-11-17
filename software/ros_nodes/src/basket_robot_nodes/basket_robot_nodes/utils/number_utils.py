def clip_int16(v: int) -> int:
    return max(-32768, min(32767, int(v)))


def clip_uint16(v: int) -> int:
    return max(0, min(65535, int(v)))


class FrameStabilityCounter:
    def __init__(self, threshold: int) -> None:
        self.threshold = threshold
        self.count = 0

    def update(self, condition: bool) -> bool:
        """Returns True once condition has been True for `threshold` consecutive frames."""
        if condition:
            self.count += 1
            if self.count >= self.threshold:
                self.count = 0
                return True
        else:
            self.count = 0
        return False
