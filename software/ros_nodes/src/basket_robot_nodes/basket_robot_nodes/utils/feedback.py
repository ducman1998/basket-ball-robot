from typing import Literal

from .number_utils import clip_int16


class FeedbackSerial:
    def __init__(
        self,
        actual_s1: int,
        actual_s2: int,
        actual_s3: int,
        pos1: int,
        pos2: int,
        pos3: int,
        sensors: int,
        delimiter: int,
    ):
        self.actual_s1 = actual_s1
        self.actual_s2 = actual_s2
        self.actual_s3 = actual_s3
        self.pos1 = pos1
        self.pos2 = pos2
        self.pos3 = pos3
        self.sensors = sensors
        self.delimiter = delimiter

    def get_encoder_pos(self, encoder_index: Literal[1, 2, 3]):
        pos = None
        if encoder_index <= 1:
            pos = self.pos1
        elif encoder_index == 2:
            pos = self.pos2
        else:
            pos = self.pos3
        return clip_int16(pos)
