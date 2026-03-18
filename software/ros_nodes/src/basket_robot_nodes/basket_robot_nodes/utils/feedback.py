from dataclasses import dataclass


@dataclass(slots=True)  # Using slots=True for memory efficiency (good practice in Python 3.10+)
class FeedbackSerial:
    """
    Represents serial feedback data from multiple sensors, including actual readings,
    calculated positions, a sensor status, and a timestamp.
    """

    actual_s1: int
    actual_s2: int
    actual_s3: int
    pos1: int
    pos2: int
    pos3: int
    sensors: int
    delimiter: int
    timestamp: float = 0.0
