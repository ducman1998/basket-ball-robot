import json
from typing import Dict, List, Optional, Tuple, Union


class GreenBall:
    def __init__(
        self,
        center: Tuple[int, int],
        radius: Union[int, float],
        area: float,
        position_2d: Union[List[float], Tuple[float, float]],
    ) -> None:
        self.center = center
        self.radius = radius
        self.area = area
        self.position_2d = position_2d

    def to_dict(self) -> Dict:
        return {
            "center": self.center,
            "radius": self.radius,
            "area": self.area,
            "position_2d": self.position_2d,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "GreenBall":
        return cls(
            center=data["center"],
            radius=data["radius"],
            area=data["area"],
            position_2d=data["position_2d"],
        )


class ImageInfo:
    def __init__(
        self,
        balls: Union[List[GreenBall], Tuple[GreenBall]],
        court_center: Optional[Tuple[float, float]] = None,  # in 2d robot base footprint frame
        court_area: Optional[float] = None,
    ) -> None:
        self.balls = balls
        self.court_center = court_center
        self.court_area = court_area

    def to_dict(self) -> Dict:
        return {
            "balls": [ball.to_dict() for ball in self.balls],
            "court_center": self.court_center,
            "court_area": self.court_area,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict) -> "ImageInfo":
        balls = [GreenBall.from_dict(b) for b in data.get("balls", [])]
        court_center = data.get("court_center", None)
        court_area = data.get("court_area", None)
        if court_center is not None:
            court_center = tuple(court_center)
        return cls(balls=balls, court_center=court_center, court_area=court_area)

    @classmethod
    def from_json(cls, json_str: str) -> "ImageInfo":
        data = json.loads(json_str)
        return cls.from_dict(data)
