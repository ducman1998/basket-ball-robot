import json
from typing import Dict, List, Tuple, Union


class GreenBall:
    def __init__(
        self,
        center: Tuple[int, int],
        radius: Union[int, float],
        position_2d: Union[List[float], Tuple[float, ...]],
    ) -> None:
        self.center = center
        self.radius = radius
        self.position_2d = position_2d

    def to_dict(self) -> Dict:
        return {
            "center": self.center,
            "radius": self.radius,
            "position_2d": self.position_2d,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "GreenBall":
        return cls(
            center=data["center"],
            radius=data["radius"],
            position_2d=data["position_2d"],
        )


class ImageInfo:
    def __init__(self, detected_balls: Union[List[GreenBall], Tuple[GreenBall]]) -> None:
        self.detected_balls = detected_balls

    def to_dict(self) -> Dict:
        return {"detected_balls": [ball.to_dict() for ball in self.detected_balls]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict) -> "ImageInfo":
        balls = [GreenBall.from_dict(b) for b in data.get("detected_balls", [])]
        return cls(detected_balls=balls)

    @classmethod
    def from_json(cls, json_str: str) -> "ImageInfo":
        data = json.loads(json_str)
        return cls.from_dict(data)
