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
        """
        Inputs:
            center: center of the green ball in image coordinates (pixels)
            radius: radius of the green ball in image (pixels)
            area: area of the green ball in image (pixels)
            position_2d: 2D position of the green ball in robot base footprint frame (mm)
        """
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


class Basket:
    def __init__(
        self,
        color: str,
        center: Tuple[int, int],
        position_2d: Optional[Tuple[float, float]],
        area: int,
    ) -> None:
        """
        Inputs:
            color: color of the basket detected
            center: center of the basket in image coordinates (pixels)
            area: area of the basket in image (pixels)
            position_2d: 2D position of the basket in robot base footprint frame (mm)

        Note: position_2d: in 2d robot base footprint frame (mm). Can be None if the basket position
              is out the overlapping region of both cameras.
              However, the basket center in image coordinates and area are always provided.
        """
        self.color = color
        self.center = center
        self.position_2d = position_2d
        self.area = area

    def to_dict(self) -> Dict:
        return {
            "color": self.color,
            "center": self.center,
            "position_2d": self.position_2d,
            "area": self.area,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Basket":
        return cls(
            color=data["color"],
            center=tuple(data["center"]),
            position_2d=tuple(data["position_2d"]) if data["position_2d"] is not None else None,
            area=data["area"],
        )


class ImageInfo:
    def __init__(
        self, balls: Union[List[GreenBall], Tuple[GreenBall]], basket: Optional[Basket] = None
    ) -> None:
        """
        Inputs:
            balls: list of detected green balls
            basket: detected basket (can be None if no basket detected)
            court_center: 2D position of the court center in robot base footprint frame (mm)
            court_area: area of the court in image (pixels)

        Note: court_center can be None if the court is not detected.
              basket can be None if no basket is detected.
        """
        self.balls = balls
        self.basket = basket

    def to_dict(self) -> Dict:
        return {
            "balls": [ball.to_dict() for ball in self.balls],
            "basket": self.basket.to_dict() if self.basket else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict) -> "ImageInfo":
        balls = [GreenBall.from_dict(b) for b in data.get("balls", [])]
        basket_data = data.get("basket", None)
        basket = Basket.from_dict(basket_data) if basket_data else None
        return cls(balls=balls, basket=basket)

    @classmethod
    def from_json(cls, json_str: str) -> "ImageInfo":
        data = json.loads(json_str)
        return cls.from_dict(data)
