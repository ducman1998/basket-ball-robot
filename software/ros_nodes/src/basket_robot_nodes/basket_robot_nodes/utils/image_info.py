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
            "center": [int(self.center[0]), int(self.center[1])],
            "radius": float(self.radius),
            "area": float(self.area),
            "position_2d": [float(self.position_2d[0]), float(self.position_2d[1])],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "GreenBall":
        return cls(**data)


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
        position_2d = (
            [float(self.position_2d[0]), float(self.position_2d[1])]
            if self.position_2d is not None
            else None
        )
        return {
            "color": self.color,
            "center": [int(self.center[0]), int(self.center[1])],
            "position_2d": position_2d,
            "area": int(self.area),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Basket":
        # Convert position_2d to tuple if it exists
        if data.get("position_2d") is not None:
            data = {**data, "position_2d": tuple(data["position_2d"])}
        return cls(**data)


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
