import json
from typing import Dict, List, Optional, Tuple, Union


class GreenBall:
    def __init__(
        self,
        center: Tuple[int, int],
        radius: Union[int, float],
        area: float,
        position_2d: Union[List[float], Tuple[float, float]],
        inside: bool = True,
    ) -> None:
        """
        Inputs:
            center: center of the green ball in image coordinates (pixels)
            radius: radius of the green ball in image (pixels)
            area: area of the green ball in image (pixels)
            position_2d: 2D position of the green ball in robot base footprint frame (mm)
            inside: whether the ball is inside the court area
        """
        self.center = center
        self.radius = radius
        self.area = area
        self.position_2d = position_2d
        self.inside = inside

    def to_dict(self) -> Dict:
        return {
            "center": [int(self.center[0]), int(self.center[1])],
            "radius": float(self.radius),
            "area": float(self.area),
            "position_2d": [float(self.position_2d[0]), float(self.position_2d[1])],
            "inside": bool(self.inside),
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


class Marker:
    def __init__(
        self,
        id: int,
        position_2d: Tuple[float, float],
        theta: float,
    ) -> None:
        """
        Inputs:
            id: marker ID
            position_2d: 2D position of the marker in robot base footprint frame (mm)
            theta: orientation of the marker in degrees
        """
        self.id = id
        self.position_2d = position_2d
        self.theta = theta

    def to_dict(self) -> Dict:
        return {
            "id": int(self.id),
            "position_2d": [float(self.position_2d[0]), float(self.position_2d[1])],
            "theta": float(self.theta),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Marker":
        return cls(**data)


class ImageInfo:
    def __init__(
        self,
        image_size: Tuple[int, int],
        balls: Union[List[GreenBall], Tuple[GreenBall]],
        markers: List[Marker],
        basket: Optional[Basket] = None,
    ) -> None:
        """
        Inputs:
            image_size: size of the image (width, height) in pixels
            balls: list or tuple of detected GreenBall objects
            markers: list of detected Marker objects
            basket: detected Basket object (optional)
        """
        self.image_size = image_size
        self.balls = balls
        self.markers = markers
        self.basket = basket

    def to_dict(self) -> Dict:
        return {
            "image_size": ([int(self.image_size[0]), int(self.image_size[1])]),
            "balls": [ball.to_dict() for ball in self.balls],
            "markers": [marker.to_dict() for marker in self.markers],
            "basket": self.basket.to_dict() if self.basket else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict, ignore_outside: bool = True) -> "ImageInfo":
        image_size = tuple(data.get("image_size", (1280, 720)))
        balls = [GreenBall.from_dict(b) for b in data.get("balls", [])]
        balls = [
            b for b in balls if (b.inside or not ignore_outside)
        ]  # filter outside balls if needed
        markers = [Marker.from_dict(m) for m in data.get("markers", [])]
        basket_data = data.get("basket", None)
        basket = Basket.from_dict(basket_data) if basket_data else None
        return cls(image_size=image_size, balls=balls, markers=markers, basket=basket)

    @classmethod
    def from_json(cls, json_str: str) -> "ImageInfo":
        data = json.loads(json_str)
        return cls.from_dict(data)
