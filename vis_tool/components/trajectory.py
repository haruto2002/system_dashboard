from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from vis_tool.components.utils import MapConfig

TrackData = dict[str, dict[int, np.ndarray]]


@dataclass
class PointStyle:
    head_radius: int = 5
    trail_radius: int = 1


@dataclass
class BoxStyle:
    rect_width: int = 40
    rect_height: int = 20
    line_thickness: int = 2


@dataclass
class MapBorderStyle:
    draw: bool = False
    color: tuple[int, int, int] = (0, 0, 0)
    thickness: int = 4


@dataclass
class DrawStyle:
    scale: float = 1.0
    point_style: PointStyle = field(default_factory=PointStyle)
    box_style: BoxStyle = field(default_factory=BoxStyle)
    map_border_style: MapBorderStyle = field(default_factory=MapBorderStyle)


def load_draw_style(path2draw_style: Path) -> DrawStyle:
    with path2draw_style.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "map_border_style" in data:
        map_border_style_data = data["map_border_style"]
        map_border_style_data["color"] = tuple(map_border_style_data["color"])
        map_border_style = MapBorderStyle(**map_border_style_data)
    else:
        map_border_style = MapBorderStyle()

    return DrawStyle(
        scale=data["scale"],
        point_style=PointStyle(**data["point_style"]),
        box_style=BoxStyle(**data["box_style"]),
        map_border_style=map_border_style,
    )


def get_color(label_id: int) -> tuple[int, int, int]:
    return (
        int((37 * label_id) % 255),
        int((17 * label_id) % 255),
        int((29 * label_id) % 255),
    )


def create_object_rectangle(
    center_point: np.ndarray,
    angle: float,
    rect_width: int,
    rect_height: int,
) -> np.ndarray:
    rect_corners = np.array(
        [
            [-rect_height / 2, -rect_width / 2],
            [rect_height / 2, -rect_width / 2],
            [rect_height / 2, rect_width / 2],
            [-rect_height / 2, rect_width / 2],
        ]
    )
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rot_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    rotated_corners = rect_corners @ rot_matrix.T
    final_corners = rotated_corners + np.array([center_point[0], center_point[1]])
    return np.round(final_corners).astype(np.int32).reshape((-1, 1, 2))


def draw_tracks(
    output_img: np.ndarray,
    data: TrackData,
    draw_style: DrawStyle,
) -> np.ndarray:
    for label_id, points in data["point"].items():
        color = get_color(label_id)
        for point in points[:-1]:
            x, y = int(point[0] * draw_style.scale), int(point[1] * draw_style.scale)
            cv2.circle(
                output_img,
                (x, y),
                radius=int(draw_style.point_style.trail_radius),
                color=color,
                thickness=-1,
            )
        cv2.circle(
            output_img,
            (
                int(points[-1][0] * draw_style.scale),
                int(points[-1][1] * draw_style.scale),
            ),
            radius=int(draw_style.point_style.head_radius),
            color=color,
            thickness=-1,
        )

    for label_id, box_center_points in data["bbox"].items():
        color = get_color(label_id)
        for box_center_point in box_center_points[:-1]:
            x, y = (
                int(box_center_point[0] * draw_style.scale),
                int(box_center_point[1] * draw_style.scale),
            )
            cv2.circle(
                output_img,
                (x, y),
                radius=int(draw_style.point_style.trail_radius),
                color=color,
                thickness=-1,
            )
        angle = np.arctan2(
            box_center_points[-1][1] - box_center_points[0][1],
            box_center_points[-1][0] - box_center_points[0][0],
        )
        rect_corners = create_object_rectangle(
            box_center_points[-1] * draw_style.scale,
            angle,
            rect_width=int(draw_style.box_style.rect_width),
            rect_height=int(draw_style.box_style.rect_height),
        )
        cv2.polylines(
            output_img,
            [rect_corners],
            isClosed=True,
            color=color,
            thickness=int(draw_style.box_style.line_thickness),
        )

    return output_img


def draw_map_border(
    output_img: np.ndarray, map_config: MapConfig, draw_style: DrawStyle
) -> np.ndarray:
    if map_config.original_map_size is None:
        return output_img

    top_left = (
        int(map_config.all_map_left_top_coor[0] * draw_style.scale),
        int(map_config.all_map_left_top_coor[1] * draw_style.scale),
    )
    bottom_right = (
        int(
            top_left[0]
            + map_config.original_map_size[0] * map_config.scale * draw_style.scale
        ),
        int(
            top_left[1]
            + map_config.original_map_size[1] * map_config.scale * draw_style.scale
        ),
    )
    cv2.rectangle(
        output_img,
        top_left,
        bottom_right,
        draw_style.map_border_style.color,
        draw_style.map_border_style.thickness,
    )
    return output_img


class MapDisplayer:
    def __init__(
        self,
        path2all_map: str,
        map_config_dict: dict[str, MapConfig],
        main_draw_style: DrawStyle,
    ):
        all_map_img = cv2.imread(path2all_map)
        assert all_map_img is not None, (
            f"All map image file does not exist: {path2all_map}"
        )

        self.map_config_dict = map_config_dict
        self.main_draw_style = main_draw_style

        self.all_map_img = cv2.resize(
            all_map_img,
            (
                int(all_map_img.shape[1] * self.main_draw_style.scale),
                int(all_map_img.shape[0] * self.main_draw_style.scale),
            ),
        )

        self.places = set(map_config_dict.keys())

    def draw(self, track_data_dict: dict[str, TrackData]) -> np.ndarray:
        assert set(track_data_dict.keys()).issubset(self.places), (
            f"track_data_dict keys ({track_data_dict.keys()}) are not a subset of expected places ({self.places})"
        )

        output_img = self.all_map_img.copy()
        for place, data in track_data_dict.items():
            map_config = self.map_config_dict[place]
            output_img = draw_tracks(
                output_img,
                data,
                draw_style=self.main_draw_style,
            )
            if self.main_draw_style.map_border_style.draw:
                output_img = draw_map_border(
                    output_img, map_config, draw_style=self.main_draw_style
                )
        return output_img

    def crop_places(self, trajectory_img: np.ndarray) -> dict[str, np.ndarray]:
        place_trajectory_img_dict = {}
        for place, map_config in self.map_config_dict.items():
            top_left = (
                int(map_config.all_map_left_top_coor[0] * self.main_draw_style.scale),
                int(map_config.all_map_left_top_coor[1] * self.main_draw_style.scale),
            )

            bottom_right = (
                int(
                    top_left[0]
                    + map_config.original_map_size[0]
                    * map_config.scale
                    * self.main_draw_style.scale
                ),
                int(
                    top_left[1]
                    + map_config.original_map_size[1]
                    * map_config.scale
                    * self.main_draw_style.scale
                ),
            )
            place_trajectory_img = trajectory_img[
                top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]
            ]
            place_trajectory_img_dict[place] = place_trajectory_img
        return place_trajectory_img_dict
