from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass
class MapConfig:
    path2homography_matrix: Path
    original_map_size: list[float] | None = None
    all_map_left_top_coor: list[float] = field(default_factory=lambda: [0.0, 0.0])
    scale: float = 1.0
    homography_matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.homography_matrix = np.loadtxt(self.path2homography_matrix)

    def to_map_coords(self, points: np.ndarray) -> np.ndarray:
        bev_points = project_points(points, self.homography_matrix)
        return bev_points * self.scale + np.array(self.all_map_left_top_coor)


def project_points(points: np.ndarray, homography_matrix: np.ndarray) -> np.ndarray:
    points = cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography_matrix)
    return points.reshape(-1, 2)


def load_map_config(path2cfg: Path) -> MapConfig:
    with path2cfg.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return MapConfig(**cfg)


class TrackPreparer:
    def __init__(self, map_config_dict: dict[str, MapConfig]):
        self.map_config_dict = map_config_dict
        self.places = set(map_config_dict.keys())

    def set_track_data(
        self, target_data: dict[str, list[dict]]
    ) -> dict[str, dict[str, dict[int, np.ndarray]]]:
        assert set(target_data.keys()).issubset(self.places), (
            f"target_data keys ({target_data.keys()}) are not a subset of expected places ({self.places})"
        )
        return {
            place: self.set_place_track_data(
                self.map_config_dict[place], target_data[place]
            )
            for place in target_data.keys()
        }

    def set_place_track_data(
        self, map_config: MapConfig, target_data: list[dict]
    ) -> dict[str, dict[int, np.ndarray]]:
        point_existing_ids, box_existing_ids = self.get_existing_ids(target_data)
        track_data: dict[str, dict[int, list]] = {
            "point": {id: [] for id in point_existing_ids},
            "bbox": {id: [] for id in box_existing_ids},
        }
        for data in target_data:
            point_data = data["objects"]["point"]
            box_data = data["objects"]["bbox"]
            for point in point_data:
                if point["id"] in point_existing_ids:
                    x, y = point["x"], point["y"]
                    coord = [x, y]
                    track_data["point"][point["id"]].append(coord)
            for box in box_data:
                if box["id"] in box_existing_ids:
                    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                    coord = [(x1 + x2) / 2, (y1 + y2) / 2]
                    track_data["bbox"][box["id"]].append(coord)

        track_data_with_array: dict[str, dict[int, np.ndarray]] = {
            "point": {
                id: map_config.to_map_coords(np.array(points))
                for id, points in track_data["point"].items()
            },
            "bbox": {
                id: map_config.to_map_coords(np.array(boxes))
                for id, boxes in track_data["bbox"].items()
            },
        }
        return track_data_with_array

    def get_existing_ids(self, target_data: list[dict]) -> tuple[set[int], set[int]]:
        latest_data = target_data[-1]
        point_existing_ids = set()
        bbox_existing_ids = set()
        for obj in latest_data["objects"]["point"]:
            point_existing_ids.add(obj["id"])
        for obj in latest_data["objects"]["bbox"]:
            bbox_existing_ids.add(obj["id"])
        return point_existing_ids, bbox_existing_ids
