from pathlib import Path

import numpy as np

from vis_tool.components.focus import FocusAreaDrawer, FocusAreaSelector
from vis_tool.components.graph import GraphGenerator
from vis_tool.components.risk import CrowdRiskScore
from vis_tool.components.risk_vis import HeatmapDisplayer
from vis_tool.components.trajectory import MapConfig, MapDisplayer, load_draw_style
from vis_tool.components.utils import TrackPreparer, load_map_config

TrackData = dict[str, dict[int, np.ndarray]]


class Dashboard:
    def __init__(
        self,
        places: list[str],
        map_config_dir: str,
        map_img_file: str,
        map_draw_style_file: str,
    ):
        self.map_config_dir = map_config_dir
        self.places = places
        self.map_config_dict = {
            place: load_map_config(Path(map_config_dir) / f"{place}.yaml")
            for place in places
        }

        self.track_preparer = TrackPreparer(self.map_config_dict)

        self.trajectory_displayer, self.img_scale = self._set_trajectory_displayer(
            map_img_file,
            self.map_config_dict,
            map_draw_style_file,
        )

        grid_size = 5
        vec_span = 10
        R = 10
        mode = "crs++"
        self.risk_scorer_dict = self._set_risk_scorer(vec_span, grid_size, R, mode)

        max_score = 10**-5
        min_score = 0.0
        resize_method = "nearest"
        self.heatmap_displayer_dict = self._set_heatmap_displayer(
            grid_size, max_score, min_score, resize_method
        )

        self.graph_generator = self._set_graph_generator()

        focus_area_rect_h = 5
        focus_area_rect_w = 5
        focus_area_top_k = 3
        focus_area_margin_scale = 1.5
        focus_area_iou_thresh = 0.0
        focus_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]
        if len(focus_colors) != focus_area_top_k:
            raise ValueError(
                f"focus_colors length must be equal to focus_area_top_k, got {len(focus_colors)} and {focus_area_top_k}"
            )
        self.focus_area_selector = FocusAreaSelector(
            rect_h=focus_area_rect_h,
            rect_w=focus_area_rect_w,
            K=focus_area_top_k,
            iou_thresh=focus_area_iou_thresh,
            suppress_across_places=False,
        )
        self.focus_area_drawer = FocusAreaDrawer(
            rect_h=focus_area_rect_h,
            rect_w=focus_area_rect_w,
            margin_scale=focus_area_margin_scale,
            grid_size=grid_size,
            back_img_scale=self.img_scale,
            focus_colors=focus_colors,
            map_config_dict=self.map_config_dict,
        )

        self.focus_area_update_frequency = 3
        self.accumulated_risk_score_map_dict = {}
        self.current_frame_num = 0
        self.focus_areas = []

    def _set_trajectory_displayer(
        self,
        map_img_file: str,
        map_config_dict: dict[str, MapConfig],
        map_draw_style_file: str,
    ) -> MapDisplayer:
        map_draw_style = load_draw_style(Path(map_draw_style_file))

        trajectory_displayer = MapDisplayer(
            path2all_map=map_img_file,
            map_config_dict=map_config_dict,
            main_draw_style=map_draw_style,
        )
        img_scale = map_draw_style.scale
        return trajectory_displayer, img_scale

    def _set_risk_scorer(
        self, vec_span: int = 10, grid_size: int = 5, R: float = 10, mode: str = "crs++"
    ) -> dict[str, CrowdRiskScore]:
        risk_scorer_dict = {
            place: CrowdRiskScore(
                map_config=self.map_config_dict[place],
                vec_span=vec_span,
                grid_size=grid_size,
                R=R,
                mode=mode,
            )
            for place in self.places
        }
        return risk_scorer_dict

    def _set_heatmap_displayer(
        self,
        grid_size: int = 5,
        max_score: float = 10**-5,
        min_score: float = 0.0,
        resize_method: str = "nearest",
    ) -> dict[str, HeatmapDisplayer]:
        heatmap_displayer_dict = {
            place: HeatmapDisplayer(
                grid_size=grid_size,
                max_score=max_score,
                min_score=min_score,
                resize_method=resize_method,
            )
            for place in self.places
        }
        return heatmap_displayer_dict

    def _set_graph_generator(self) -> GraphGenerator:
        graph_generator = GraphGenerator()
        return graph_generator

    def display(self, data, timestamp: float):
        self.current_frame_num += 1
        # 軌跡データの用意
        track_data_dict = self.track_preparer.set_track_data(data)

        # 軌跡描画
        trajectory_img, place_trajectory_img_dict = self._draw_trajectory(
            track_data_dict
        )

        # 危険度の計算
        crs_map_dict = self._calc_risk_score(track_data_dict)

        # 危険度マップ描画
        heatmap_img_dict = self._draw_risk_heatmap(
            crs_map_dict,
            place_trajectory_img_dict,
        )

        # グラフ描画
        self.graph_generator.update(timestamp, data, crs_map_dict)
        graph_img_dict = self.graph_generator.generate_graph_imgs()

        # 注目領域計算用の危険度マップの更新
        for place, crs_map in crs_map_dict.items():
            if place not in self.accumulated_risk_score_map_dict:
                self.accumulated_risk_score_map_dict[place] = crs_map
            else:
                self.accumulated_risk_score_map_dict[place] += crs_map

        # 注目領域の決定（設定頻度ごとに更新）
        if self.current_frame_num % self.focus_area_update_frequency == 0:
            self.focus_areas = self.focus_area_selector.select_topk(
                self.accumulated_risk_score_map_dict,
            )
            self.accumulated_risk_score_map_dict = {}

        trajectory_img_with_focus_areas, focus_area_outputs = (
            self.focus_area_drawer.draw_focus_areas_on_all_place(
                self.focus_areas,
                trajectory_img,
            )
        )
        heatmap_img_dict_with_focus_areas = (
            self.focus_area_drawer.draw_focus_areas_on_each_place(
                self.focus_areas,
                heatmap_img_dict,
            )
        )

        return (
            trajectory_img_with_focus_areas,
            heatmap_img_dict_with_focus_areas,
            focus_area_outputs,
            graph_img_dict,
        )

    def _draw_trajectory(self, track_data_dict: dict[str, TrackData]):
        trajectory_img = self.trajectory_displayer.draw(track_data_dict)
        place_trajectory_img_dict = self.trajectory_displayer.crop_places(
            trajectory_img
        )
        return trajectory_img, place_trajectory_img_dict

    def _calc_risk_score(
        self,
        track_data_dict: dict[str, TrackData],
    ):
        crs_map_dict = {}
        for place, track_data in track_data_dict.items():
            scorer = self.risk_scorer_dict[place]
            crs_map = scorer.run(track_data["point"])
            crs_map = np.clip(crs_map, 0, None)
            crs_map_dict[place] = crs_map
        return crs_map_dict

    def _draw_risk_heatmap(
        self,
        crs_map_dict: dict[str, np.ndarray],
        place_trajectory_img_dict: dict[str, np.ndarray],
    ):
        heatmap_img_dict = {}
        for place, crs_map in crs_map_dict.items():
            heatmap_displayer = self.heatmap_displayer_dict[place]
            back_img = place_trajectory_img_dict[place]
            output_img = heatmap_displayer.draw_heatmap(back_img, crs_map)
            heatmap_img_dict[place] = output_img
        return heatmap_img_dict
