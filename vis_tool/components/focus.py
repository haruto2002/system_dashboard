from dataclasses import dataclass

import cv2
import numpy as np

from vis_tool.components.utils import MapConfig


@dataclass
class FocusAreaCandidate:
    place: str
    y: int
    x: int
    score: float

    def _get_area(
        self,
        map_top_left_x: int,
        map_top_left_y: int,
        rect_h: int,
        rect_w: int,
        margin_size_x: int,
        margin_size_y: int,
        hm_grid_size: int,
        back_img_scale: float,
    ) -> tuple[int, int, int, int]:
        l, t, r, b = (
            self.x - margin_size_x,
            self.y - margin_size_y,
            self.x + rect_w + margin_size_x,
            self.y + rect_h + margin_size_y,
        )
        l_on_back_img = (map_top_left_x + l * hm_grid_size) * back_img_scale
        t_on_back_img = (map_top_left_y + t * hm_grid_size) * back_img_scale
        r_on_back_img = (map_top_left_x + r * hm_grid_size) * back_img_scale
        b_on_back_img = (map_top_left_y + b * hm_grid_size) * back_img_scale

        return (l_on_back_img, t_on_back_img, r_on_back_img, b_on_back_img)


class FocusAreaSelector:
    def __init__(
        self,
        rect_h: int,
        rect_w: int,
        K: int,
        iou_thresh: float = 0.0,
        suppress_across_places: bool = False,
    ):
        self.rect_h = rect_h
        self.rect_w = rect_w
        self.K = K
        self.iou_thresh = iou_thresh
        self.suppress_across_places = suppress_across_places

    def _iou_same_size(self, y1, x1, y2, x2, h, w):
        inter_h = max(0, min(y1 + h, y2 + h) - max(y1, y2))
        inter_w = max(0, min(x1 + w, x2 + w) - max(x1, x2))
        inter = inter_h * inter_w

        area = h * w
        union = 2 * area - inter
        iou = inter / union if union > 0 else 0.0
        return iou

    def _nms_rects(self, candidates: list[FocusAreaCandidate]):
        selected: list[FocusAreaCandidate] = []

        for candidate in candidates:
            keep = True

            for sel_candidate in selected:
                if (
                    not self.suppress_across_places
                    and candidate.place != sel_candidate.place
                ):
                    continue

                if (
                    self._iou_same_size(
                        candidate.y,
                        candidate.x,
                        sel_candidate.y,
                        sel_candidate.x,
                        self.rect_h,
                        self.rect_w,
                    )
                    > self.iou_thresh
                ):
                    keep = False
                    break

            if keep:
                selected.append(candidate)

            if len(selected) >= self.K:
                break

        return selected

    def _rect_sum_map(self, heatmap: np.ndarray):
        H, W = heatmap.shape

        if self.rect_h > H or self.rect_w > W:
            return None

        S = (
            np.pad(heatmap, ((1, 0), (1, 0)), mode="constant")
            .cumsum(axis=0)
            .cumsum(axis=1)
        )

        return (
            S[self.rect_h :, self.rect_w :]
            - S[: -self.rect_h, self.rect_w :]
            - S[self.rect_h :, : -self.rect_w]
            + S[: -self.rect_h, : -self.rect_w]
        )

    def select_topk(
        self,
        heatmap_dict: dict[str, np.ndarray],
    ) -> list[FocusAreaCandidate]:
        """
        heatmap_dict:
            {place: heatmap} の辞書。
            各 heatmap の shape は違っていてもよい。

        return:
            selected_candidates: list[FocusAreaCandidate]
        """

        places = list(heatmap_dict.keys())
        candidates: list[FocusAreaCandidate] = []

        for place in places:
            heatmap = heatmap_dict[place]

            if heatmap.ndim != 2:
                raise ValueError(
                    f"{place}: heatmap must be 2D, got shape {heatmap.shape}"
                )

            rect_sums = self._rect_sum_map(heatmap)

            if rect_sums is None:
                continue

            flat = rect_sums.ravel()
            order = np.argsort(flat)[::-1]

            ys, xs = np.unravel_index(order, rect_sums.shape)

            for y, x, score in zip(ys, xs, flat[order]):
                candidates.append(
                    FocusAreaCandidate(place, int(y), int(x), float(score))
                )

        candidates.sort(key=lambda t: t.score, reverse=True)

        selected_candidates = self._nms_rects(candidates)

        return selected_candidates


class FocusAreaDrawer:
    def __init__(
        self,
        rect_h: int,
        rect_w: int,
        margin_scale: float,
        grid_size: int,
        back_img_scale: float,
        focus_colors: list[tuple[int, int, int]],
        map_config_dict: dict[str, MapConfig] | None = None,
    ):
        self.rect_h = rect_h
        self.rect_w = rect_w
        self.margin_size_x, self.margin_size_y = self._get_margin_size(
            rect_h, rect_w, margin_scale
        )

        self.grid_size = grid_size
        self.back_img_scale = back_img_scale
        self.focus_colors = focus_colors
        self.map_config_dict = map_config_dict

        self.each_place_line_width = 4
        self.all_place_line_width = 7

    def _get_margin_size(self, rect_h: int, rect_w: int, margin_scale: float):
        rect_h_with_margin = int(rect_h * margin_scale)
        rect_w_with_margin = int(rect_w * margin_scale)
        margin_size_y = (rect_h_with_margin - rect_h) // 2
        margin_size_x = (rect_w_with_margin - rect_w) // 2
        return margin_size_x, margin_size_y

    def draw_focus_areas_on_each_place(
        self,
        focus_areas: list[FocusAreaCandidate],
        back_img_dict: dict[str, np.ndarray],
    ):
        img_dict_with_focus_areas = back_img_dict.copy()
        map_top_left_x, map_top_left_y = 0, 0
        for i, focus_area in enumerate(focus_areas):
            place = focus_area.place
            l, t, r, b = focus_area._get_area(
                map_top_left_x,
                map_top_left_y,
                self.rect_h,
                self.rect_w,
                self.margin_size_x,
                self.margin_size_y,
                self.grid_size,
                self.back_img_scale,
            )
            cv2.rectangle(
                img_dict_with_focus_areas[place],
                (int(l), int(t)),
                (int(r), int(b)),
                self.focus_colors[i],
                self.each_place_line_width,
            )
        return img_dict_with_focus_areas

    def draw_focus_areas_on_all_place(
        self,
        focus_areas: list[FocusAreaCandidate],
        back_img: np.ndarray,
    ) -> tuple[np.ndarray, list[tuple[str, np.ndarray, float]]]:

        img_with_focus_areas = back_img.copy()
        topk = []
        for i, focus_area in enumerate(focus_areas):
            place = focus_area.place
            map_config = self.map_config_dict[place]
            map_top_left_x = map_config.all_map_left_top_coor[0]
            map_top_left_y = map_config.all_map_left_top_coor[1]
            l, t, r, b = focus_area._get_area(
                map_top_left_x,
                map_top_left_y,
                self.rect_h,
                self.rect_w,
                self.margin_size_x,
                self.margin_size_y,
                self.grid_size,
                self.back_img_scale,
            )
            cv2.rectangle(
                img_with_focus_areas,
                (int(l), int(t)),
                (int(r), int(b)),
                self.focus_colors[i],
                self.all_place_line_width,
            )
            cropped_img = img_with_focus_areas[int(t) : int(b), int(l) : int(r)]
            topk.append((place, cropped_img, focus_area.score))

        return img_with_focus_areas, topk
