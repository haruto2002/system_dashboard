import cv2
import numpy as np


class HeatmapDisplayer:
    def __init__(
        self,
        grid_size: int,
        max_score: float | None,
        min_score: float | None,
        resize_method: str = "linear",
    ):
        self.grid_size = grid_size
        self.max_score = max_score
        self.min_score = min_score
        self.resize_method = resize_method

    def draw_heatmap(
        self,
        back_img: np.ndarray,
        map_data: np.ndarray,
        vec_data: np.ndarray | None = None,
        grid: bool = False,
    ):
        height, width, _ = back_img.shape

        if self.max_score is None:
            self.max_score = np.max(map_data)
            print(f"max_score: {self.max_score}")
        if self.min_score is None:
            self.min_score = np.min(map_data)
        normalized_map_data = self.normalize_map_data(map_data)
        normalized_map_color_data = (normalized_map_data * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized_map_color_data, cv2.COLORMAP_JET)

        resized_heatmap = self.resize_map_data(heatmap, height, width)

        if vec_data is not None:
            self.draw_vec(back_img, vec_data)

        output = cv2.addWeighted(back_img, 0.5, resized_heatmap, 0.5, 0)

        if grid:
            grid_size = int(self.grid_size * (output.shape[0] / map_data.shape[0]))
            self.add_grid(output, grid_size)

        return output

    def resize_map_data(self, heatmap, height, width):
        if self.resize_method == "linear":
            resized_map_data = cv2.resize(
                heatmap,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

        elif self.resize_method == "nearest":
            resized_map_data = cv2.resize(
                heatmap,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            raise ValueError(f"Invalid resize method: {self.resize_method}")

        return resized_map_data

    def normalize_map_data(self, map_data):
        map_data[np.isnan(map_data)] = self.min_score
        clipped_map_data = np.clip(map_data, self.min_score, self.max_score)
        normalized_map_data = (clipped_map_data - self.min_score) / (
            self.max_score - self.min_score
        )
        return normalized_map_data

    def add_grid(self, back_img: np.ndarray, grid_size: int):
        height, width, _ = back_img.shape
        for i in range(height // grid_size):
            y_coor = i * grid_size
            for j in range(width // grid_size):
                x_coor = j * grid_size
                cv2.line(
                    back_img,
                    (x_coor, 0),
                    (x_coor, height),
                    (0, 0, 0),
                    1,
                )
                cv2.line(
                    back_img,
                    (0, y_coor),
                    (width, y_coor),
                    (0, 0, 0),
                    1,
                )

    # ベクトル表示用の機能だが、現状は使わない予定
    def draw_vec(self, back_img, vec_data):
        arrow_len = 5
        tipLength = 0.3
        arrow_scale = 50
        for x, y, vx, vy in vec_data:
            # ベクトルを圧縮・スケール
            vec = np.array([vx, vy], dtype=float)
            vec = self.compress_vec_data(vec) * arrow_scale

            # 開始点・終了点
            start = (int(x), int(y))
            end = (int(x + vec[0]), int(y + vec[1]))

            # 角度を計算（-π～+π）
            angle = np.arctan2(vec[1], vec[0])

            # Hue を 0–179 にマッピング
            hue = ((angle + np.pi) / (2 * np.pi) * 179).astype(int)
            # HSV 画像（1×1）を作って BGR に変換
            hsv_pixel = np.uint8([[[hue, 255, 255]]])  # H, S=255, V=255
            bgr_color = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0].tolist()

            # 矢印を描画
            cv2.arrowedLine(
                back_img,
                start,
                end,
                color=bgr_color,
                thickness=arrow_len,
                tipLength=tipLength,
            )

    def compress_vec_data(self, vel):
        norm = np.linalg.norm(vel)
        over_ratio = norm / 0.35
        if norm > 0.35:
            vel = vel / norm * 0.35 * (over_ratio * 0.5)
        return vel
