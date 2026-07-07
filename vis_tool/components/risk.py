import numpy as np
from scipy.spatial.distance import cdist

from vis_tool.components.utils import MapConfig


class CrowdRiskScore:
    def __init__(
        self,
        map_config: MapConfig,
        vec_span: int,
        grid_size: int,
        R: float,
        mode: str = "crs",
    ):
        self.map_config = map_config
        self.vec_span = vec_span
        self.grid_size = grid_size
        self.R = R
        self.mode = mode

        self.left_top_coor = map_config.all_map_left_top_coor
        self.area_size = np.array(map_config.original_map_size) * map_config.scale
        self.area_x_size = int(self.area_size[0])
        self.area_y_size = int(self.area_size[1])
        self.x_grid_num = self.area_x_size // self.grid_size
        self.y_grid_num = self.area_y_size // self.grid_size

    def run(
        self,
        track_data: dict[int, np.ndarray],
    ):
        vec_data = self.generate_vec_data(track_data)
        crs_map = self.calc_map_score(vec_data)
        return crs_map

    def calc_map_score(self, vec_data: np.ndarray):
        crs_map = np.zeros((self.y_grid_num, self.x_grid_num))
        around_range = self.R * 3
        for i in range(self.y_grid_num):
            center_y_coor = self.left_top_coor[1] + self.grid_size * (i + 1 / 2)
            for j in range(self.x_grid_num):
                center_x_coor = self.left_top_coor[0] + self.grid_size * (j + 1 / 2)
                center_pos = np.array([center_x_coor, center_y_coor])
                around_data = vec_data[
                    (vec_data[:, 0, 0] < center_pos[0] + around_range)
                    & (vec_data[:, 0, 0] > center_pos[0] - around_range)
                    & (vec_data[:, 0, 1] < center_pos[1] + around_range)
                    & (vec_data[:, 0, 1] > center_pos[1] - around_range)
                ]

                if len(around_data) == 0:
                    crs_map[i, j] = 0.0
                    continue

                if self.mode == "crs":
                    crs = self.calc_crs(around_data, center_pos)
                elif self.mode == "crs+":
                    crs = self.calc_crs_with_opposing_crowd_presence(
                        around_data, center_pos
                    )
                elif self.mode == "crs++":
                    crs = self.calc_crs_with_opposing_crowd_presence_more_complex(
                        around_data, center_pos
                    )
                else:
                    raise ValueError(f"Invalid mode: {self.mode}")
                crs_map[i, j] = crs

        return crs_map

    # Calculate Final Score
    def calc_crs(self, around_data, center_pos):
        local_density, distance_decay = self.get_gaussian_kernel_density(
            center_pos, around_data[:, 0], R=self.R
        )
        div = self.calc_div(around_data, center_pos, distance_decay)
        return -div * local_density

    def calc_crs_with_opposing_crowd_presence(self, around_data, center_pos):
        local_density, distance_decay = self.get_gaussian_kernel_density(
            center_pos, around_data[:, 0], R=self.R
        )
        div = self.calc_div(around_data, center_pos, distance_decay)
        opposing_crowd_presence = self.calc_opposing_crowd_presence(
            around_data, center_pos
        )
        return -div * local_density * opposing_crowd_presence

    def calc_crs_with_opposing_crowd_presence_more_complex(
        self, around_data, center_pos
    ):
        local_density, distance_decay = self.get_gaussian_kernel_density(
            center_pos, around_data[:, 0], R=self.R
        )
        div = self.calc_div_with_opposing_crowd_presence(
            around_data, center_pos, distance_decay
        )
        return -div * local_density

    # Components
    def calc_div(self, around_data, center_pos, distance_decay):
        vx_list = []
        vy_list = []
        for pos, vec in around_data:
            vx, vy = vec[0], vec[1]
            if pos[0] < center_pos[0]:
                vx *= -1
            if pos[1] < center_pos[1]:
                vy *= -1
            vx_list.append(vx)
            vy_list.append(vy)
        vx_array = np.array(vx_list)
        vy_array = np.array(vy_list)
        decay_vx_array = vx_array * distance_decay
        decay_vy_array = vy_array * distance_decay
        div = np.sum(decay_vx_array) + np.sum(decay_vy_array)
        return div

    def calc_opposing_crowd_presence(self, around_data, center_pos):
        (
            left_local_density,
            right_local_density,
            up_local_density,
            down_local_density,
        ) = self.get_four_local_density(around_data, center_pos)
        opposing_crowd_presence = (
            left_local_density * right_local_density
            + up_local_density * down_local_density
        )
        return opposing_crowd_presence

    def calc_div_with_opposing_crowd_presence(
        self, around_data, center_pos, distance_decay
    ):
        (
            left_local_density,
            right_local_density,
            up_local_density,
            down_local_density,
        ) = self.get_four_local_density(around_data, center_pos)
        vx_list = []
        vy_list = []
        for pos, vec in around_data:
            vx, vy = vec[0], vec[1]
            if pos[0] < center_pos[0]:
                vx *= -1
                vx *= right_local_density
            else:
                vx *= left_local_density
            if pos[1] < center_pos[1]:
                vy *= -1
                vy *= down_local_density
            else:
                vy *= up_local_density
            vx_list.append(vx)
            vy_list.append(vy)
        vx_array = np.array(vx_list)
        vy_array = np.array(vy_list)
        decay_vx_array = vx_array * distance_decay
        decay_vy_array = vy_array * distance_decay
        div = np.sum(decay_vx_array) + np.sum(decay_vy_array)
        return div

    # Tools for Components
    def get_four_local_density(self, around_data, center_pos):
        left_around_data = around_data[around_data[:, 0, 0] < center_pos[0]]
        right_around_data = around_data[around_data[:, 0, 0] > center_pos[0]]
        up_around_data = around_data[around_data[:, 0, 1] < center_pos[1]]
        down_around_data = around_data[around_data[:, 0, 1] > center_pos[1]]
        left_local_density, left_distance_decay = self.get_gaussian_kernel_density(
            center_pos, left_around_data[:, 0], R=self.R
        )
        right_local_density, right_distance_decay = self.get_gaussian_kernel_density(
            center_pos, right_around_data[:, 0], R=self.R
        )
        up_local_density, up_distance_decay = self.get_gaussian_kernel_density(
            center_pos, up_around_data[:, 0], R=self.R
        )
        down_local_density, down_distance_decay = self.get_gaussian_kernel_density(
            center_pos, down_around_data[:, 0], R=self.R
        )

        return (
            left_local_density,
            right_local_density,
            up_local_density,
            down_local_density,
        )

    def get_gaussian_kernel_density(self, eval_point, positions, R):
        if len(positions) == 0:
            return 0, None
        if positions.ndim == 1:
            positions = np.expand_dims(positions, axis=0)

        distances = cdist(np.expand_dims(eval_point, axis=0), positions)

        kernel_vals = np.exp(-0.5 * (distances / R) ** 2) / (2 * np.pi * R**2)

        density = np.sum(kernel_vals[0])

        return density, kernel_vals[0]

    # Vector Data Preparation
    def generate_vec_data(
        self,
        track_data: dict[int, np.ndarray],
    ):
        """
        track_data: {id: [[x, y], [x, y], ...]}
        """

        vec_data = []
        for tracklet in track_data.values():
            pos, vec = self.get_vec(tracklet)
            vec_data.append([pos, vec])
        vec_data = np.array(vec_data)
        if vec_data.ndim != 3:
            raise ValueError(
                f"vec_data must be 3D array (n, 2, 2), got {vec_data.shape}"
            )

        return vec_data

    def get_vec(self, tracklet: np.ndarray):
        start_idx = max(0, len(tracklet) - self.vec_span)
        start_point = tracklet[start_idx]
        end_point = tracklet[-1]
        vec = (end_point - start_point) / (len(tracklet) - start_idx)
        pos = end_point
        return pos, vec
