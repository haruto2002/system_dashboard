from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    }
)


class GraphGenerator:
    def __init__(self):
        self.num_people_df = pd.DataFrame()
        self.risk_score_df = pd.DataFrame()

        # サイズの設定
        self.width_px = 1920
        self.height_px = 1080
        self.dpi = 100
        self.fig_width = self.width_px / self.dpi
        self.fig_height = self.height_px / self.dpi
        self.axes_rect = [0.10, 0.15, 0.78, 0.72]

        # フォントサイズの設定
        self.tick_labelsize = 15
        self.xlabel_fontsize = 25
        self.ylabel_fontsize = 25
        self.title_fontsize = 40

        # グラフのスタイルの設定
        self.risk_score_color = "tab:red"
        self.num_people_color = "tab:green"
        self.linestyle = "-"
        self.linewidth = 3
        self.marker = "o"
        self.markersize = 5
        self.grid_linewidth = 1

        # グラフの表示範囲の設定
        self.ylim_min_scale = 0.7
        self.ylim_max_scale = 1.3

    def update(
        self, timestamp: float, mot_data_dict: dict, risk_score_map_dict: dict
    ) -> None:
        new_num_people_data = pd.DataFrame(
            {
                "timestamp": datetime.fromtimestamp(timestamp),
                **{
                    place: len(mot_data_dict[place][-1]["objects"]["point"])
                    for place in mot_data_dict.keys()
                },
            },
            index=[0],
        )
        new_risk_score_data = pd.DataFrame(
            {
                "timestamp": datetime.fromtimestamp(timestamp),
                **{
                    place: np.sum(risk_score_map_dict[place])
                    for place in risk_score_map_dict.keys()
                },
            },
            index=[0],
        )
        self.num_people_df = pd.concat(
            [self.num_people_df, new_num_people_data], ignore_index=True
        )
        self.risk_score_df = pd.concat(
            [self.risk_score_df, new_risk_score_data], ignore_index=True
        )
        self.num_people_df = self.num_people_df.sort_values(by="timestamp")
        self.risk_score_df = self.risk_score_df.sort_values(by="timestamp")

    def generate_graph_imgs(self) -> dict[str, np.ndarray]:
        places = self.num_people_df.columns.drop("timestamp")
        graph_imgs = {place: None for place in places}

        for place in places:
            fig = plt.figure(figsize=(self.fig_width, self.fig_height), dpi=self.dpi)

            # 左軸: number of people
            ax_left = fig.add_axes(self.axes_rect)

            # 右軸: risk score
            ax_right = ax_left.twinx()

            # number of people の折れ線（左軸）
            line_left = ax_left.plot(
                self.num_people_df["timestamp"],
                self.num_people_df[place],
                linestyle=self.linestyle,
                linewidth=self.linewidth * 2,
                marker=self.marker,
                markersize=self.markersize * 2,
                color=self.num_people_color,
                label="Number of People",
            )

            # risk score の折れ線（右軸）
            line_right = ax_right.plot(
                self.risk_score_df["timestamp"],
                self.risk_score_df[place],
                linestyle=self.linestyle,
                linewidth=self.linewidth,
                marker=self.marker,
                markersize=self.markersize,
                color=self.risk_score_color,
                label="Risk Score",
            )

            # 軸ラベル
            ax_left.set_xlabel("Date time", fontsize=self.xlabel_fontsize)
            ax_left.set_ylabel("Number of People", fontsize=self.ylabel_fontsize)
            ax_right.set_ylabel("Risk Score", fontsize=self.ylabel_fontsize)

            # tick サイズ
            ax_left.tick_params(axis="both", labelsize=self.tick_labelsize)
            ax_right.tick_params(axis="y", labelsize=self.tick_labelsize)

            # x軸の範囲
            x_min = min(
                self.risk_score_df["timestamp"].min(),
                self.num_people_df["timestamp"].min(),
            )
            x_max = max(
                self.risk_score_df["timestamp"].max(),
                self.num_people_df["timestamp"].max(),
            )
            ax_left.set_xlim(x_min, x_max)

            # 左軸: number of people の y 範囲
            people_min = self.num_people_df[place].min()
            people_max = self.num_people_df[place].max()
            ax_left.set_ylim(
                people_min * self.ylim_min_scale,
                people_max * self.ylim_max_scale,
            )

            # 右軸: risk score の y 範囲
            risk_min = self.risk_score_df[place].min()
            risk_max = self.risk_score_df[place].max()
            ax_right.set_ylim(
                risk_min * self.ylim_min_scale,
                risk_max * self.ylim_max_scale,
            )

            # グリッドは左軸基準だけにする
            ax_left.grid(True, axis="y", linewidth=self.grid_linewidth)

            # x軸フォーマット
            ax_left.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

            # タイトル
            ax_left.set_title(place, fontsize=self.title_fontsize)

            # 凡例をまとめる
            lines = line_left + line_right
            labels = [line.get_label() for line in lines]
            ax_left.legend(
                lines, labels, loc="upper left", fontsize=self.tick_labelsize
            )

            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=self.dpi, facecolor="white")
            buf.seek(0)

            graph_img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            graph_img = cv2.imdecode(graph_img_array, cv2.IMREAD_COLOR)

            if graph_img is None:
                raise ValueError(f"Failed to decode graph image for place: {place}")

            graph_imgs[place] = graph_img

            plt.close(fig)

        return graph_imgs


class NumPeopleGraphGenerator:
    def __init__(self):
        self.data_df = pd.DataFrame()

        # サイズの設定
        self.width_px = 1920
        self.height_px = 1080
        self.dpi = 100
        self.fig_width = self.width_px / self.dpi
        self.fig_height = self.height_px / self.dpi
        self.axes_rect = [0.10, 0.15, 0.85, 0.72]

        # フォントサイズの設定
        self.tick_labelsize = 15
        self.xlabel_fontsize = 25
        self.ylabel_fontsize = 25
        self.title_fontsize = 40

        # グラフのスタイルの設定
        self.color = "blue"
        self.linestyle = "-"
        self.linewidth = 3
        self.marker = "o"
        self.markersize = 5
        self.grid_linewidth = 1

        # グラフの表示範囲の設定
        self.ylim_min_scale = 0.7
        self.ylim_max_scale = 1.3

    def update(self, timestamp: float, data: dict) -> None:
        new_data = pd.DataFrame(
            {
                "timestamp": datetime.fromtimestamp(timestamp),
                **{
                    place: len(data[place][-1]["objects"]["point"])
                    for place in data.keys()
                },
            },
            index=[0],
        )
        self.data_df = pd.concat([self.data_df, new_data], ignore_index=True)
        self.data_df = self.data_df.sort_values(by="timestamp")

    def generate_graph_imgs(self) -> dict[str, np.ndarray]:
        places = self.data_df.columns.drop("timestamp")
        graph_imgs = {place: None for place in places}

        for place in places:
            fig = plt.figure(figsize=(self.fig_width, self.fig_height), dpi=self.dpi)

            # axes の位置とサイズを固定
            ax = fig.add_axes(self.axes_rect)

            ax.plot(
                self.data_df["timestamp"],
                self.data_df[place],
                linestyle=self.linestyle,
                linewidth=self.linewidth,
                marker=self.marker,
                markersize=self.markersize,
                # color=self.color,
            )

            ax.tick_params(labelsize=self.tick_labelsize)
            ax.set_xlabel("Date time", fontsize=self.xlabel_fontsize)
            ax.set_ylabel("Number of People", fontsize=self.ylabel_fontsize)

            ax.set_xlim(
                self.data_df["timestamp"].min(), self.data_df["timestamp"].max()
            )
            ax.set_ylim(
                self.data_df[place].min() * self.ylim_min_scale,
                self.data_df[place].max() * self.ylim_max_scale,
            )

            ax.grid(True, axis="y", linewidth=self.grid_linewidth)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

            ax.set_title(place, fontsize=self.title_fontsize)

            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=self.dpi, facecolor="white")
            buf.seek(0)

            graph_img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            graph_img = cv2.imdecode(graph_img_array, cv2.IMREAD_COLOR)

            if graph_img is None:
                raise ValueError(f"Failed to decode graph image for place: {place}")

            graph_imgs[place] = graph_img

            plt.close(fig)

        return graph_imgs


def set_num_people_data():
    import json

    data_dir = Path("received_synced_data")
    data_files = sorted(data_dir.glob("*.json"))
    graph_generator = NumPeopleGraphGenerator()
    for data_file in tqdm(data_files):
        timestamp = float(data_file.stem.split("_")[1])
        data = json.load(open(data_file, "r", encoding="utf-8"))
        graph_generator.update(timestamp, data)

    df = graph_generator.data_df
    df.to_csv("vis_tool/tmp/data_num_people.csv", index=False)


def main():
    df = pd.read_csv("vis_tool/tmp/data_num_people.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    graph_generator = NumPeopleGraphGenerator()
    graph_generator.data_df = df
    graph_imgs = graph_generator.generate_graph_imgs()
    for i, graph_img in enumerate(graph_imgs):
        i += 1
        cv2.imwrite(f"vis_tool/tmp/graph_{i:03d}.png", graph_img)


if __name__ == "__main__":
    set_num_people_data()
    main()
