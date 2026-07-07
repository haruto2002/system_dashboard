import json
from pathlib import Path

import cv2
from tqdm import tqdm

from vis_tool.dashboard import Dashboard


def create_dashboard_image():
    pass


def main():
    map_img_file = "vis_tool/map_data/yokohama_2020508/all_map.jpg"
    map_config_dir = "vis_tool/map_data/yokohama_2020508/map_config"
    map_draw_style_file = "vis_tool/map_data/yokohama_2020508/map_draw_style.yaml"

    places = ["worldporter", "akarenga", "chosha", "kokusaibashi"]

    dashboard = Dashboard(
        places=places,
        map_config_dir=map_config_dir,
        map_img_file=map_img_file,
        map_draw_style_file=map_draw_style_file,
    )
    save_dir = Path("tmp")
    save_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path("received_synced_data")
    json_files = sorted(list(data_dir.glob("*.json")))
    for i, json_file in tqdm(enumerate(json_files), total=len(json_files)):
        data = json.load(open(json_file, "r", encoding="utf-8"))
        timestamp = float(json_file.stem.split("_")[1])

        # データから各可視化を実施
        trajectory_img, heatmap_img_dict, focus_area_outputs, graph_img_dict = (
            dashboard.display(data, timestamp)
        )

        # 各生成物を保存
        components_save_dir = save_dir / f"frame_{i:06d}"
        components_save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(components_save_dir / "trajectory_img.jpg", trajectory_img)
        for place, heatmap_img in heatmap_img_dict.items():
            cv2.imwrite(components_save_dir / f"heatmap_img_{place}.jpg", heatmap_img)
        for j, (place, focus_areas_img, score) in enumerate(focus_area_outputs):
            cv2.imwrite(components_save_dir / f"top{i + 1:02d}.jpg", focus_areas_img)
        for place, graph_img in graph_img_dict.items():
            cv2.imwrite(components_save_dir / f"graph_{place}.jpg", graph_img)

        # 各生成物を一つの画像として保存
        # dashboard_img = create_dashboard_image()
        # cv2.imwrite(save_dir / "dashboard.jpg", dashboard_img)

        if i > 5:
            break


if __name__ == "__main__":
    main()
