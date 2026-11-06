"""
train_lbph.py
步驟 2：讀取 data/ 底下每位人員的人臉圖片，訓練 LBPH 模型
輸出：models/lbph_model.yml、models/labels.json

用法：
    python train_lbph.py
"""
import argparse
import json
import os

import cv2
import numpy as np

FACE_SIZE = (200, 200)


def parse_args():
    parser = argparse.ArgumentParser(description="訓練 LBPH 人臉辨識模型")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-dir", default="models")
    return parser.parse_args()


def load_dataset(data_dir):
    faces, labels, label_map = [], [], {}
    people = sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )
    for label_id, person in enumerate(people):
        label_map[label_id] = person
        person_dir = os.path.join(data_dir, person)
        for fname in sorted(os.listdir(person_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(person_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(cv2.resize(img, FACE_SIZE))
            labels.append(label_id)
        print(f"[INFO] {person} (label {label_id})：{labels.count(label_id)} 張")
    return faces, labels, label_map


def main():
    args = parse_args()
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"找不到資料夾 {args.data_dir}，請先執行 collect_faces.py")

    faces, labels, label_map = load_dataset(args.data_dir)
    if len(faces) == 0:
        raise ValueError("沒有可用的訓練圖片")

    # 需安裝 opencv-contrib-python 才有 cv2.face
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )
    recognizer.train(faces, np.array(labels))

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "lbph_model.yml")
    labels_path = os.path.join(args.model_dir, "labels.json")
    recognizer.write(model_path)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"[DONE] 共 {len(faces)} 張、{len(label_map)} 人")
    print(f"       模型：{model_path}")
    print(f"       標籤：{labels_path}")


if __name__ == "__main__":
    main()
