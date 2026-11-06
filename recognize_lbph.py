"""
recognize_lbph.py
步驟 3：Haar 偵測人臉 → 裁切灰階 → LBPH 比對 → 距離 ≤ 閾值 判定身分，否則為 Unknown

用法：
    python recognize_lbph.py                       # 使用攝影機即時辨識
    python recognize_lbph.py --image test.jpg      # 辨識單張圖片
    python recognize_lbph.py --threshold 60        # 調整閾值（越小越嚴格）
"""
import argparse
import json
import os

import cv2

FACE_SIZE = (200, 200)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_alt.xml")


def parse_args():
    parser = argparse.ArgumentParser(description="LBPH 人臉辨識")
    parser.add_argument("--image", help="要辨識的圖片路徑（不指定則開啟攝影機）")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=70.0,
                        help="LBPH 距離閾值，距離 ≤ 閾值才視為已知人員（預設 70）")
    parser.add_argument("--model-dir", default="models")
    return parser.parse_args()


def load_model(model_dir):
    model_path = os.path.join(model_dir, "lbph_model.yml")
    labels_path = os.path.join(model_dir, "labels.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError("找不到模型，請先執行 train_lbph.py")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    with open(labels_path, encoding="utf-8") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    return recognizer, label_map


def recognize_frame(frame, face_cascade, recognizer, label_map, threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # scaleFactor=1.1、minNeighbors=3
    faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(80, 80))
    for (x, y, w, h) in faces:
        face_img = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
        label_id, distance = recognizer.predict(face_img)  # distance 越小越相似

        if distance <= threshold:
            name, color = label_map.get(label_id, "Unknown"), (0, 255, 0)
        else:
            name, color = "Unknown", (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{name} ({distance:.1f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame, len(faces)


def main():
    args = parse_args()
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise IOError(f"無法載入 Haar Cascade：{CASCADE_PATH}")
    recognizer, label_map = load_model(args.model_dir)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise FileNotFoundError(f"無法讀取圖片 {args.image}")
        frame, n = recognize_frame(frame, face_cascade, recognizer, label_map,
                                   args.threshold)
        print("------> Found {0} faces! <------".format(n))
        cv2.imshow("LBPH Recognition", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise IOError(f"無法開啟攝影機 {args.camera}")
    print("[INFO] 即時辨識中，按 q 離開")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, _ = recognize_frame(frame, face_cascade, recognizer, label_map,
                                   args.threshold)
        cv2.imshow("LBPH Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
