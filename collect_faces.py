"""
collect_faces.py
步驟 1：使用攝影機 + Haar Cascade 偵測人臉，裁切灰階人臉並存入 data/<人員ID>/

用法：
    python collect_faces.py --id student01
    python collect_faces.py --id student02 --count 80 --camera 0
"""
import argparse
import os

import cv2

FACE_SIZE = (200, 200)  # 統一人臉尺寸，訓練與辨識需一致
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_alt.xml")


def parse_args():
    parser = argparse.ArgumentParser(description="收集人臉樣本")
    parser.add_argument("--id", required=True, help="人員資料夾名稱，例如 student01")
    parser.add_argument("--count", type=int, default=50, help="要收集的張數（預設 50）")
    parser.add_argument("--camera", type=int, default=0, help="攝影機編號（預設 0）")
    parser.add_argument("--data-dir", default="data", help="資料存放根目錄")
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = os.path.join(args.data_dir, args.id)
    os.makedirs(save_dir, exist_ok=True)

    # 讀取專案根目錄的 Haar Cascade 分類器模型檔
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise IOError(f"無法載入 Haar Cascade：{CASCADE_PATH}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise IOError(f"無法開啟攝影機 {args.camera}")

    # 接續既有檔案編號，避免覆蓋
    saved = len([f for f in os.listdir(save_dir) if f.lower().endswith(".jpg")])
    target = saved + args.count
    print(f"[INFO] 開始收集 {args.id} 的人臉，按 q 提前結束")

    while saved < target:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 讀取影像失敗")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # scaleFactor=1.1、minNeighbors=3
        faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(80, 80))

        # 每張畫面只取最大的一張臉，避免誤存他人
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_img = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
            saved += 1
            cv2.imwrite(os.path.join(save_dir, f"{saved:03d}.jpg"), face_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, f"{args.id}: {saved}/{target}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Collect Faces", frame)

        if cv2.waitKey(100) & 0xFF == ord("q"):  # 每 0.1 秒存一張，讓角度有變化
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] {save_dir} 目前共 {saved} 張人臉")


if __name__ == "__main__":
    main()
