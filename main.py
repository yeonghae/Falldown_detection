import os
import cv2
import numpy as np
from ultralytics import YOLO
from Models.segment_anything.segment_anything import sam_model_registry, SamPredictor
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from Models.Depth_Anything.depth_anything.dpt import DepthAnything
from Models.Depth_Anything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

# YOLOv11 모델 로드
model = YOLO("yolo11n.pt")

# Segment Anything 모델 로드
sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"  # SAM 체크포인트 경로
sam_model_type = "vit_h"
sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam.to(device=device)
sam_predictor = SamPredictor(sam)

# Depth Anything 모델 로드
depth_model = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14').to(device).eval()
transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

# 입력 및 출력 디렉토리 설정
INPUT_VIDEO_DIR = "input/"
OUTPUT_IMAGE_DIR = "output/segmentation"
OUTPUT_DEPTH_DIR = "output/depth"
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DEPTH_DIR, exist_ok=True)

# 사람 클래스 ID (YOLO 모델에 따라 다를 수 있음, 일반적으로 COCO 데이터셋에서 0이 사람 클래스)
PERSON_CLASS_ID = 0

# 비디오에서 사람 객체 탐지, 배경 제거, 뎁스 이미지 생성 수행
def detect_segment_and_depth(input_dir, output_seg_dir, output_depth_dir, model, predictor, depth_model, transform):
    for video_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, video_name)

        # 비디오 캡처
        cap = cv2.VideoCapture(input_path)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO 모델로 탐지 수행
            results = model(frame)

            masks = []  # 객체 마스크 저장

            for result in results[0].boxes:
                if int(result.cls) == PERSON_CLASS_ID:  # 사람 클래스 필터링
                    x1, y1, x2, y2 = map(int, result.xyxy[0])

                    # 탐지된 영역의 이미지 및 마스크 생성
                    predictor.set_image(frame)

                    # Segment Anything 예측 수행
                    input_box = np.array([x1, y1, x2, y2])
                    mask, _, _ = predictor.predict(box=input_box, point_coords=None, point_labels=None, multimask_output=False)

                    # 마스크를 frame 크기로 변환
                    mask = mask[0]  # SAM에서 반환된 마스크 첫 번째 값
                    mask_resized = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

                    # 탐지된 마스크와 바운딩 박스 저장
                    masks.append((mask_resized, (x1, y1, x2, y2)))

            # 결과 이미지 생성
            if masks:
                combined_mask = np.zeros(frame.shape[:2], dtype=bool)
                for mask, bbox in masks:
                    combined_mask = np.logical_or(combined_mask, mask.astype(bool))

                # 배경 제거
                segmented_frame = frame.copy()
                segmented_frame[~combined_mask] = 0

                # 바운딩 박스를 원본 이미지에서 유지
                for _, bbox in masks:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(segmented_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 결과 이미지 저장
                seg_output_path = os.path.join(output_seg_dir, f"{os.path.splitext(video_name)[0]}_frame_{frame_idx}.png")
                cv2.imwrite(seg_output_path, segmented_frame)

            # 뎁스 이미지 생성
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
            transformed_frame = transform({'image': frame_rgb})['image']
            transformed_frame = torch.from_numpy(transformed_frame).unsqueeze(0).to(device)

            with torch.no_grad():
                depth = depth_model(transformed_frame)

            depth = F.interpolate(depth[None], (frame.shape[0], frame.shape[1]), mode='bilinear', align_corners=False)[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

            depth = depth.cpu().numpy().astype(np.uint8)
            depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

            # 뎁스 이미지 저장
            depth_output_path = os.path.join(output_depth_dir, f"{os.path.splitext(video_name)[0]}_frame_{frame_idx}_depth.png")
            cv2.imwrite(depth_output_path, depth_color)

            frame_idx += 1

        cap.release()
        print(f"Processed video: {video_name}")

if __name__ == "__main__":
    detect_segment_and_depth(INPUT_VIDEO_DIR, OUTPUT_IMAGE_DIR, OUTPUT_DEPTH_DIR, model, sam_predictor, depth_model, transform)
