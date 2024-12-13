import cv2
import os
import numpy as np
from glitch_this import ImageGlitcher
from PIL import Image

# Glitch 생성기 초기화
glitcher = ImageGlitcher()

# 입력 및 출력 폴더 경로 설정
input_folder = "input/joi_clip/test"
output_folder = "output/glitch"

# 출력 폴더 생성
os.makedirs(output_folder, exist_ok=True)

def apply_rgb_glitch(frame, intensity=5):
    """
    Apply an RGB channel split and shift glitch effect.
    
    :param frame: Input frame (BGR format)
    :param intensity: Maximum pixel shift for the glitch effect
    :return: Frame with RGB glitch effect applied
    """
    height, width, _ = frame.shape

    # Split BGR channels
    b, g, r = cv2.split(frame)

    # Random shift values
    shift_x = np.random.randint(-intensity, intensity + 1)
    shift_y = np.random.randint(-intensity, intensity + 1)

    # Shift channels
    r_shifted = np.roll(r, shift_x, axis=1)  # Horizontal shift
    g_shifted = np.roll(g, shift_y, axis=0)  # Vertical shift

    # Merge channels back
    glitched_frame = cv2.merge([b, g_shifted, r_shifted])
    return glitched_frame


def process_video(input_video_path, output_video_path):
    """
    Process a single video: apply glitch effects and save the result.
    
    :param input_video_path: Path to the input video
    :param output_video_path: Path to save the processed video
    """
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # print(f"Processing video: {input_video_path}")
    # print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}, Total Frames: {frame_count}")

    # VideoWriter 초기화
    output_video = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
    )

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV 프레임을 PIL 이미지로 변환
        frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Glitch-this 라이브러리를 통한 기본 글리치 효과 적용
        glitched_image = glitcher.glitch_image(frame_image, 0.8, gif=False)

        # Glitched 이미지를 OpenCV 포맷으로 변환
        glitched_frame = cv2.cvtColor(np.array(glitched_image), cv2.COLOR_RGB2BGR)

        # 추가 RGB 채널 분리 및 왜곡 효과 적용
        glitched_frame = apply_rgb_glitch(glitched_frame, intensity=10)

        # 결과 프레임 저장
        output_video.write(glitched_frame)

        frame_idx += 1
        # print(f"Processed frame {frame_idx}/{frame_count}", end="\r")

    cap.release()
    output_video.release()
    # print(f"\nSaved glitched video to: {output_video_path}")


# 폴더 내 모든 영상 처리
for video_file in os.listdir(input_folder):
    if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # 지원하는 파일 확장자
        input_video_path = os.path.join(input_folder, video_file)
        output_video_path = os.path.join(output_folder, video_file)

        process_video(input_video_path, output_video_path)

print("All videos processed successfully.")
