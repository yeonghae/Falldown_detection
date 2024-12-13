import cv2
import os
import itertools

# 입력 및 출력 폴더 경로 설정
input_folder = "joi_clip/test"
output_folder = "output/combine"

def get_video_properties(video_path):
    """
    영상의 속성(fps, 너비, 높이, 프레임 수)을 가져오는 함수
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, width, height, frame_count

def alternate_segments(video1_path, video2_path, output_video_path1, output_video_path2, segment_length=8):
    """
    두 영상의 프레임을 일정 구간(segment_length) 단위로 번갈아가며 섞어서 두 개의 새로운 영상을 생성하는 함수
    한 영상이 더 길 경우, 남은 부분은 긴 영상의 프레임으로 채웁니다.
    """
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print(f"영상을 열 수 없습니다: {video1_path}, {video2_path}")
        return

    # 두 영상의 속성 가져오기 (동일한 FPS 및 해상도를 가정)
    fps1, width1, height1, frame_count1 = get_video_properties(video1_path)
    fps2, width2, height2, frame_count2 = get_video_properties(video2_path)

    # 두 영상의 해상도를 동일하게 설정 (필요시 리사이즈)
    output_width = min(width1, width2)
    output_height = min(height1, height2)
    fps = min(fps1, fps2)

    # 두 개의 출력 영상을 위한 VideoWriter 초기화
    output_video1 = cv2.VideoWriter(
        output_video_path1,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, output_height)
    )

    output_video2 = cv2.VideoWriter(
        output_video_path2,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, output_height)
    )

    # 프레임을 번갈아가며 섞기
    frame_buffer1 = []
    frame_buffer2 = []
    segment_counter = 0

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 and not ret2:
            break

        if ret1:
            frame1 = cv2.resize(frame1, (output_width, output_height))
            frame_buffer1.append(frame1)
        if ret2:
            frame2 = cv2.resize(frame2, (output_width, output_height))
            frame_buffer2.append(frame2)

        if len(frame_buffer1) >= segment_length or not ret1:
            if segment_counter % 2 == 0:
                # 첫 번째 출력 영상: video1 -> video2 순서로 프레임 작성
                for frame in frame_buffer1:
                    output_video1.write(frame)
                for frame in frame_buffer2:
                    output_video1.write(frame)

                # 두 번째 출력 영상: video2 -> video1 순서로 프레임 작성
                for frame in frame_buffer2:
                    output_video2.write(frame)
                for frame in frame_buffer1:
                    output_video2.write(frame)

            # 버퍼 초기화 및 세그먼트 카운터 증가
            frame_buffer1.clear()
            segment_counter += 1

        if len(frame_buffer2) >= segment_length or not ret2:
            frame_buffer2.clear()

    # 긴 영상의 남은 프레임 작성
    while ret1:
        ret1, frame1 = cap1.read()
        if ret1:
            frame1 = cv2.resize(frame1, (output_width, output_height))
            output_video1.write(frame1)
            output_video2.write(frame1)

    while ret2:
        ret2, frame2 = cap2.read()
        if ret2:
            frame2 = cv2.resize(frame2, (output_width, output_height))
            output_video1.write(frame2)
            output_video2.write(frame2)

    # 자원 해제
    cap1.release()
    cap2.release()
    output_video1.release()
    output_video2.release()
    # print(f"결과 영상이 저장되었습니다: {output_video_path1}, {output_video_path2}")

# 폴더 내 영상 목록 가져오기
video_files = [
    os.path.join(input_folder, f) for f in os.listdir(input_folder)
    if f.endswith((".mp4", ".avi", ".mov", ".mkv"))
]

# 모든 영상 조합 생성 (임의의 두 영상 쌍 선택)
video_combinations = list(itertools.combinations(video_files, 2))

# 각 조합에 대해 섞인 영상 생성
for idx, (video1, video2) in enumerate(video_combinations):
    output_video_path1 = os.path.join(output_folder, f"combined_video_{idx + 1}_1.mp4")
    output_video_path2 = os.path.join(output_folder, f"combined_video_{idx + 1}_2.mp4")
    # print(f"{os.path.basename(video1)}와 {os.path.basename(video2)}를 결합 중...")
    alternate_segments(video1, video2, output_video_path1, output_video_path2, segment_length=8)


print("All combined videos created successfully.")
