import os
from moviepy.editor import VideoFileClip, concatenate_videoclips

# 비디오 파일이 있는 폴더 경로
folder_path = "input/combine"

# 출력 파일 경로 지정
output_path = "output/combineall/glitch_video.mp4"

# 출력 경로의 디렉토리 생성 (존재하지 않을 경우)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 폴더 내의 모든 MP4 파일 가져오기
video_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".mp4")]

# 파일 이름을 정렬 (예: V1-0110, V1-0111 순서대로)
video_files.sort()

# VideoFileClip 객체 생성
video_clips = [VideoFileClip(file) for file in video_files]

# 비디오 클립 연결
final_clip = concatenate_videoclips(video_clips)

# 결과 파일 저장
final_clip.write_videofile(output_path, codec="libx264")
