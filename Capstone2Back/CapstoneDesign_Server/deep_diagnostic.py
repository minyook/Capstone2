import cv2
import numpy as np
from pathlib import Path
import os
from processing.video_analyzer import analyze_frame_vision

def run_deep_diagnostic():
    # 1. 테스트할 프레임 확보
    frame_dir = Path("frames")
    # 가장 최근 작업 폴더 찾기
    subdirs = sorted([d for d in frame_dir.iterdir() if d.is_dir()], key=os.path.getmtime, reverse=True)
    
    target_frames = []
    if subdirs:
        print(f"Checking frames in: {subdirs[0]}")
        target_frames = sorted(list(subdirs[0].glob("*.jpg")))[:20] # 처음 20장
    
    if not target_frames:
        # 루트 frames 폴더 확인
        target_frames = sorted(list(frame_dir.glob("*.jpg")))[:20]

    if not target_frames:
        print("Error: No frames found to analyze. Please upload a video first.")
        return

    print(f"Starting diagnostic on {len(target_frames)} frames...")
    
    diag_out = Path("out/diagnostic")
    diag_out.mkdir(parents=True, exist_ok=True)

    for i, frame_path in enumerate(target_frames):
        print(f"\n--- Frame {i}: {frame_path.name} ---")
        
        # 전체 분석 실행
        result = analyze_frame_vision(str(frame_path), i * 0.2)
        
        # YOLO 결과 기록
        print(f"YOLO Person: {result.yolo.has_person}")
        if result.yolo.has_person:
            print(f"YOLO BBox: {result.yolo.person_bbox}")
            print(f"YOLO Keypoints Count: {len([k for k in result.yolo.keypoints if k[0] > 0])}")
        
        # MediaPipe 결과 기록
        print(f"MediaPipe Face: {result.face.has_face}")
        if not result.face.has_face:
            print(f"MediaPipe Error: {result.face.error}")
        else:
            print(f"Face Metrics: Smile={result.face.smile:.2f}, Gaze_H={result.face.gaze_h:.2f}")

        # 시각적 검증용 이미지 저장 (원본에 BBox 표시)
        img = cv2.imread(str(frame_path))
        if img is not None:
            if result.yolo.has_person and result.yolo.person_bbox:
                x1, y1, x2, y2 = map(int, result.yolo.person_bbox)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 키포인트 표시
                for kp in result.yolo.keypoints:
                    if kp[0] > 0:
                        cv2.circle(img, (int(kp[0]), int(kp[1])), 3, (0, 0, 255), -1)
            
            status_txt = f"Face: {'OK' if result.face.has_face else 'FAIL'}"
            cv2.putText(img, status_txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imwrite(str(diag_out / f"diag_{frame_path.name}"), img)

    print(f"\nDiagnostic complete. Check '{diag_out}' for results.")

if __name__ == "__main__":
    run_deep_diagnostic()
