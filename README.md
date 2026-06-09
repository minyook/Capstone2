# 🌙 Overnight.AI
> **AI 기반 발표 및 스피치 멀티모달 자동 평가 시스템**
>
> 본 프로젝트는 고도화된 딥러닝 분석 엔진(FastAPI)과 반응형 시각화 대시보드(React)를 융합하여, 발표자의 **영상(Vision), 음성(Audio), 언어/슬라이드(Text/PPT)** 데이터를 초정밀 멀티모달 파이프라인으로 동적 분석합니다. 계측된 정량적 수치를 바탕으로 발표 역량을 다각도로 평가하고 맞춤형 피드백을 제공하는 프리미엄 스피치 코칭 서비스입니다.

---

## 🏗️ System Architecture (시스템 구조)

```mermaid
graph TD
    subgraph Client [React Frontend App]
        UI[사용자 UI 대시보드]
        Chart[Recharts 역량 차트]
        Player[실시간 피드백 비디오 플레이어]
    end

    subgraph Server [FastAPI Backend Engine]
        API[FastAPI 라우터 & 작업 큐]
        FFmpeg[FFmpeg 전처리 및 프레임 분리]
        
        subgraph Vision_Pipeline [Vision 분석]
            YOLO[YOLOv8-Pose 제스처/포스처 분석]
            MP[MediaPipe Face Landmarker 시선/표정 분석]
        end

        subgraph Audio_Pipeline [Audio 분석]
            Whisper[Whisper Local STT 단어별 타임스탬프]
            Praat[Praat 음성 운율 & Jitter/Shimmer 분석]
        end

        subgraph Slide_Pipeline [Content 검증]
            SV[Slide Verify 영상-장표 매칭 엔진]
        end
        
        Combiner[Data Combiner 멀티모달 시간축 정렬]
        Feedback[Feedback Engine 채점 및 피드백 생성]
    end

    subgraph LLM_Providers [AI Scoring & Feedback]
        Ollama[Local Ollama: Gemma3:4b]
        Gemini[Cloud Gemini API]
    end

    %% Flow
    UI -->|1. 비디오 및 PPT 업로드| API
    API -->|2. 백그라운드 큐 할당| FFmpeg
    FFmpeg -->|오디오 분리| Audio_Pipeline
    FFmpeg -->|프레임 추출| Vision_Pipeline
    FFmpeg -->|슬라이드 대조| Slide_Pipeline
    
    Vision_Pipeline -->|시각 데이터| Combiner
    Audio_Pipeline -->|음성/텍스트 데이터| Combiner
    
    Combiner -->|정렬된 데이터| Feedback
    Slide_Pipeline -->|슬라이드 일치 점수| Feedback
    
    Feedback -->|3. 프롬프트 생성| LLM_Providers
    LLM_Providers -->|4. 평가 점수 & 마크다운 피드백| Feedback
    Feedback -->|5. 최종 통합 JSON 생성| API
    API -->|6. 실시간 폴링 및 대시보드 반영| UI
