import { useEffect, useRef, useState } from "react";

import { Link, useNavigate } from "react-router-dom";

import { useFolders } from "../context/FoldersContext";
import { registerFolderFiles } from "../data/folderFilesStorage";

import "./Evaluate.css";



type Step = 1 | 2 | 3;



export function Evaluate() {

  const navigate = useNavigate();

  const { folders, scopeId } = useFolders();

  const [step, setStep] = useState<Step>(1);

  const [folderId, setFolderId] = useState<string>("");

  const [presentationTitle, setPresentationTitle] = useState("");

  const [pptName, setPptName] = useState<string | null>(null);
  const [pptFile, setPptFile] = useState<File | null>(null);

  const [videoName, setVideoName] = useState<string | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [persona, setPersona] = useState<string>("soft");

  const [isRecording, setIsRecording] = useState(false);

  const [cameraError, setCameraError] = useState<string>("");

  const [recordedVideoUrl, setRecordedVideoUrl] = useState<string | null>(null);

  const [recordedVideoName, setRecordedVideoName] = useState<string | null>(null);

  const [selectedVideoPreviewUrl, setSelectedVideoPreviewUrl] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string>("");

  const previewRef = useRef<HTMLVideoElement | null>(null);

  const streamRef = useRef<MediaStream | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);

  const chunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    return () => {
      if (recorderRef.current && recorderRef.current.state !== "inactive") {
        recorderRef.current.stop();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (recordedVideoUrl) {
        URL.revokeObjectURL(recordedVideoUrl);
      }
    };
  }, [recordedVideoUrl]);

  useEffect(() => {
    setPresentationTitle("");
  }, [folderId]);

  useEffect(() => {
    const videoEl = previewRef.current;
    const stream = streamRef.current;
    if (!videoEl || !stream) return;
    videoEl.srcObject = stream;
    videoEl.play().catch(() => {
      // autoplay 정책으로 인해 실패할 수 있어 무시 (사용자 상호작용 후 재생됨)
    });
  }, [isRecording]);

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (previewRef.current) {
      previewRef.current.srcObject = null;
    }
  };

  const handleRecordToggle = async () => {
    if (isRecording) {
      recorderRef.current?.stop();
      return;
    }

    setCameraError("");

    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError("이 브라우저는 카메라 접근을 지원하지 않습니다.");
      return;
    }

    if (typeof MediaRecorder === "undefined") {
      setCameraError("이 브라우저는 녹화(MediaRecorder)를 지원하지 않습니다.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
      streamRef.current = stream;

      const recorder = new MediaRecorder(stream);
      recorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        setIsRecording(false);
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "video/webm" });
        const ext = blob.type.includes("mp4") ? "mp4" : "webm";
        const stampedName = `recorded-${new Date().toISOString().replace(/[:.]/g, "-")}.${ext}`;
        setRecordedVideoName(stampedName);
        
        // Blob을 File 객체로 변환하여 저장
        const file = new File([blob], stampedName, { type: blob.type });
        setVideoFile(file);

        if (recordedVideoUrl) {
          URL.revokeObjectURL(recordedVideoUrl);
        }
        const url = URL.createObjectURL(blob);
        setRecordedVideoUrl(url);
        stopCamera();
      };

      recorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error(error);
      setCameraError("카메라/마이크 권한이 없거나 장치에 접근할 수 없습니다.");
      stopCamera();
    }
  };

  const handleSelectRecordedVideo = () => {
    if (!recordedVideoName || !recordedVideoUrl || !videoFile) return;
    setVideoName(recordedVideoName);
    setSelectedVideoPreviewUrl(recordedVideoUrl);
    setStep((s) => (s < 3 ? 3 : s));
    setCameraError("");
  };

  const handleRetakeRecordedVideo = () => {
    if (recordedVideoUrl) {
      URL.revokeObjectURL(recordedVideoUrl);
    }
    setRecordedVideoUrl(null);
    setRecordedVideoName(null);
    setVideoFile(null);
    setCameraError("");
  };



  const hasFolders = folders.length > 0;

  const canAnalyze = Boolean(folderId && pptName && pptFile && videoName && videoFile && hasFolders);

  const handleAnalyzeClick = async () => {
    if (!pptFile || !videoFile) {
      setSubmitError("PPT와 영상 파일을 모두 선택해 주세요.");
      return;
    }

    setSubmitError("");
    setIsSubmitting(true);
    try {
      // 1. PPT 분석 요청
      const pptFormData = new FormData();
      pptFormData.append("file", pptFile);

      const pptRes = await fetch("http://127.0.0.1:8000/api/ppt/analyze", {
        method: "POST",
        body: pptFormData,
      });
      if (!pptRes.ok) {
        const err = (await pptRes.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(err?.detail ?? "PPT 분석 요청에 실패했습니다.");
      }

      // 2. 영상 업로드 및 분석 시작 요청
      const videoFormData = new FormData();
      videoFormData.append("file", videoFile);
      videoFormData.append("persona", persona);

      const videoRes = await fetch("http://127.0.0.1:8000/api/upload", {
        method: "POST",
        body: videoFormData,
      });
      if (!videoRes.ok) {
        throw new Error("영상 업로드 및 분석 요청에 실패했습니다.");
      }
      const videoData = await videoRes.json();
      const jobId = videoData.job_id;

      // 3. 로컬 저장소에 제출 정보 등록
      const submission = await registerFolderFiles(scopeId, folderId, {
        pptName,
        videoName,
        presentationTitle: presentationTitle.trim() || undefined,
      });
      if (submission) {
        if (selectedVideoPreviewUrl) {
          try {
            const raw = sessionStorage.getItem("overnight-video-preview-by-submission-v1");
            const map = raw ? (JSON.parse(raw) as Record<string, string>) : {};
            map[submission.id] = selectedVideoPreviewUrl;
            sessionStorage.setItem("overnight-video-preview-by-submission-v1", JSON.stringify(map));
          } catch {}
        }
        if (pptFile) {
          try {
            const pptUrl = URL.createObjectURL(pptFile);
            const raw = sessionStorage.getItem("overnight-ppt-blob-by-submission-v1");
            const map = raw ? (JSON.parse(raw) as Record<string, string>) : {};
            map[submission.id] = pptUrl;
            sessionStorage.setItem("overnight-ppt-blob-by-submission-v1", JSON.stringify(map));
          } catch {}
        }
        
        // Job ID 저장 (Analysis 페이지에서 결과 조회를 위해)
        try {
          const raw = sessionStorage.getItem("overnight-analysis-job-ids-v1");
          const map = raw ? (JSON.parse(raw) as Record<string, string>) : {};
          map[submission.id] = jobId;
          sessionStorage.setItem("overnight-analysis-job-ids-v1", JSON.stringify(map));
        } catch {}
      }
      navigate(
        submission
          ? `/analysis?submissionId=${encodeURIComponent(submission.id)}`
          : "/analysis"
      );
    } catch (error) {
      const msg = error instanceof Error ? error.message : "서버와 연결할 수 없습니다.";
      setSubmitError(msg);
    } finally {
      setIsSubmitting(false);
    }
  };



  return (

    <div className="page evaluate">

      <div className="page-inner page-inner--wide">

        <p className="evaluate-kicker">STT · 비전 · 융합 채점 파이프라인</p>

        <h1 className="evaluate-title">발표 자료 제출</h1>

        <p className="evaluate-lead">

          발표 음성은 글자로 옮겨 내용을 보고, 영상에서는 시선·제스처·표정을 함께 봅니다. <strong>저장 폴더</strong>를 고른

          뒤 PPT와 영상을 올리면 항목별 점수와 PDF·Excel 리포트로 이어집니다.

        </p>



        <section className="evaluate-panel evaluate-panel--folder" aria-labelledby="sf">

          <h2 id="sf" className="evaluate-panel__title">

            저장할 발표 폴더

          </h2>

          <p className="evaluate-panel__desc">

            녹화·업로드한 영상과 PPT, 이후 채점 결과가 이 폴더에 정리됩니다. 폴더는 <strong>문서</strong> 화면에서

            추가·삭제할 수 있습니다.

          </p>

          <div className="evaluate-folder-row">

            <label className="visually-hidden" htmlFor="evaluate-folder">

              발표 폴더 선택

            </label>

            {hasFolders ? (

              <select

                id="evaluate-folder"

                className="evaluate-folder-select"

                value={folderId}

                onChange={(e) => setFolderId(e.target.value)}

              >

                <option value="">저장할 폴더를 선택하세요</option>

                {folders.map((f) => (

                  <option key={f.id} value={f.id}>

                    {f.name}

                  </option>

                ))}

              </select>

            ) : (

              <p className="evaluate-folder-empty">

                등록된 폴더가 없습니다.{" "}

                <Link to="/projects" className="evaluate-folder-link">

                  문서에서 폴더 만들기

                </Link>

              </p>

            )}

            <Link to="/projects" className="evaluate-folder-link">

              폴더 관리

            </Link>

          </div>

          {hasFolders && folderId ? (
            <div className="evaluate-presentation-title">
              <label className="evaluate-presentation-title__label" htmlFor="evaluate-presentation-title">
                발표 제목 (문서 화면에서 하위로 묶입니다)
              </label>
              <input
                id="evaluate-presentation-title"
                className="evaluate-presentation-title__input"
                type="text"
                maxLength={80}
                placeholder="예: 1주차 중간발표, 최종 리허설"
                value={presentationTitle}
                onChange={(e) => setPresentationTitle(e.target.value)}
                autoComplete="off"
              />
              <p className="evaluate-presentation-title__hint">비워 두면 문서 화면에서 「(제목 없음)」으로 묶입니다.</p>
            </div>
          ) : null}

        </section>



        <section className="evaluate-panel" aria-labelledby="persona-title">
          <h2 id="persona-title" className="evaluate-panel__title">
            AI 코칭 스타일
          </h2>
          <p className="evaluate-panel__desc">원하는 AI 전문가의 피드백 스타일을 선택해 주세요.</p>
          <div className="evaluate-persona-row">
            <button
              type="button"
              className={`evaluate-persona-btn ${persona === "soft" ? "active" : ""}`}
              onClick={() => setPersona("soft")}
            >
              🌸 부드러운 조언자
            </button>
            <button
              type="button"
              className={`evaluate-persona-btn ${persona === "sharp" ? "active" : ""}`}
              onClick={() => setPersona("sharp")}
            >
              🔥 냉철한 전문가
            </button>
          </div>
        </section>

        <ol className="evaluate-steps" aria-label="진행 단계">

          <li className={step >= 1 ? "evaluate-steps__item--active" : ""}>

            <span className="evaluate-steps__num">1</span>

            PPT 업로드

          </li>

          <li className={step >= 2 ? "evaluate-steps__item--active" : ""}>

            <span className="evaluate-steps__num">2</span>

            발표 영상

          </li>

          <li className={step >= 3 ? "evaluate-steps__item--active" : ""}>

            <span className="evaluate-steps__num">3</span>

            분석 실행

          </li>

        </ol>



        <section className="evaluate-panel" aria-labelledby="s1">

          <h2 id="s1" className="evaluate-panel__title">

            ① PPT 파일

          </h2>

          <p className="evaluate-panel__desc">채점 기준과 비교할 슬라이드 자료를 올려주세요. (.pptx 등)</p>

          <label className="evaluate-drop">

            <input

              type="file"

              accept=".ppt,.pptx,application/vnd.ms-powerpoint,application/vnd.openxmlformats-officedocument.presentationml.presentation"

              className="evaluate-drop__input"

              onChange={(e) => {

                const f = e.target.files?.[0];

                setPptName(f?.name ?? null);
                setPptFile(f ?? null);

                if (f) setStep((s) => (s < 2 ? 2 : s));

              }}

            />

            <span className="evaluate-drop__ui">

              {pptName ? (

                <>

                  <strong>{pptName}</strong>

                  <span className="evaluate-drop__hint">다른 파일로 바꾸려면 클릭</span>

                </>

              ) : (

                <>

                  <span className="evaluate-drop__icon" aria-hidden>

                    📊

                  </span>

                  클릭하여 PPT 선택

                </>

              )}

            </span>

          </label>

        </section>



        <section className="evaluate-panel" aria-labelledby="s2">

          <h2 id="s2" className="evaluate-panel__title">

            ② 발표 영상

          </h2>

          <p className="evaluate-panel__desc">

            아래에서 녹화하거나 파일을 업로드하세요. 음성·얼굴이 나와야 태도·음성 항목을 채점할 수 있습니다.

          </p>

          <div className="evaluate-preview" role="region" aria-label="카메라 미리보기">
            {recordedVideoUrl && !isRecording ? (
              <video className="evaluate-preview__video" src={recordedVideoUrl} controls playsInline />
            ) : (
              <>
                <video ref={previewRef} className="evaluate-preview__video" autoPlay muted playsInline />
                {!isRecording ? (
                  <p className="evaluate-preview__placeholder">
                    녹화 시작을 누르면 카메라 미리보기가 켜집니다. 기존 파일을 쓸 경우 아래에서 영상 파일을 선택해 주세요.
                  </p>
                ) : null}
              </>
            )}
          </div>
          {recordedVideoUrl && !isRecording ? (
            <div className="evaluate-preview-actions">
              <button type="button" className="evaluate-btn evaluate-btn--secondary" onClick={handleRetakeRecordedVideo}>
                다시 찍기
              </button>
              <button type="button" className="evaluate-btn evaluate-btn--primary" onClick={handleSelectRecordedVideo}>
                이 영상 선택하기
              </button>
            </div>
          ) : null}
          {cameraError ? <p className="evaluate-note evaluate-note--error">{cameraError}</p> : null}

          <div className="evaluate-row">

            <button type="button" className="evaluate-btn evaluate-btn--secondary" onClick={handleRecordToggle}>
              {isRecording ? "녹화 중지" : "녹화 시작"}
            </button>

            <label className="evaluate-btn evaluate-btn--ghost">

              영상 파일 선택

              <input

                type="file"

                accept="video/*"

                className="evaluate-drop__input"

                onChange={(e) => {

                  const f = e.target.files?.[0];

                  setVideoName(f?.name ?? null);
                  setVideoFile(f ?? null);
                  if (f) {
                    setSelectedVideoPreviewUrl(URL.createObjectURL(f));
                  }

                  if (f) setStep((s) => (s < 3 ? 3 : s));

                }}

              />

            </label>

          </div>

          {videoName && (

            <p className="evaluate-filetag">

              선택됨: <strong>{videoName}</strong>

            </p>

          )}

        </section>



        <section className="evaluate-panel evaluate-panel--last" aria-labelledby="s3">

          <h2 id="s3" className="evaluate-panel__title">

            ③ 분석

          </h2>

          <p className="evaluate-panel__desc">

            준비가 되면 서버로 전송해 PPT·발표 대조 및 멀티모달 채점을 실행합니다.

          </p>

          <button

            type="button"

            className="evaluate-btn evaluate-btn--primary evaluate-btn--block"

            disabled={!canAnalyze || isSubmitting}
            onClick={handleAnalyzeClick}

          >

            {canAnalyze

              ? isSubmitting
                ? "PPT 분석 중..."
                : "채점 시작하기"

              : !hasFolders

                ? "폴더를 먼저 만드세요"

                : !folderId || !pptName || !videoName

                  ? "폴더·PPT·영상을 모두 준비해 주세요"

                  : "채점 시작하기"}

          </button>

          {submitError ? <p className="evaluate-note evaluate-note--error">{submitError}</p> : null}

          {!canAnalyze && (

            <p className="evaluate-note">

              ※ <strong>폴더</strong>, PPT, 영상을 모두 준비하면 채점을 시작할 수 있습니다. 폴더 목록은 문서 화면과

              문서 화면과 같이 이 기기에만 저장됩니다.

            </p>

          )}

        </section>



        <p className="evaluate-foot">

          <Link to="/guide">채점 항목 상세 보기</Link>

          <span aria-hidden> · </span>

          <Link to="/">홈으로</Link>

        </p>

      </div>

    </div>

  );

}

