import { useMemo, useRef, useState, useEffect } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useFirestoreSyncRevision } from "../context/FirestoreSyncContext";
import { useFolders } from "../context/FoldersContext";
import { findSubmissionById, submissionPrimaryFileName } from "../data/folderFilesStorage";
import {
  loadScoresForView,
  saveAnalysisResultForSubmission,
  totalFromScores,
  type StoredRubricScores,
} from "../data/analysisResultStorage";
import { RUBRIC } from "../data/rubric";
import {
  loadSlideVerifyForSubmission,
  summarizeLowMatchReasons,
  verdictLabelKo,
  type SlideVerifyResult,
} from "../data/slideVerifyStorage";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { exportPDF, exportExcel } from "../utils/exportReport";
import "./Analysis.css";

// Helper for loading step detection
const getActiveStep = (msg: string): number => {
  const m = msg.toLowerCase();
  if (m.includes("품질") || m.includes("0/6") || m.includes("전송") || m.includes("검증")) return 1;
  if (m.includes("yolo") || m.includes("시각") || m.includes("3/6")) return 2;
  if (m.includes("whisper") || m.includes("음성") || m.includes("운율") || m.includes("praat") || m.includes("4/6") || m.includes("5/6") || m.includes("6/6")) return 3;
  if (m.includes("gemma") || m.includes("피드백") || m.includes("7/7") || m.includes("ai 피드백")) return 4;
  return 1;
};

// Speech tips and context-aware guidance for the loading screen
const getDynamicLoadingMessage = (step: number): string => {
  switch (step) {
    case 1:
      return "💡 스피치 팁: 첫 30초 안에 청중의 관심을 사로잡으려면 강력한 오프닝 질문을 던져보세요!";
    case 2:
      return "💡 자세 팁: 발표할 때 양손을 가볍게 열어 열린 자세를 취하면 청중에게 개방적이고 신뢰감을 주는 이미지를 전달합니다.";
    case 3:
      return "💡 목소리 팁: 중요한 핵심 개념을 강조하기 직전에는 1.5초간 의도적 침묵(Pause)을 지켜 청중의 호기심을 유도해 보세요.";
    case 4:
      return "💡 설득 팁: 수사학적 신뢰도(Ethos)를 높이기 위해 구체적 수치와 학술적 근거를 현상 설명과 결합해 제시하면 설득력이 증폭됩니다.";
    default:
      return "💡 성공적인 발표를 위한 AI의 정밀 분석이 안전하게 진행되고 있습니다.";
  }
};

export function Analysis() {
  const { scopeId } = useFolders();
  const fsRevision = useFirestoreSyncRevision();
  const [searchParams] = useSearchParams();
  const submissionId = searchParams.get("submissionId");
  const previewVideoRef = useRef<HTMLVideoElement | null>(null);

  const [overallFeedback, setOverallFeedback] = useState<string | null>(null);
  const [timelineFeedback, setTimelineFeedback] = useState<Record<string, string>>({});
  const [activeTip, setActiveTip] = useState<string | null>(null);
  const [rawData, setRawData] = useState<any[]>([]);
  const [voiceBlock, setVoiceBlock] = useState<{
    metrics: Record<string, unknown>;
    scores: Record<string, number>;
  } | null>(null);
  const [analysisStatus, setAnalysisStatus] = useState<string>("waiting");
  const [slideVerify, setSlideVerify] = useState<SlideVerifyResult | null>(null);
  const [pdfProgress, setPdfProgress] = useState<string>("");
  
  // 🌟 실시간 로딩용 타이머 및 상태
  const [analysisMessage, setAnalysisMessage] = useState<string>("0/6: 품질 검사 중...");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [progressVal, setProgressVal] = useState(0);
  const [timelineData, setTimelineData] = useState<any[]>([]);
  
  const [scoresRevision, setScoresRevision] = useState(0);

  const scores = useMemo<StoredRubricScores | null>(
    () => loadScoresForView(scopeId, submissionId),
    [scopeId, submissionId, fsRevision, scoresRevision]
  );

  const pieData = useMemo(() => {
    if (!scores) return [];
    const colors = ["#3b82f6", "#7b61ff", "#10b981"];
    return RUBRIC.map((cat, idx) => {
      const scoreData = scores[cat.id];
      return {
        name: cat.title.split(' ')[0],
        fullName: cat.title,
        value: scoreData?.category ?? 0,
        max: cat.maxScore,
        color: colors[idx % colors.length],
        subItems: cat.items.map((label, i) => ({
          label: label.split(' (')[0],
          score: scoreData?.items[i] ?? 0,
          max: cat.itemMaxes[i]
        }))
      };
    });
  }, [scores]);

  const radarData = useMemo(() => {
    if (!scores) return [];
    return RUBRIC.map((cat) => ({
      subject: cat.title.split(' ')[0],
      score: Math.round(((scores[cat.id]?.category ?? 0) / cat.maxScore) * 100),
      fullMark: 100,
    }));
  }, [scores]);

  const CustomPieTooltip = ({ active }: any) => {
    if (active) {
      return (
        <div className="analysis-pie-tooltip analysis-pie-tooltip--full">
          <div className="analysis-pie-tooltip__title">📊 전체 부문별 상세 점수</div>
          <div className="analysis-pie-tooltip__list">
            {pieData.map((entry) => (
              <div key={entry.name} className="analysis-pie-tooltip__item is-active">
                <div className="analysis-pie-tooltip__main">
                  <span className="analysis-pie-tooltip__dot" style={{ backgroundColor: entry.color }} />
                  <span className="analysis-pie-tooltip__name">{entry.name}</span>
                  <span className="analysis-pie-tooltip__score">{entry.value} / {entry.max}</span>
                </div>
                <div className="analysis-pie-tooltip__details">
                  {entry.subItems.map((sub, i) => (
                    <div key={i} className="analysis-pie-tooltip__sub-row">
                      <span className="analysis-pie-tooltip__sub-label">{sub.label}</span>
                      <span className="analysis-pie-tooltip__sub-val">{sub.score}/{sub.max}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <div className="analysis-pie-tooltip__footer">
            * 마우스를 올리면 모든 분석 항목이 한눈에 표시됩니다.
          </div>
        </div>
      );
    }
    return null;
  };

  // 비디오 재생 시간에 맞춰 실시간 피드백 업데이트 (동적 자막기 포함)
  const handleTimeUpdate = () => {
    if (!previewVideoRef.current) return;
    const time = previewVideoRef.current.currentTime;
    
    let matchedTip = "";

    // 1. 3-stage matching from timelineFeedback (Backend)
    if (timelineFeedback && Object.keys(timelineFeedback).length > 0) {
      const timeFixed = time.toFixed(1);
      const timeFixedNoZero = String(Math.round(time));
      
      matchedTip = timelineFeedback[timeFixed] || timelineFeedback[timeFixedNoZero] || timelineFeedback[String(time)] || "";
      
      if (!matchedTip) {
        const keys = Object.keys(timelineFeedback);
        let minDiff = 1.5;
        for (const key of keys) {
          const keyFloat = parseFloat(key);
          if (!isNaN(keyFloat)) {
            const diff = Math.abs(keyFloat - time);
            if (diff < minDiff) {
              minDiff = diff;
              matchedTip = timelineFeedback[key];
            }
          }
        }
      }
    }

    // 2. Dynamic Fallback Generator (If no matched tip or legacy single tip)
    const isLegacyOrEmpty = !timelineFeedback || Object.keys(timelineFeedback).length <= 1;
    if (isLegacyOrEmpty || !matchedTip) {
      const currentSegment = timelineData.find((seg: any) => time >= seg.start && time <= seg.end);
      
      // Find matching visual frame
      let currentFrame: any = null;
      if (rawData.length > 0) {
        const totalDuration = previewVideoRef.current.duration || 1;
        const expectedIdx = Math.round(time * (rawData.length / totalDuration));
        let minDiff = 2.5;

        for (let i = expectedIdx - 10; i <= expectedIdx + 10; i++) {
          if (i < 0 || i >= rawData.length) continue;
          const d = rawData[i];
          const diff = Math.abs(d.time - time);
          if (diff < minDiff) {
            minDiff = diff;
            currentFrame = d;
          }
        }
      }

      if (currentFrame || currentSegment) {
        const isArmCrossed = currentSegment?.is_arm_crossed ?? currentFrame?.yolo?.is_arm_crossed ?? false;
        const gazeH = currentFrame?.face?.gaze_h ?? 0;
        const gazeV = currentFrame?.face?.gaze_v ?? 0;
        const smile = currentFrame?.face?.smile ?? 0;
        const gesture = currentSegment?.dominant_gesture ?? currentFrame?.yolo?.gesture_name ?? "Unknown";

        const fillers = currentSegment?.fillers_count ?? 0;
        const speechRate = currentSegment?.speech_rate_cps ?? 0;

        if (isArmCrossed) {
          matchedTip = "⚠️ [자세 피드백] 발표 중 팔짱을 끼면 청중에게 방어적인 인상을 줄 수 있습니다. 양손을 가볍게 열어 신뢰감을 주도록 해 보세요.";
        } else if (gazeH > 0.08 || gazeH < -0.08 || gazeV > 0.35 || gazeV < -0.2) {
          matchedTip = "👁️ [시선 피드백] 현재 시선이 측면이나 스크린 바깥으로 분산되고 있습니다. 정면 청중을 부드럽게 응시하세요.";
        } else if (fillers > 0) {
          matchedTip = `🎙️ [발화 피드백] 이 구간에서 불필요한 말버릇이 쓰였습니다. 긴장될 땐 잠시 묵음(Pause)을 가져보세요.`;
        } else if (speechRate > 7.5) {
          matchedTip = `🎙️ [속도 피드백] 말이 조금 빠른 편입니다 (${speechRate.toFixed(1)} cps). 한 호흡 가다듬고 여유 있게 템포를 조절해 보세요.`;
        } else if (smile > 0.35) {
          matchedTip = "🌸 [표정 피드백] 아주 온화하고 밝은 미소로 신뢰감을 높여주고 있습니다. 매우 우수한 비언어적 커뮤니케이션입니다.";
        } else if (gesture !== "Unknown" && gesture !== "기본 자세" && gesture !== "Low") {
          matchedTip = `👍 [제스처 피드백] '${gesture}' 동작을 효과적으로 활용하여 시각적 집중력을 높이고 있습니다.`;
        } else {
          matchedTip = "🧘 [자세 평정심] 정면을 바르게 바라보며 정갈한 자세로 스피치를 전개하고 있습니다.";
        }
      }
    }

    if (matchedTip && matchedTip !== activeTip) {
      setActiveTip(matchedTip);
    }
  };

  const submissionMeta = useMemo(
    () => (submissionId ? findSubmissionById(scopeId, submissionId) : null),
    [scopeId, submissionId, fsRevision]
  );

  useEffect(() => {
    setSlideVerify(loadSlideVerifyForSubmission(submissionId));
  }, [submissionId]);

  // 슬라이드 일치 검증은 제출 후 백그라운드 완료 → 주기적으로 다시 읽기 (절대 손대지 말 것 - 100점 만점 구조에 맞춰 세부 항목만 유연하게 보정)
  useEffect(() => {
    if (!submissionId || slideVerify) return;
    const poll = setInterval(() => {
      const loaded = loadSlideVerifyForSubmission(submissionId);
      if (!loaded) return;
      setSlideVerify(loaded);
      const prev = loadScoresForView(scopeId, submissionId);
      if (prev) {
        // [수정금지 영역 유지 + 포맷만 보정] 
        // 전체 점수가 100점 만점이 되었으므로 Content 영역은 50점 만점으로 스케일링 필요
        const rawMatch = loaded.overall_match_percent; // 0~100
        const contentScore = Math.round(rawMatch * 0.5); // 50점 만점
        const item1 = Math.round(rawMatch * 0.15); // 논리 구조 (15)
        const item2 = Math.round(((rawMatch + loaded.visual_match_percent) / 2) * 0.10); // 헤드라인 전략 (10)
        const item3 = Math.round(loaded.visual_match_percent * 0.15); // 시각적 가독성 (15)
        const item4 = Math.round(loaded.slide_coverage_percent * 0.10); // 데이터 신뢰성 (10)
        
        saveAnalysisResultForSubmission(scopeId, submissionId, {
          ...prev,
          content: {
            category: contentScore,
            items: [item1, item2, item3, item4],
          },
        });
      }
      setScoresRevision((n) => n + 1);
    }, 2500);
    return () => clearInterval(poll);
  }, [submissionId, slideVerify, scopeId]);

  const slideVerifyReasons = useMemo(
    () => (slideVerify ? summarizeLowMatchReasons(slideVerify) : []),
    [slideVerify]
  );

  const hasData =
    scores !== null || overallFeedback !== null || voiceBlock !== null || slideVerify !== null;
  const total = useMemo(() => (scores ? totalFromScores(scores) : null), [scores]);
  const previewVideoUrl = useMemo(() => {
    if (!submissionId) return null;
    try {
      const raw = sessionStorage.getItem("overnight-video-preview-by-submission-v1");
      if (!raw) return null;
      const map = JSON.parse(raw) as Record<string, string>;
      return map[submissionId] ?? null;
    } catch {
      return null;
    }
  }, [submissionId]);

  // 로딩 진행바 타이머
  useEffect(() => {
    if (analysisStatus.toLowerCase() === "analyzing" || analysisStatus.toLowerCase() === "waiting" || analysisStatus.toLowerCase() === "checking") {
      const interval = setInterval(() => setElapsedSeconds(prev => prev + 1), 1000);
      return () => clearInterval(interval);
    } else {
      setElapsedSeconds(0);
    }
  }, [analysisStatus]);

  useEffect(() => {
    if (analysisStatus.toLowerCase() === "analyzing" || analysisStatus.toLowerCase() === "waiting" || analysisStatus.toLowerCase() === "checking") {
      const step = getActiveStep(analysisMessage);
      const stepBase = (step - 1) * 25;
      let t = 0;
      const progressTimer = setInterval(() => {
        t += 0.5;
        const decay = 1 - Math.exp(-t / 12);
        const addedProgress = decay * 24.5;
        setProgressVal(stepBase + addedProgress);
      }, 500);
      return () => clearInterval(progressTimer);
    } else {
      setProgressVal(0);
    }
  }, [analysisStatus, analysisMessage]);

  const formatMMSS = (sec: number): string => {
    const mins = Math.floor(sec / 60);
    const secs = sec % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  // 서버 분석 결과 폴링 및 데이터 로드
  useEffect(() => {
    if (!submissionId) return;

    let jobId: string | null = null;
    try {
      const raw = sessionStorage.getItem("overnight-analysis-job-ids-v1");
      if (raw) {
        const map = JSON.parse(raw) as Record<string, string>;
        jobId = map[submissionId] ?? null;
      }
    } catch {}

    if (!jobId) {
      setAnalysisStatus("no_job");
      return;
    }

    let timerId: ReturnType<typeof setInterval>;

    const checkStatus = async () => {
      try {
        const res = await fetch(`http://127.0.0.1:8000/api/status/${jobId}`);
        if (!res.ok) return;
        const data = await res.json();
        
        setAnalysisStatus(data.status);
        if (data.message) {
          setAnalysisMessage(data.message);
        }
        
        if (data.status.toLowerCase() === "complete" && data.result) {
          const resData = data.result;
          setOverallFeedback(resData.llama_feedback);
          setTimelineFeedback(resData.timeline_feedback || {});
          setRawData(resData.raw_data || []);
          setTimelineData(resData.timeline_data || resData.aligned_transcript_data || []);
          
          const sum = resData.analysis_summary;
          if (sum?.voice_metrics && Object.keys(sum.voice_metrics).length > 0) {
            setVoiceBlock({
              metrics: sum.voice_metrics as Record<string, unknown>,
              scores: (sum.voice_scores || {}) as Record<string, number>,
            });
          } else {
            setVoiceBlock(null);
          }

          if (resData.ai_scores) {
            // 🌟 AI가 직접 채점한 점수가 존재하면 복잡한 하드코딩 수식 없이 100% 동기화 매핑!
            import("../data/analysisResultStorage").then(({ saveAnalysisResultForSubmission }) => {
              saveAnalysisResultForSubmission(scopeId, submissionId, resData.ai_scores);
              setScoresRevision((n) => n + 1);
            });
          } else if (resData.analysis_summary) {
            // 🛡️ 백엔드 연동 폴백 및 로컬 점수 계산 수식 구동
            const summary = resData.analysis_summary;
            import("../data/analysisResultStorage").then(({ saveAnalysisResultForSubmission }) => {
              const gazeScoreVal = Math.round((summary.gaze_score || 0.8) * 100);
              const smileScoreVal = Math.round((summary.smile_score || 0.5) * 100);
              const gestureScoreVal = summary.gesture_status === "활발함" ? 90 : 70;
              
              const attitudeItems = [
                Math.round(gazeScoreVal * 0.10),
                Math.round((smileScoreVal * 0.5 + gestureScoreVal * 0.5) * 0.10)
              ];
              const attitudeScore = attitudeItems[0] + attitudeItems[1];
              
              const vScores = summary.voice_scores || {};
              const vItems = vScores.items_100 || [80, 80, 80, 80];
              const stabScore = Math.round((vScores.voice_stability_item ?? vItems[1] ?? 80) / 10);
              const bodyStab = Math.max(1, Math.min(10, Math.round((gazeScoreVal * 0.5 + (100 - (summary.voice_metrics?.fillers_per_minute ?? 0) * 15)) / 20)));
              const fluency = Math.round((vScores.filler_control ?? vItems[2] ?? 80) / 10);
              
              const voiceItems = [stabScore, bodyStab, fluency];
              const voiceScore = Math.max(0, Math.min(30, voiceItems.reduce((a, b) => a + b, 0)));
              
              const storedVerify = loadSlideVerifyForSubmission(submissionId);
              let contentScore = 25;
              let contentItems = [8, 5, 8, 4];
              if (storedVerify) {
                const rawMatch = storedVerify.overall_match_percent;
                contentScore = Math.round(rawMatch * 0.5);
                contentItems = [
                  Math.round(rawMatch * 0.15),
                  Math.round(((rawMatch + storedVerify.visual_match_percent) / 2) * 0.10),
                  Math.round(storedVerify.visual_match_percent * 0.15),
                  Math.round(storedVerify.slide_coverage_percent * 0.10),
                ];
              }
              
              const speechErrorCnt = summary.voice_metrics?.repeated_phrase_hits ?? 0;
              const hasPptData = summary.ppt_summary !== "PPT 분석 데이터 없음";
              
              if (speechErrorCnt > 0 || !hasPptData) {
                const penaltyFactor = !hasPptData ? 0.45 : 0.7;
                contentScore = Math.round(contentScore * penaltyFactor);
                contentItems = contentItems.map(item => Math.round(item * penaltyFactor));
              }

              const calculatedScores: any = {
                "attitude": { category: attitudeScore, items: attitudeItems },
                "content": { category: contentScore, items: contentItems },
                "voice": { category: voiceScore, items: voiceItems }
              };
              saveAnalysisResultForSubmission(scopeId, submissionId, calculatedScores);
              setScoresRevision((n) => n + 1);
            });
          }
          
          clearInterval(timerId);
        } else if (data.status.toLowerCase() === "error") {
          clearInterval(timerId);
        }
      } catch (e) {
        console.error("Status check failed", e);
      }
    };

    timerId = setInterval(checkStatus, 3000);
    checkStatus();

    return () => clearInterval(timerId);
  }, [submissionId]);

  if (
    analysisStatus.toLowerCase() === "analyzing" ||
    analysisStatus.toLowerCase() === "waiting" ||
    analysisStatus.toLowerCase() === "checking"
  ) {
    const step = getActiveStep(analysisMessage);
    return (
      <div className="analysis-loading-page">
        <div className="loading-container">
          <div className="loading-card">
            <div className="loading-spinner-wrapper">
              <div className="loading-spinner-outer" />
              <div className="loading-spinner-inner" />
              <div className="loading-timer">{formatMMSS(elapsedSeconds)}</div>
            </div>

            <h2 className="loading-title">발표 멀티모달 분석 실행 중</h2>
            <p className="loading-status-text">{analysisMessage}</p>

            <div className="loading-progressbar-container">
              <div className="loading-progressbar-track">
                <div
                  className="loading-progressbar-fill"
                  style={{ width: `${Math.min(99, Math.max(5, progressVal))}%` }}
                />
              </div>
              <div className="loading-progressbar-meta">
                <span className="loading-status-text">분석 진행 중...</span>
                <span className="loading-percent-text">{Math.round(Math.min(99, Math.max(5, progressVal)))}%</span>
              </div>
            </div>

            <div className="loading-steps-indicator">
              <div className={`loading-step ${step >= 1 ? "active" : ""} ${step > 1 ? "completed" : ""}`}>
                <div className="step-dot" />
                <span className="step-name">전송 & 검증</span>
              </div>
              <div className={`loading-step ${step >= 2 ? "active" : ""} ${step > 2 ? "completed" : ""}`}>
                <div className="step-dot" />
                <span className="step-name">YOLO 제스처</span>
              </div>
              <div className={`loading-step ${step >= 3 ? "active" : ""} ${step > 3 ? "completed" : ""}`}>
                <div className="step-dot" />
                <span className="step-name">Whisper & Praat</span>
              </div>
              <div className={`loading-step ${step >= 4 ? "active" : ""}`}>
                <div className="step-dot" />
                <span className="step-name">AI 피드백</span>
              </div>
            </div>

            <p className="loading-tip-msg">{getDynamicLoadingMessage(step)}</p>
          </div>
        </div>
      </div>
    );
  }

  const emptyDesc = "이 제출에 대한 채점 결과가 아직 없습니다. 분석을 시작해 보세요.";

  return (
    <div className="page analysis">
      <div className="page-inner page-inner--wide">
        <p className="analysis-kicker">시각화 대시보드</p>
        <h1 className="analysis-page-title">멀티모달 채점 결과</h1>
        {submissionMeta ? (
          <p className="analysis-page-desc analysis-page-desc--meta">
            <strong>{submissionPrimaryFileName(submissionMeta)}</strong>
            <span className="analysis-page-desc__sep" aria-hidden>
              {" "}
              ·{" "}
            </span>
            제출 시각 기준 기록입니다. 발표 기록에서 다른 제출을 고르면 해당 결과로 바뀝니다.
          </p>
        ) : null}
        {hasData ? (
          <p className="analysis-page-desc">
            음성·영상을 함께 본 항목별 점수입니다. 결과를 확인하고 아래에서 PDF·Excel로 내보낼 수 있습니다.
          </p>
        ) : (
          <p className="analysis-page-desc">{emptyDesc}</p>
        )}

        <div
          className={
            "analysis-player" + (!hasData ? " analysis-player--placeholder" : "")
          }
        >
          <button
            type="button"
            className="analysis-play"
            aria-label="재생"
            disabled={!previewVideoUrl}
            onClick={() => {
              if (!previewVideoRef.current) return;
              previewVideoRef.current.play().catch(() => {
                /* ignore autoplay/play errors */
              });
            }}
          >
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" aria-hidden>
              <path d="M8 5v14l11-7-11-7z" fill="currentColor" />
            </svg>
          </button>
          {previewVideoUrl ? (
            <video
              ref={previewVideoRef}
              className="analysis-player__video"
              src={previewVideoUrl}
              controls
              playsInline
              onTimeUpdate={handleTimeUpdate}
            />
          ) : null}
          <span className="analysis-player__cap">
            {previewVideoUrl ? "발표 영상 다시보기" : "영상 미리보기가 없습니다"}
          </span>
        </div>

        {activeTip && (
          <div className="analysis-live-tip">
            <span className="analysis-live-tip__icon">💡</span>
            <span className="analysis-live-tip__text">{activeTip}</span>
          </div>
        )}

        {!slideVerify && submissionId && analysisStatus !== "no_job" ? (
          <section className="analysis-section analysis-slide-verify analysis-slide-verify--pending">
            <h2>PPT 슬라이드 일치 (영상 대조)</h2>
            <p className="analysis-slide-verify__formula">
              슬라이드 일치 분석 중입니다… (PPT 렌더 · 영상 프레임 추출 · 30초~2분)
            </p>
          </section>
        ) : null}

        {slideVerify ? (
          <section className="analysis-section analysis-slide-verify">
            <h2>PPT 슬라이드 일치 (영상 대조)</h2>
            <div className="analysis-slide-verify__head">
              <span
                className={
                  "analysis-slide-verify__badge analysis-slide-verify__badge--" +
                  slideVerify.verdict
                }
              >
                {verdictLabelKo(slideVerify.verdict)}
              </span>
              <div className="analysis-slide-verify__percent">
                <span className="analysis-slide-verify__num">
                  {slideVerify.overall_match_percent}
                </span>
                <span className="analysis-slide-verify__unit">% 종합 일치</span>
              </div>
            </div>
            <p className="analysis-slide-verify__formula">
              화면 유사도 {slideVerify.visual_match_percent}% × 슬라이드 커버리지{" "}
              {slideVerify.slide_coverage_percent}%
              {slideVerify.video_type === "audience" ? " · 제3자 촬영 모드" : ""}
            </p>
            {slideVerify.diagnostics?.slide_present_rate != null &&
            slideVerify.video_type === "audience" ? (
              <p className="analysis-slide-verify__diag">
                스크린 검출{" "}
                {Math.round((slideVerify.diagnostics.slide_present_rate ?? 0) * 100)}% · 원근 OK{" "}
                {Math.round((slideVerify.diagnostics.perspective_ok_rate ?? 0) * 100)}%
              </p>
            ) : null}
            {slideVerify.detected_slide_sequence?.length > 0 ? (
              <p className="analysis-slide-verify__seq">
                영상에서 감지한 슬라이드 순서:{" "}
                {slideVerify.detected_slide_sequence.join(" → ")}장
              </p>
            ) : null}
            {slideVerify.overall_match_percent < 70 && slideVerifyReasons.length > 0 ? (
              <div className="analysis-slide-verify__reasons">
                <h3>일치율이 낮은 이유</h3>
                <ul>
                  {slideVerifyReasons.map((r) => (
                    <li key={r}>{r}</li>
                  ))}
                </ul>
              </div>
            ) : null}
            {(slideVerify.issues?.length ?? 0) > 0 ? (
              <div className="analysis-slide-verify__issues">
                <h3>상세 이슈</h3>
                <ul>
                  {slideVerify.issues.slice(0, 12).map((issue, idx) => (
                    <li key={`${issue.issue_type}-${idx}`}>{issue.message}</li>
                  ))}
                </ul>
              </div>
            ) : null}
          </section>
        ) : null}

        {voiceBlock ? (
          <section className="analysis-section">
            <h2>발표 음성 (음성 분석)</h2>
            <div className="analysis-voice-metrics">
              <div className="analysis-voice-metric">
                <span className="analysis-voice-metric__label">말버릇(필러)</span>
                <strong className="analysis-voice-metric__val">{String(voiceBlock.metrics.filler_total ?? 0)}</strong>
                <span className="analysis-voice-metric__unit">
                  회 (분당 {Number(voiceBlock.metrics.fillers_per_minute ?? 0).toFixed(1)})
                </span>
              </div>
              <div className="analysis-voice-metric">
                <span className="analysis-voice-metric__label">반복 구절</span>
                <strong className="analysis-voice-metric__val">{String(voiceBlock.metrics.repeated_phrase_hits ?? 0)}</strong>
                <span className="analysis-voice-metric__unit">건</span>
              </div>
              <div className="analysis-voice-metric">
                <span className="analysis-voice-metric__label">긴 무음</span>
                <strong className="analysis-voice-metric__val">{String(voiceBlock.metrics.silence_pause_count ?? 0)}</strong>
                <span className="analysis-voice-metric__unit">
                  회 · {Number(voiceBlock.metrics.silence_total_sec ?? 0).toFixed(1)}초
                </span>
              </div>
              <div className="analysis-voice-metric">
                <span className="analysis-voice-metric__label">말 속도</span>
                <strong className="analysis-voice-metric__val">
                  {Number(voiceBlock.metrics.avg_speech_rate_cps ?? 0).toFixed(2)}
                </strong>
                <span className="analysis-voice-metric__unit">글자/초</span>
              </div>
            </div>
          </section>
        ) : null}

        {hasData && pieData.length > 0 && (
          <section className="analysis-section analysis-charts-dashboard">
            <div className="analysis-section__header-group">
              <span className="analysis-section__icon">📈</span>
              <h2>분석 지표 시각화</h2>
            </div>
            
            <div className="analysis-charts-grid">
              <div className="analysis-chart-card analysis-chart-card--radar">
                <div className="analysis-chart-card__header">
                  <h3>발표 역량 밸런스</h3>
                  <p>영역별 만점 대비 달성률(%)</p>
                </div>
                <div className="analysis-chart-wrapper">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
                      <PolarGrid stroke="#e5e7eb" />
                      <PolarAngleAxis dataKey="subject" tick={{ fill: '#4b5563', fontSize: 12, fontWeight: 600 }} />
                      <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                      <Radar
                        name="발표 점수"
                        dataKey="score"
                        stroke="#7b61ff"
                        fill="#7b61ff"
                        fillOpacity={0.6}
                        animationBegin={300}
                        animationDuration={1500}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="analysis-chart-card analysis-chart-card--pie">
                <div className="analysis-chart-card__header">
                  <h3>평가 부문별 비중</h3>
                  <p>100점 만점 내 영역별 배점</p>
                </div>
                <div className="analysis-chart-wrapper">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={pieData}
                        cx="50%"
                        cy="50%"
                        innerRadius="55%"
                        outerRadius="75%"
                        paddingAngle={8}
                        dataKey="value"
                        stroke="none"
                        animationBegin={500}
                        animationDuration={1500}
                      >
                        {pieData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip content={<CustomPieTooltip />} />
                      <Legend 
                        verticalAlign="bottom" 
                        align="center" 
                        iconType="circle"
                        wrapperStyle={{ paddingTop: "20px", fontSize: "12px", fontWeight: 600 }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </section>
        )}

        {overallFeedback && (
          <section className="analysis-section">
            <h2>AI 전문가 심층 피드백</h2>
            <div className="analysis-feedback-card">
              <div className="analysis-feedback-content">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{overallFeedback}</ReactMarkdown>
              </div>
            </div>
          </section>
        )}

        <section className="analysis-section">
          <h2>종합</h2>
          <div
            className={
              "analysis-total" + (hasData ? " analysis-total--filled" : " analysis-total--empty")
            }
          >
            <span className="analysis-total__label">Total</span>
            <div className="analysis-total__score" aria-live="polite">
              {hasData && total !== null ? (
                <>
                  <span className="analysis-total__num">{total}</span>
                  <span className="analysis-total__max">/ 100</span>
                </>
              ) : (
                <span className="analysis-total__num analysis-total__num--empty">—</span>
              )}
            </div>
            <p className="analysis-total__note">
              {hasData
                ? "발표 내용(50) · 목소리(30) · 태도(20) 영역 점수를 종합해 계산한 결과입니다."
                : "채점 결과가 있으면 종합 점수가 계산됩니다."}
            </p>
          </div>
        </section>

        <section className="analysis-section">
          <h2>항목별 상세 채점 결과</h2>
          <div className="analysis-rubric">
            {RUBRIC.map((cat) => {
              const d = scores?.[cat.id];
              return (
                <div
                  key={cat.id}
                  className={"analysis-cat" + (!hasData ? " analysis-cat--empty" : "")}
                >
                  <div className="analysis-cat__head">
                    <div>
                      <h3>{cat.title}</h3>
                      <p className="analysis-cat__sub">{cat.subtitle}</p>
                    </div>
                    <span
                      className={
                        "analysis-cat__badge" + (!hasData ? " analysis-cat__badge--empty" : "")
                      }
                    >
                      {d ? `${d.category}점 / ${cat.maxScore}점` : "—"}
                    </span>
                  </div>
                  <ul className="analysis-cat__items">
                    {cat.items.map((label, i) => (
                      <li key={label}>
                        <span className="analysis-cat__label">{label}</span>
                        <span className="analysis-cat__itemscore">
                          {d?.items[i] != null ? `${d.items[i]}점 / ${cat.itemMaxes[i]}점` : "—"}
                        </span>
                      </li>
                    ))}
                  </ul>
                  <div className="analysis-bar" aria-hidden>
                    <span
                      className={!hasData ? "analysis-bar__fill analysis-bar__fill--empty" : "analysis-bar__fill"}
                      style={
                        hasData && d ? { width: `${Math.min(100, Math.max(0, (d.category / cat.maxScore) * 100))}%` } : { width: 0 }
                      }
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        <section className="analysis-section">
          <h2>리포트 내보내기</h2>
          <div className="analysis-export">
            <button
              type="button"
              className="analysis-btn analysis-btn--outline"
              disabled={!hasData || !!pdfProgress}
              title={!hasData ? "채점 결과가 있을 때 사용할 수 있습니다" : undefined}
              onClick={() => {
                exportExcel(submissionMeta, scores, voiceBlock, slideVerify, overallFeedback);
              }}
            >
              EXCEL
            </button>
            <button
              type="button"
              className="analysis-btn analysis-btn--fill"
              disabled={!hasData || !!pdfProgress}
              title={!hasData ? "채점 결과가 있을 때 사용할 수 있습니다" : undefined}
              onClick={async () => {
                try {
                  await exportPDF(
                    submissionMeta,
                    scores,
                    voiceBlock,
                    slideVerify,
                    overallFeedback,
                    setPdfProgress
                  );
                } catch (e) {
                  console.error("PDF export failed:", e);
                  alert("PDF 내보내기 중 오류가 발생했습니다.");
                  setPdfProgress("");
                }
              }}
            >
              {pdfProgress || "PDF"}
            </button>
          </div>
        </section>

        <p className="analysis-foot">
          <Link to="/notes">발표 기록</Link>
          <span aria-hidden> · </span>
          <Link to="/evaluate">다시 평가하기</Link>
          <span aria-hidden> · </span>
          <Link to="/mypage">마이페이지로</Link>
        </p>
      </div>
    </div>
  );
}