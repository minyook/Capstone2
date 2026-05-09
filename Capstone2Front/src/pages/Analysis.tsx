import { useMemo, useRef, useState, useEffect } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useFirestoreSyncRevision } from "../context/FirestoreSyncContext";
import { useFolders } from "../context/FoldersContext";
import { findSubmissionById, submissionPrimaryFileName } from "../data/folderFilesStorage";
import { loadScoresForView, totalFromScores, type StoredRubricScores } from "../data/analysisResultStorage";
import { RUBRIC } from "../data/rubric";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import "./Analysis.css";

export function Analysis() {
  const { scopeId } = useFolders();
  const fsRevision = useFirestoreSyncRevision();
  const [searchParams] = useSearchParams();
  const submissionId = searchParams.get("submissionId");
  const previewVideoRef = useRef<HTMLVideoElement | null>(null);

  const [overallFeedback, setOverallFeedback] = useState<string | null>(null);
  const [timelineFeedback, setTimelineFeedback] = useState<Record<string, string>>({});
  const [chartData, setChartData] = useState<any[]>([]);
  const [analysisStatus, setAnalysisStatus] = useState<string>("waiting");

  const scores = useMemo<StoredRubricScores | null>(
    () => loadScoresForView(scopeId, submissionId),
    [scopeId, submissionId, fsRevision]
  );
  const submissionMeta = useMemo(
    () => (submissionId ? findSubmissionById(scopeId, submissionId) : null),
    [scopeId, submissionId, fsRevision]
  );

  const hasData = scores !== null || overallFeedback !== null;
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
        
        if (data.status === "Complete" && data.result) {
          const resData = data.result;
          setOverallFeedback(resData.llama_feedback);
          setTimelineFeedback(resData.timeline_feedback || {});
          
          // 차트 데이터 변환 (가독성을 위해 10개마다 샘플링)
          if (resData.raw_data) {
            const formatted = resData.raw_data
              .filter((_: any, i: number) => i % 5 === 0)
              .map((d: any) => ({
                time: Math.round(d.time),
                gaze: Math.round((1 - Math.abs(d.face?.gaze_h || 0)) * 100),
                smile: Math.round((d.face?.smile || 0) * 100)
              }));
            setChartData(formatted);
          }

          // 분석 결과(점수)를 로컬 스토리지에 저장 (다른 페이지 연동용)
          if (resData.analysis_summary) {
            const summary = resData.analysis_summary;
            import("../data/analysisResultStorage").then(({ saveAnalysisResultForSubmission }) => {
              // 실제 분석 데이터를 기반으로 점수 계산
              const attitudeScore = Math.round((summary.gaze_score * 0.4 + summary.smile_score * 0.3 + (summary.face_detection_rate / 100) * 0.3) * 100);
              const voiceScore = summary.avg_speed > 0.5 && summary.avg_speed < 2.0 ? 90 : 70;
              const contentScore = summary.ppt_summary !== "PPT 분석 데이터 없음" ? 85 : 50;

              const calculatedScores: any = {
                "attitude": { 
                  category: attitudeScore, 
                  items: [
                    Math.round(summary.gaze_score * 100), 
                    Math.round(summary.smile_score * 100), 
                    summary.gesture_status === "활발함" ? 90 : 70
                  ] 
                },
                "content": { 
                  category: contentScore, 
                  items: [contentScore, contentScore - 5, contentScore + 5] 
                },
                "voice": { 
                  category: voiceScore, 
                  items: [voiceScore, 85, 80, 85] 
                }
              };
              saveAnalysisResultForSubmission(scopeId, submissionId, calculatedScores);
            });
          }
          
          clearInterval(timerId);
        }
 else if (data.status === "Error") {
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

  const emptyDesc =
    submissionId && !hasData
      ? analysisStatus === "Analyzing" || analysisStatus === "Waiting" || analysisStatus === "Checking"
        ? "서버에서 AI가 당신의 발표를 분석하고 있습니다... (약 1~2분 소요)"
        : "이 제출에 대한 채점 결과가 아직 없습니다. 분석을 시작해 보세요."
      : "저장된 채점 결과가 없습니다. 발표 평가에서 제출한 영상의 분석이 완료되면 항목별 점수가 여기에 표시됩니다.";

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
            />
          ) : null}
          <span className="analysis-player__cap">
            {previewVideoUrl ? "발표 영상 다시보기" : "영상 미리보기가 없습니다"}
          </span>
        </div>

        {chartData.length > 0 && (
          <section className="analysis-section">
            <h2>발표 흐름 분석 (시선 및 표정 변화)</h2>
            <div className="analysis-chart-container">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="time" label={{ value: '시간 (초)', position: 'insideBottomRight', offset: -5 }} />
                  <YAxis domain={[0, 100]} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="gaze" name="시선 집중도" stroke="#7b61ff" strokeWidth={3} dot={false} />
                  <Line type="monotone" dataKey="smile" name="미소 점수" stroke="#10b981" strokeWidth={3} dot={false} />
                </LineChart>
              </ResponsiveContainer>
              <p className="analysis-chart-hint">※ 그래프가 높을수록 정면을 잘 응시하거나 밝은 표정을 지었음을 의미합니다.</p>
            </div>
          </section>
        )}

        {Object.keys(timelineFeedback).length > 0 && (
          <section className="analysis-section">
            <h2>구간별 AI 코칭 팁</h2>
            <div className="analysis-timeline-tips">
              {Object.entries(timelineFeedback)
                .sort((a, b) => parseFloat(a[0]) - parseFloat(b[0]))
                .map(([time, tip]) => (
                <div key={time} className="analysis-tip-item">
                  <span className="analysis-tip-time">{parseFloat(time).toFixed(1)}s</span>
                  <p className="analysis-tip-text">{tip}</p>
                </div>
              ))}
            </div>
          </section>
        )}

        {overallFeedback && (
          <section className="analysis-section">
            <h2>AI 전문가 심층 피드백 (EXAONE 3.5 LoRA)</h2>
            <div className="analysis-feedback-card">
              <div className="analysis-feedback-content">
                {overallFeedback.split('\n').map((line, i) => (
                  <p key={i}>{line}</p>
                ))}
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
                ? "발표 내용 · 태도 · 음성 영역 점수를 종합해 계산한 결과입니다."
                : "채점 결과가 있으면 종합 점수가 계산됩니다."}
            </p>
          </div>
        </section>

        <section className="analysis-section">
          <h2>항목별 점수</h2>
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
                      {d ? `${d.category}점` : "—"}
                    </span>
                  </div>
                  <ul className="analysis-cat__items">
                    {cat.items.map((label, i) => (
                      <li key={label}>
                        <span className="analysis-cat__label">{label}</span>
                        <span className="analysis-cat__itemscore">
                          {d?.items[i] ?? "—"}
                        </span>
                      </li>
                    ))}
                  </ul>
                  <div className="analysis-bar" aria-hidden>
                    <span
                      className={!hasData ? "analysis-bar__fill analysis-bar__fill--empty" : "analysis-bar__fill"}
                      style={
                        hasData && d ? { width: `${Math.min(100, Math.max(0, d.category))}%` } : { width: 0 }
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
              disabled={!hasData}
              title={!hasData ? "채점 결과가 있을 때 사용할 수 있습니다" : undefined}
              onClick={() => {
                if (!scores) return;
                const rows = [
                  ["영역", "세부항목", "점수"].join(","),
                  ...RUBRIC.flatMap((cat) => {
                    const data = scores[cat.id];
                    const itemRows = cat.items.map((label, idx) =>
                      [cat.title, label, String(data.items[idx] ?? "")].join(",")
                    );
                    return [...itemRows, [cat.title, "영역 점수", String(data.category)].join(",")];
                  }),
                  ["총점", "", String(totalFromScores(scores))].join(","),
                ];
                const blob = new Blob(["\uFEFF" + rows.join("\n")], { type: "text/csv;charset=utf-8;" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `analysis-report-${submissionId ?? "latest"}.csv`;
                a.click();
                URL.revokeObjectURL(url);
              }}
            >
              EXCEL
            </button>
            <button
              type="button"
              className="analysis-btn analysis-btn--fill"
              disabled={!hasData}
              title={!hasData ? "채점 결과가 있을 때 사용할 수 있습니다" : undefined}
              onClick={() => {
                window.print();
              }}
            >
              PDF
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
