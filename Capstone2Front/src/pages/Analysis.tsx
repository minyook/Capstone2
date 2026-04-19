import { useMemo } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { findSubmissionById, submissionPrimaryFileName } from "../data/folderFilesStorage";
import { loadScoresForView, totalFromScores, type StoredRubricScores } from "../data/analysisResultStorage";
import { RUBRIC } from "../data/rubric";
import "./Analysis.css";

export function Analysis() {
  const [searchParams] = useSearchParams();
  const submissionId = searchParams.get("submissionId");

  const scores = useMemo<StoredRubricScores | null>(
    () => loadScoresForView(submissionId),
    [submissionId]
  );
  const submissionMeta = useMemo(
    () => (submissionId ? findSubmissionById(submissionId) : null),
    [submissionId]
  );

  const hasData = scores !== null;
  const total = useMemo(() => (scores ? totalFromScores(scores) : null), [scores]);

  const emptyDesc =
    submissionId && !hasData
      ? "이 제출에 대한 채점 결과가 아직 없습니다. 서버 분석이 끝나면 같은 화면에서 점수를 불러올 수 있습니다."
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
            음성·영상을 함께 본 항목별 점수입니다. 필요하면 <strong>가중치</strong>를 조정할 수 있고, 아래에서 PDF·Excel로
            내보낼 수 있습니다.
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
            disabled={!hasData}
          >
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" aria-hidden>
              <path d="M8 5v14l11-7-11-7z" fill="currentColor" />
            </svg>
          </button>
          <span className="analysis-player__cap">
            {hasData ? "발표 영상 다시보기" : "채점 완료 후 영상을 연결할 수 있습니다"}
          </span>
        </div>

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
                ? "발표 내용 · 태도 · 음성 영역 가중 평균 (가중치는 교수 설정에 맞게 조정 가능)"
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
            >
              EXCEL
            </button>
            <button
              type="button"
              className="analysis-btn analysis-btn--fill"
              disabled={!hasData}
              title={!hasData ? "채점 결과가 있을 때 사용할 수 있습니다" : undefined}
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
