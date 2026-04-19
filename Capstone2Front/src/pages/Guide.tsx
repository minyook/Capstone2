import { useState } from "react";
import { Link } from "react-router-dom";
import { IconArrowLeft } from "../components/Icons";
import { RUBRIC } from "../data/rubric";
import "./Guide.css";

type Tab = "criteria" | "howto";

const cardClass: Record<string, string> = {
  content: "c-card--blue",
  attitude: "c-card--mint",
  voice: "c-card--lavender",
};

const iconClass: Record<string, string> = {
  content: "c-card__icon--blue",
  attitude: "c-card__icon--mint",
  voice: "c-card__icon--lavender",
};

export function Guide() {
  const [tab, setTab] = useState<Tab>("criteria");

  return (
    <div className="page guide">
      <div className="page-inner page-inner--wide">
        <header className="guide-top">
          <Link to="/" className="guide-back" aria-label="홈으로">
            <IconArrowLeft />
          </Link>
          <h1 className="guide-page-title">가이드</h1>
        </header>

        <div className="guide-tabs" role="tablist" aria-label="가이드 구분">
          <button
            type="button"
            role="tab"
            aria-selected={tab === "criteria"}
            className={"guide-tab" + (tab === "criteria" ? " guide-tab--on" : "")}
            onClick={() => setTab("criteria")}
          >
            채점 항목
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === "howto"}
            className={"guide-tab" + (tab === "howto" ? " guide-tab--on" : "")}
            onClick={() => setTab("howto")}
          >
            이용 방법
          </button>
        </div>

        {tab === "criteria" && (
          <>
            <p className="guide-intro">
              슬라이드와 말하는 내용을 비교하고, 영상·음성을 함께 보며 아래 세 영역을 채점합니다. 항목별 피드백은 결과
              화면에서 확인할 수 있습니다.
            </p>

            <div className="guide-criteria-grid">
              {RUBRIC.map((cat) => (
                <article key={cat.id} className={`c-card ${cardClass[cat.id]}`}>
                  <div className={`c-card__icon ${iconClass[cat.id]}`}>
                    {cat.id === "content" && (
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden>
                        <path d="M4 6h16v12H4V6z" stroke="currentColor" strokeWidth="2" />
                        <path d="M8 10h8M8 14h5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                      </svg>
                    )}
                    {cat.id === "attitude" && (
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden>
                        <rect x="3" y="5" width="18" height="14" rx="2" stroke="currentColor" strokeWidth="2" />
                        <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2" />
                      </svg>
                    )}
                    {cat.id === "voice" && (
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden>
                        <path d="M12 14a3 3 0 003-3V5a3 3 0 10-6 0v6a3 3 0 003 3z" fill="currentColor" />
                      </svg>
                    )}
                  </div>
                  <h3>{cat.title}</h3>
                  <p className="c-card__summary">{cat.summary}</p>
                  <ul className="c-card__list">
                    {cat.items.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </article>
              ))}
            </div>

            <div className="c-note">
              <span className="c-note__emoji" aria-hidden>
                💯
              </span>
              <p>
                종합 점수는 <strong>발표 내용 · 발표 태도 · 발표 음성</strong> 영역에 가중치를 두어 산출하며, PPT와 발화
                텍스트 비교는 <strong>발표 내용</strong> 영역에 반영됩니다.
              </p>
            </div>
          </>
        )}

        {tab === "howto" && (
          <section className="howto" aria-labelledby="howto-title">
            <header className="howto__header">
              <h2 id="howto-title" className="howto__title">
                처음부터 끝까지, 이렇게 쓰면 됩니다
              </h2>
              <p className="howto__lead">
                녹화·PPT·채점 기록은 <strong>폴더 하나</strong>에 모입니다. 슬라이드와 말하는 내용, 영상·음성을 함께
                분석해 피드백으로 이어집니다.
              </p>
            </header>

            <div className="howto__steps">
              <article className="howto-step howto-step--1">
                <div className="howto-step__badge" aria-hidden>
                  1
                </div>
                <div className="howto-step__body">
                  <h3 className="howto-step__name">폴더 만들기 · 선택</h3>
                  <p className="howto-step__text">
                    <Link to="/projects" className="howto-step__link">
                      문서
                    </Link>
                    에서 주제별로 새 폴더를 만들거나 기존 폴더를 고릅니다. 그 안에 영상·PPT·채점 결과가 같이 쌓입니다.
                  </p>
                  <p className="howto-step__text">
                    <Link to="/evaluate" className="howto-step__link">
                      발표 평가
                    </Link>
                    를 시작할 때도 &lsquo;저장할 발표 폴더&rsquo;를 먼저 선택하면 같은 방식으로 정리됩니다.
                  </p>
                </div>
              </article>

              <article className="howto-step howto-step--2">
                <div className="howto-step__badge" aria-hidden>
                  2
                </div>
                <div className="howto-step__body">
                  <h3 className="howto-step__name">PPT와 영상 올리기</h3>
                  <p className="howto-step__text">
                    선택한 폴더를 기준으로 PPT를 업로드하고, 카메라로 녹화하거나 영상 파일을 올립니다. 슬라이드와 실제
                    발화를 맞춰 보며 <strong>내용 일치와 논리 흐름</strong>을 평가합니다.
                  </p>
                </div>
              </article>

              <article className="howto-step howto-step--3">
                <div className="howto-step__badge" aria-hidden>
                  3
                </div>
                <div className="howto-step__body">
                  <h3 className="howto-step__name">점수 확인 · 리포트</h3>
                  <p className="howto-step__text">
                    시선·표정·제스처 같은 <strong>발표 태도</strong>와 말하기 속도·말버릇 등 <strong>발표 음성</strong>은
                    영상·음성 분석 결과로 점수에 반영됩니다.
                  </p>
                  <p className="howto-step__text">
                    항목별 점수는 화면에서 확인하고, 필요하면 Excel이나 PDF로 내려받을 수 있습니다.
                  </p>
                </div>
              </article>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
