import { Link } from "react-router-dom";
import "./Home.css";

const FEATURES = [
  {
    title: "음성",
    desc: "말하는 내용을 글로 옮겨 억양·말버릇까지 살펴봅니다.",
    icon: "🎙️",
  },
  {
    title: "영상",
    desc: "시선·제스처·표정을 수치로 정리합니다.",
    icon: "🎬",
  },
  {
    title: "채점",
    desc: "항목별 점수를 합산하고, 교수 설정에 맞게 가중치를 조정할 수 있습니다.",
    icon: "📊",
  },
  {
    title: "리포트",
    desc: "피드백을 PDF·Excel로 내보내 보관·제출하기 쉽게 정리합니다.",
    icon: "📄",
  },
] as const;

export function Home() {
  return (
    <div className="page landing">
      <div className="landing__mesh" aria-hidden />
      <div className="landing__inner">
        <section className="landing__hero" aria-labelledby="landing-title">
          <div className="landing__hero-copy">
            <p className="landing__eyebrow">AI Presentation Grading</p>
            <h1 id="landing-title" className="landing__title">
              OvernightAI
            </h1>
            <p className="landing__lead">
              음성과 영상을 함께 분석해 <strong>내용·논리</strong>와 <strong>태도·음성</strong>을 정리합니다. 점수와
              리포트로 한눈에 확인하고, 필요한 형식으로 내보낼 수 있습니다.
            </p>
            <div className="landing__cta">
              <Link to="/evaluate" className="landing__btn landing__btn--primary">
                발표 평가 시작
              </Link>
              <Link to="/projects" className="landing__btn landing__btn--ghost">
                문서 열기
              </Link>
            </div>
          </div>
        </section>

        <section className="landing__features" aria-labelledby="landing-features-heading">
          <h2 id="landing-features-heading" className="landing__section-title">
            핵심 기능
          </h2>
          <p className="landing__section-lead">음성·영상 분석부터 채점, 리포트까지 이어지는 흐름입니다.</p>
          <ul className="landing__feature-grid">
            {FEATURES.map((f) => (
              <li key={f.title} className="landing__feature">
                <span className="landing__feature-icon" aria-hidden>
                  {f.icon}
                </span>
                <h3 className="landing__feature-title">{f.title}</h3>
                <p className="landing__feature-desc">{f.desc}</p>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}
