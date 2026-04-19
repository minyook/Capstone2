import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { useFolders } from "../context/FoldersContext";
import "./MyPage.css";

export function MyPage() {
  const navigate = useNavigate();
  const { user, signOutUser } = useAuth();
  const { folders } = useFolders();
  const [loggingOut, setLoggingOut] = useState(false);

  const folderCount = folders.length;
  const totalPresentationCount = 0;
  const gradeACount = 0;

  const label = user?.displayName?.trim() || user?.email?.split("@")[0] || "게스트";
  const email = user?.email ?? "로그인이 필요합니다";

  async function logout() {
    setLoggingOut(true);
    try {
      await signOutUser();
      navigate("/login", { replace: true });
    } finally {
      setLoggingOut(false);
    }
  }

  return (
    <div className="page mypage">
      <div className="page-inner page-inner--wide">
        <div className="mypage-profile">
          <div className="mypage-avatar" aria-hidden>
            {user?.photoURL ? (
              <img src={user.photoURL} alt="" width={40} height={40} className="mypage-avatar__img" />
            ) : (
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="9" r="4" fill="#9ca3af" />
                <path d="M4 20a8 8 0 0116 0" stroke="#9ca3af" strokeWidth="2" />
              </svg>
            )}
          </div>
          <h1 className="mypage-name">{label}</h1>
          <p className="mypage-email">{email}</p>
        </div>

        <h2 className="mypage-section-title">계정 활동 내역</h2>
        <div className="mypage-card">
          <div className="mypage-stat">
            <span>총 발표 횟수</span>
            <strong>{totalPresentationCount}회</strong>
          </div>
          <div className="mypage-stat">
            <span>등록된 폴더</span>
            <strong>{folderCount}개</strong>
          </div>
          <div className="mypage-stat">
            <span>우수 발표(A등급)</span>
            <strong>{gradeACount}회</strong>
          </div>
        </div>

        <div className="mypage-notes-head">
          <span className="mypage-notes-head__t">전체 노트</span>
          <span className="mypage-notes-head__c">Total {folderCount}</span>
        </div>
        <Link to="/notes" className="mypage-notes-card">
          <span className="mypage-notes-card__icon">📁</span>
          <div className="mypage-notes-card__text">
            <strong>전체 노트 보기</strong>
            <span>저장된 모든 발표 기록 확인</span>
          </div>
          <span className="mypage-notes-card__chev" aria-hidden>
            ›
          </span>
        </Link>

        {user ? (
          <button
            type="button"
            className="mypage-logout"
            onClick={logout}
            disabled={loggingOut}
          >
            {loggingOut ? "로그아웃 중…" : "로그아웃"}
          </button>
        ) : (
          <Link to="/login" className="mypage-logout mypage-logout--link">
            로그인하기
          </Link>
        )}
      </div>
    </div>
  );
}
