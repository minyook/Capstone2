import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { sendPasswordResetEmail } from "firebase/auth";
import { auth } from "../firebase/config";
import { useAuth } from "../context/AuthContext";
import { mapAuthError } from "../firebase/authErrors";
import { AuthLayout } from "../components/AuthLayout";
import "./auth.css";

export function ForgotPassword() {
  const navigate = useNavigate();
  const { firebaseConfigured } = useAuth();
  const [email, setEmail] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState(false);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSent(false);
    if (!auth || !firebaseConfigured) return;
    setBusy(true);
    try {
      await sendPasswordResetEmail(auth, email.trim());
      setSent(true);
    } catch (err: unknown) {
      const code = err && typeof err === "object" && "code" in err ? String((err as { code: string }).code) : "";
      setError(mapAuthError(code));
    } finally {
      setBusy(false);
    }
  }

  return (
    <AuthLayout>
      <div className="auth-form">
        <div className="auth-form__head">
          <button type="button" className="auth-back-link" onClick={() => navigate("/login")}>
            ← 로그인
          </button>
          <h1 className="auth-form__title">비밀번호 찾기</h1>
        </div>
        <p className="auth-form__desc">가입하신 이메일로 재설정 링크를 보내드립니다.</p>

        {!firebaseConfigured && (
          <div className="auth-banner auth-banner--info" role="status">
            지금은 비밀번호 재설정 메일을 보낼 수 없습니다. 잠시 후 다시 시도해 주세요.
          </div>
        )}

        {error && (
          <p className="auth-error" role="alert">
            {error}
          </p>
        )}
        {sent && (
          <p className="auth-success" role="status">
            메일함으로 재설정 링크를 보냈습니다. 스팸함도 확인해 주세요.
          </p>
        )}

        <form onSubmit={submit} noValidate>
          <div className="auth-field">
            <label className="auth-label" htmlFor="forgot-email">
              이메일
            </label>
            <input
              id="forgot-email"
              className="auth-input"
              type="email"
              placeholder="name@example.com"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              disabled={!firebaseConfigured || busy}
              required
            />
          </div>
          <p className="auth-hint">* 스팸 메일함도 꼭 확인해주세요.</p>

          <button type="submit" className="auth-btn auth-btn--mt" disabled={!firebaseConfigured || busy}>
            {busy ? "전송 중…" : "재설정 링크 전송"}
          </button>
        </form>

        <p className="auth-footer">
          <Link to="/login">로그인으로 돌아가기</Link>
        </p>
      </div>
    </AuthLayout>
  );
}
