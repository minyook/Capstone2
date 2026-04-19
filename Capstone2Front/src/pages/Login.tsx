import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { GoogleAuthProvider, reload, signInWithEmailAndPassword, signInWithPopup } from "firebase/auth";
import { auth } from "../firebase/config";
import { useAuth } from "../context/AuthContext";
import { mapAuthError } from "../firebase/authErrors";
import { AuthLayout } from "../components/AuthLayout";
import { IconGoogle } from "../components/SocialAuthIcons";
import { IconEye, IconEyeOff } from "../components/Icons";
import "./auth.css";

export function Login() {
  const navigate = useNavigate();
  const { firebaseConfigured } = useAuth();
  const [showPw, setShowPw] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const formEnabled = firebaseConfigured && Boolean(auth);
  const canSubmit = formEnabled && !busy;
  const socialEnabled = formEnabled;

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (!auth) {
      setError("로그인을 처리할 수 없습니다. 잠시 후 다시 시도해 주세요.");
      return;
    }
    setBusy(true);
    try {
      const cred = await signInWithEmailAndPassword(auth, email.trim(), password);
      await reload(cred.user);
      navigate("/", { replace: true });
    } catch (err: unknown) {
      const code = err && typeof err === "object" && "code" in err ? String((err as { code: string }).code) : "";
      setError(mapAuthError(code));
    } finally {
      setBusy(false);
    }
  }

  async function signInWithGoogle() {
    if (!auth) return;
    setError(null);
    setBusy(true);
    try {
      const cred = await signInWithPopup(auth, new GoogleAuthProvider());
      if (cred.user) await reload(cred.user);
      navigate("/", { replace: true });
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
        <h1 className="auth-form__title">로그인</h1>
        <p className="auth-form__desc">가입하신 이메일과 비밀번호로 로그인하세요.</p>

        {!firebaseConfigured && (
          <div className="auth-banner auth-banner--info" role="status">
            지금은 로그인을 이용할 수 없습니다. 잠시 후 다시 시도하거나, 문제가 계속되면 관리자에게 문의해 주세요.
          </div>
        )}

        {error && (
          <p className="auth-error" role="alert">
            {error}
          </p>
        )}

        <form onSubmit={submit} noValidate>
          <div className="auth-field">
            <label className="auth-label" htmlFor="login-email">
              이메일
            </label>
            <input
              id="login-email"
              className="auth-input"
              type="email"
              placeholder="name@example.com"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              disabled={!formEnabled || busy}
              required
            />
          </div>
          <div className="auth-field">
            <label className="auth-label" htmlFor="login-password">
              비밀번호
            </label>
            <div className="auth-input-wrap">
              <input
                id="login-password"
                className="auth-input"
                type={showPw ? "text" : "password"}
                placeholder="비밀번호 입력"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={!formEnabled || busy}
                required
              />
              <button
                type="button"
                className="auth-input-icon"
                onClick={() => setShowPw(!showPw)}
                aria-label={showPw ? "비밀번호 숨기기" : "비밀번호 보기"}
                disabled={!formEnabled || busy}
              >
                {showPw ? <IconEyeOff /> : <IconEye />}
              </button>
            </div>
          </div>

          <div className="auth-row-end">
            <Link className="auth-link" to="/forgot-password">
              비밀번호를 잊으셨나요?
            </Link>
          </div>

          <button type="submit" className="auth-btn" disabled={!canSubmit}>
            {busy ? "처리 중…" : "로그인"}
          </button>
        </form>

        <div className="auth-divider">또는</div>
        <div className="auth-social">
          <button
            type="button"
            className="auth-social-btn auth-social-btn--google"
            disabled={!socialEnabled || busy}
            onClick={signInWithGoogle}
          >
            <IconGoogle />
            Google로 계속하기
          </button>
        </div>
        <p className="auth-footer">
          계정이 없으신가요? <Link to="/register">회원가입</Link>
        </p>
      </div>
    </AuthLayout>
  );
}
