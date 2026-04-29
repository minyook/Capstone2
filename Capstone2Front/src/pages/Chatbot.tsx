import { useEffect, useId, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import "./Chatbot.css";

type ChatRole = "user" | "bot";

type ChatMessage = {
  id: string;
  role: ChatRole;
  text: string;
  /** 사용자가 PPT를 올린 턴 */
  isPptUpload?: boolean;
};

function newId(): string {
  return typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `m-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

/** 데모: PPT는 일단 기존 가짜 피드백 유지 (추후 서버 연동) */
function pptFeedbackDemo(fileName: string): string {
  return `「${fileName}」을(를) 받았습니다. 파일 내용을 바로 읽지는 않지만, 일반적인 발표 구성 관점에서 피드백을 드립니다.\n\n• 슬라이드 구조: 도입(문제·목표) → 본론(근거·절차) → 결론(요약·다음 단계)처럼 한 슬라이드에 메시지가 하나씩 드러나는지 확인해 보세요.`;
}

export function Chatbot() {
  const uploadId = useId();
  const bodyRef = useRef<HTMLDivElement>(null);
  const [text, setText] = useState("");
  const [isLoading, setIsLoading] = useState(false); // 🌟 로딩 상태 추가

  const [messages, setMessages] = useState<ChatMessage[]>(() => [
    {
      id: "welcome-msg",
      role: "bot",
      text: "안녕하세요! 발표 준비를 돕는 AI 코치입니다. 발표 구성, 대본 작성, 긴장감 해소 등 궁금한 점을 편하게 물어보세요!",
    },
  ]);

  // 채팅이 추가될 때마다 스크롤 맨 아래로 이동
  useEffect(() => {
    const el = bodyRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, isLoading]);

  // PPT 파일 처리 (기존 유지)
  function handlePptFiles(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    const lower = file.name.toLowerCase();
    const ok = lower.endsWith(".ppt") || lower.endsWith(".pptx");
    if (!ok) {
      setMessages((prev) => [
        ...prev,
        { id: newId(), role: "user", text: `${file.name} 업로드 시도` },
        { id: newId(), role: "bot", text: "`.ppt` 또는 `.pptx` 파일만 지원합니다." },
      ]);
      return;
    }
    setMessages((prev) => [
      ...prev,
      { id: newId(), role: "user", text: `PPT 업로드: ${file.name}`, isPptUpload: true },
      { id: newId(), role: "bot", text: pptFeedbackDemo(file.name) },
    ]);
  }

  // 🌟 [핵심] 실제 백엔드 API와 통신하는 함수
  async function handleSend() {
    const t = text.trim();
    if (!t || isLoading) return; // 빈 칸이거나 로딩 중이면 무시
    
    setText(""); // 입력창 초기화

    // 1. 사용자 메시지 화면에 먼저 즉시 표시
    setMessages((prev) => [
      ...prev,
      { id: newId(), role: "user", text: t }
    ]);
    setIsLoading(true);

    try {
      // 2. 현재까지의 채팅 기록을 백엔드 LLaMA 양식으로 변환
      // (프론트엔드의 'bot'을 백엔드의 'assistant'로 이름만 바꿔줍니다)
      const historyForBackend = messages
        .filter((m) => m.id !== "welcome-msg") // 첫 환영 인사는 제외
        .map((m) => ({
          role: m.role === "bot" ? "assistant" : "user",
          content: m.text,
        }));

      // 3. FastAPI 백엔드로 POST 요청 보내기
      const res = await fetch("http://127.0.0.1:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: t,
          chat_history: historyForBackend,
        }),
      });

      if (!res.ok) throw new Error("서버 응답 오류");

      const data = await res.json();
      
      // 4. 백엔드가 준 최신 대화 기록에서 AI의 마지막 답변만 가져오기
      const updatedHistory = data.chat_history;
      const lastAiMessage = updatedHistory[updatedHistory.length - 1];

      // 5. 화면에 AI 답변 추가
      setMessages((prev) => [
        ...prev,
        { id: newId(), role: "bot", text: lastAiMessage.content }
      ]);

    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        { id: newId(), role: "bot", text: "서버와 연결할 수 없습니다. (FastAPI 서버가 켜져 있는지 확인해주세요!)" }
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="chatbot-page">
      <header className="chatbot-page__head">
        <h1 className="chatbot-page__title">챗봇</h1>
        <p className="chatbot-page__sub">
          PPT를 올리면 피드백을 드리고, 질문하시면 발표·채점 관점에서 답해 드립니다.
        </p>
        <div className="chatbot-page__upload">
          <input
            id={uploadId}
            type="file"
            className="chatbot-page__file"
            accept=".ppt,.pptx,application/vnd.ms-powerpoint,application/vnd.openxmlformats-officedocument.presentationml.presentation"
            onChange={(e) => {
              handlePptFiles(e.target.files);
              e.target.value = "";
            }}
          />
          <label htmlFor={uploadId} className="chatbot-page__upload-btn">
            PPT 올리기
          </label>
          <span className="chatbot-page__upload-hint">.ppt · .pptx</span>
        </div>
      </header>

      <div className="chatbot-page__body" ref={bodyRef}>
        {messages.map((m) => (
          <div key={m.id} className={"chatbot-msg" + (m.role === "user" ? " chatbot-msg--user" : "")}>
            {m.role === "bot" && (
              <div className="chatbot-msg__avatar" aria-hidden>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" />
                  <path d="M12 7v5l3 2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                </svg>
              </div>
            )}
            <div
              className={
                "chatbot-bubble" +
                (m.role === "user" ? " chatbot-bubble--user" : "") +
                (m.isPptUpload ? " chatbot-bubble--ppt" : "")
              }
            >
              {m.role === "user" && m.isPptUpload ? (
                <span className="chatbot-bubble__tag">PPT</span>
              ) : null}
              {/* 줄바꿈 처리를 위해 백틱/엔터를 <br/>로 치환하는 스타일 추가 */}
              <div className="chatbot-bubble__text markdown-container">
                <ReactMarkdown>{m.text}</ReactMarkdown>
              </div>
            </div>
          </div>
        ))}
        
        {/* 🌟 로딩 중일 때 표시되는 AI 애니메이션 (가짜 버블) */}
        {isLoading && (
          <div className="chatbot-msg">
            <div className="chatbot-msg__avatar" aria-hidden>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" />
                <path d="M12 7v5l3 2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
            </div>
            <div className="chatbot-bubble">
              <div className="chatbot-bubble__text" style={{ fontStyle: "italic", color: "#888" }}>
                답변을 생성하고 있습니다...
              </div>
            </div>
          </div>
        )}
      </div>

      <footer className="chatbot-inputbar">
        <input
          className="chatbot-input"
          placeholder="발표 관련 궁금한 점을 물어보세요!"
          value={text}
          disabled={isLoading} // 🌟 로딩 중 입력 방지
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
        />
        <button 
          type="button" 
          className="chatbot-send" 
          aria-label="보내기" 
          onClick={handleSend}
          disabled={isLoading} // 🌟 로딩 중 클릭 방지
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
            <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" fill="currentColor" />
          </svg>
        </button>
      </footer>
    </div>
  );
}