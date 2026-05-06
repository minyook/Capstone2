import { useEffect, useId, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import "./Chatbot.css";

type ChatRole = "user" | "bot";

type ChatMessage = {
  id: string;
  role: ChatRole;
  text: string;
  attachedFileName?: string;
};

function newId(): string {
  return typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `m-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

const ATTACH_ONLY_FALLBACK = "첨부한 파일에 대한 피드백을 요청합니다.";

function userMessageContentForApi(m: ChatMessage): string {
  if (m.role !== "user") return m.text;
  const body = m.text.trim() || ATTACH_ONLY_FALLBACK;
  if (m.attachedFileName) {
    return `[첨부 파일: ${m.attachedFileName}]\n\n${body}`;
  }
  return m.text;
}

function buildOutgoingApiMessage(textTrimmed: string, file: File | null): string {
  const body = textTrimmed || ATTACH_ONLY_FALLBACK;
  if (file) {
    return `[첨부 파일: ${file.name}]\n\n${body}`;
  }
  return textTrimmed;
}

const IMAGE_EXT = /\.(png|jpe?g|webp|gif|bmp)$/i;
const PPT_EXT = /\.(pptx|ppt)$/i;

function attachmentBadgeLabel(fileName: string): string {
  const n = fileName.toLowerCase();
  if (IMAGE_EXT.test(n)) return "이미지";
  if (PPT_EXT.test(n)) return "PPT";
  return "파일";
}

function isAllowedAttachment(file: File): boolean {
  const n = file.name.toLowerCase();
  if (IMAGE_EXT.test(n) || PPT_EXT.test(n)) return true;
  if (file.type.startsWith("image/")) return true;
  return (
    file.type === "application/vnd.ms-powerpoint" ||
    file.type ===
      "application/vnd.openxmlformats-officedocument.presentationml.presentation"
  );
}

export function Chatbot() {
  const attachInputId = useId();
  const pageRef = useRef<HTMLDivElement>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const bodyDragDepthRef = useRef(0);
  const footerDragDepthRef = useRef(0);

  const [text, setText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [bodyDropActive, setBodyDropActive] = useState(false);
  const [footerDropActive, setFooterDropActive] = useState(false);
  const [pendingFile, setPendingFile] = useState<File | null>(null);

  const [messages, setMessages] = useState<ChatMessage[]>(() => [
    {
      id: "welcome-msg",
      role: "bot",
      text: "발표 준비를 돕는 AI 코치입니다. **이미지** 또는 **PPT**를 끌어다 놓고 메시지와 함께 보내면 내용을 분석해 드립니다.",
    },
  ]);

  useEffect(() => {
    const el = bodyRef.current;
    if (!el) return;
    requestAnimationFrame(() => {
      el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    });
  }, [messages, isLoading]);

  /** 채팅 화면 안에서 스크린샷·복사 이미지 Ctrl+V / ⌘V 붙여넣기 */
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const root = pageRef.current;
      if (!root?.contains(e.target as Node)) return;
      const cd = e.clipboardData;
      if (!cd) return;

      const attachImage = (file: File) => {
        e.preventDefault();
        if (!isAllowedAttachment(file)) {
          setMessages((prev) => [
            ...prev,
            { id: newId(), role: "user", text: `${file.name} 붙여넣기` },
            {
              id: newId(),
              role: "bot",
              text:
                "이미지(png, jpg, webp 등) 또는 발표 파일(ppt, pptx)만 첨부할 수 있습니다.",
            },
          ]);
          return;
        }
        setPendingFile(file);
      };

      for (let i = 0; i < cd.items.length; i++) {
        const item = cd.items[i];
        if (item.kind !== "file" || !item.type.startsWith("image/")) continue;
        const blob = item.getAsFile();
        if (!blob) continue;
        let ext = blob.type.split("/")[1] || "png";
        ext = ext.replace("+xml", "").replace("jpeg", "jpg");
        if (ext.includes(";")) ext = ext.split(";")[0];
        attachImage(new File([blob], `붙여넣기-${Date.now()}.${ext}`, { type: blob.type }));
        return;
      }

      const files = cd.files;
      if (files?.length) {
        for (let i = 0; i < files.length; i++) {
          const f = files[i];
          if (f.type.startsWith("image/")) {
            attachImage(f);
            return;
          }
        }
      }
    };

    document.addEventListener("paste", onPaste);
    return () => document.removeEventListener("paste", onPaste);
  }, []);

  const canSend = !isLoading && (!!text.trim() || !!pendingFile);

  function warnBadFile(name: string) {
    setMessages((prev) => [
      ...prev,
      { id: newId(), role: "user", text: `${name} 첨부 시도` },
      {
        id: newId(),
        role: "bot",
        text:
          "이미지(png, jpg, webp 등) 또는 발표 파일(ppt, pptx)만 첨부할 수 있습니다.",
      },
    ]);
  }

  function stageFile(file: File | undefined | null) {
    if (!file) return;
    if (!isAllowedAttachment(file)) {
      warnBadFile(file.name);
      return;
    }
    setPendingFile(file);
  }

  function pickAttachable(dt: DataTransfer | null): File | null {
    const files = dt?.files;
    if (!files?.length) return null;
    return Array.from(files).find((f) => isAllowedAttachment(f)) ?? null;
  }

  function applyDrop(dt: DataTransfer | null) {
    const file = pickAttachable(dt);
    if (file) {
      stageFile(file);
      return;
    }
    if (dt?.files?.length) warnBadFile(dt.files[0].name);
  }

  function handleBodyDragEnter(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (!e.dataTransfer.types.includes("Files")) return;
    bodyDragDepthRef.current += 1;
    setBodyDropActive(true);
  }

  function handleBodyDragLeave(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    bodyDragDepthRef.current -= 1;
    if (bodyDragDepthRef.current <= 0) {
      bodyDragDepthRef.current = 0;
      setBodyDropActive(false);
    }
  }

  function handleBodyDragOver(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.types.includes("Files")) e.dataTransfer.dropEffect = "copy";
  }

  function handleBodyDrop(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    bodyDragDepthRef.current = 0;
    setBodyDropActive(false);
    applyDrop(e.dataTransfer);
  }

  function handleFooterDragEnter(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (!e.dataTransfer.types.includes("Files")) return;
    footerDragDepthRef.current += 1;
    setFooterDropActive(true);
  }

  function handleFooterDragLeave(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    footerDragDepthRef.current -= 1;
    if (footerDragDepthRef.current <= 0) {
      footerDragDepthRef.current = 0;
      setFooterDropActive(false);
    }
  }

  function handleFooterDragOver(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.types.includes("Files")) e.dataTransfer.dropEffect = "copy";
  }

  function handleFooterDrop(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    footerDragDepthRef.current = 0;
    setFooterDropActive(false);
    applyDrop(e.dataTransfer);
  }

  async function handleSend() {
    const raw = text.trim();
    const file = pendingFile;
    if (!raw && !file) return;
    if (isLoading) return;

    const historyForBackend = messages
      .filter((m) => m.id !== "welcome-msg")
      .map((m) => ({
        role: m.role === "bot" ? "assistant" : "user",
        content: m.role === "user" ? userMessageContentForApi(m) : m.text,
      }));

    const apiMessage = buildOutgoingApiMessage(raw, file);

    setText("");
    setPendingFile(null);

    setMessages((prev) => [
      ...prev,
      {
        id: newId(),
        role: "user",
        text: raw,
        attachedFileName: file?.name,
      },
    ]);
    setIsLoading(true);

    try {
      // 3. 파일이 있으면 일반 API, 없으면 스트리밍 API 사용
      if (file) {
        const res = await fetch("http://127.0.0.1:8000/api/chat/with-file", {
          method: "POST",
          body: (() => {
            const fd = new FormData();
            fd.append("message", apiMessage);
            fd.append("chat_history", JSON.stringify(historyForBackend));
            fd.append("file", file);
            return fd;
          })(),
        });

        if (!res.ok) throw new Error("서버 응답 오류");

        const data = await res.json();
        const updatedHistory = data.chat_history;
        const lastAiMessage = updatedHistory[updatedHistory.length - 1];

        setMessages((prev) => [
          ...prev,
          { id: newId(), role: "bot", text: lastAiMessage.content },
        ]);
      } else {
        // 스트리밍 API 호출
        const res = await fetch("http://127.0.0.1:8000/api/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: apiMessage,
            chat_history: historyForBackend,
          }),
        });

        if (!res.ok) throw new Error("서버 응답 오류");

        const reader = res.body?.getReader();
        if (!reader) throw new Error("스트림을 읽을 수 없습니다.");

        const decoder = new TextDecoder();
        let botMessageId = newId();

        setMessages((prev) => [
          ...prev,
          { id: botMessageId, role: "bot", text: "" },
        ]);
        setIsLoading(false);

        let accumulatedText = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          accumulatedText += chunk;

          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === botMessageId ? { ...msg, text: accumulatedText } : msg
            )
          );
        }
      }
    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        {
          id: newId(),
          role: "bot",
          text: "서버와 연결할 수 없습니다. (FastAPI 서버가 켜져 있는지 확인해주세요!)",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="chatbot-page" ref={pageRef}>
      <header className="chatbot-page__head">
        <h1 className="chatbot-page__title">챗봇</h1>
        <p className="chatbot-page__sub">
          이미지·PPT를 끌어다 놓거나, 채팅 화면에서 포커스를 둔 뒤{" "}
          <strong>Ctrl+V</strong>(Mac: <strong>⌘V</strong>)로 캡처·복사 이미지를 붙여 넣을 수 있어요.
        </p>
      </header>

      <div
        className={
          "chatbot-page__body" + (bodyDropActive ? " chatbot-page__body--drag" : "")
        }
        ref={bodyRef}
        onDragEnter={handleBodyDragEnter}
        onDragLeave={handleBodyDragLeave}
        onDragOver={handleBodyDragOver}
        onDrop={handleBodyDrop}
      >
        <div className="chatbot-page__thread">
          {messages.map((m) => (
            <div
              key={m.id}
              className={
                "chatbot-msg" + (m.role === "user" ? " chatbot-msg--user" : "")
              }
            >
              {m.role === "bot" && (
                <div className="chatbot-msg__avatar" aria-hidden>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" />
                    <path
                      d="M12 7v5l3 2"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                    />
                  </svg>
                </div>
              )}
              <div
                className={
                  "chatbot-bubble" +
                  (m.role === "user" ? " chatbot-bubble--user" : "") +
                  (m.attachedFileName ? " chatbot-bubble--attachment" : "")
                }
              >
                {m.role === "user" && m.attachedFileName ? (
                  <div className="chatbot-bubble__file-row">
                    <span className="chatbot-bubble__tag">
                      {attachmentBadgeLabel(m.attachedFileName)}
                    </span>
                    <span className="chatbot-bubble__filename">{m.attachedFileName}</span>
                  </div>
                ) : null}
                {m.text.trim() ? (
                  <div className="chatbot-bubble__text markdown-container">
                    <ReactMarkdown>{m.text}</ReactMarkdown>
                  </div>
                ) : null}
                {m.role === "user" && m.attachedFileName && !m.text.trim() ? (
                  <p className="chatbot-bubble__text-muted">(메시지 없이 첨부만 전송)</p>
                ) : null}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="chatbot-msg">
              <div className="chatbot-msg__avatar" aria-hidden>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" />
                  <path
                    d="M12 7v5l3 2"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                  />
                </svg>
              </div>
              <div className="chatbot-bubble">
                <div className="chatbot-bubble__text chatbot-bubble__text--loading">
                  답변을 생성하고 있습니다...
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <div
        className={
          "chatbot-composer" + (footerDropActive ? " chatbot-composer--drag" : "")
        }
        onDragEnter={handleFooterDragEnter}
        onDragLeave={handleFooterDragLeave}
        onDragOver={handleFooterDragOver}
        onDrop={handleFooterDrop}
      >
        <input
          ref={fileInputRef}
          id={attachInputId}
          type="file"
          className="chatbot-composer__file-input"
          accept=".ppt,.pptx,.png,.jpg,.jpeg,.webp,.gif,.bmp,image/*,application/vnd.ms-powerpoint,application/vnd.openxmlformats-officedocument.presentationml.presentation"
          onChange={(e) => {
            stageFile(e.target.files?.[0] ?? null);
            e.target.value = "";
          }}
        />

        {pendingFile ? (
          <div className="chatbot-composer__attachments">
            <div className="chatbot-attach-chip">
              <span className="chatbot-attach-chip__badge">
                {attachmentBadgeLabel(pendingFile.name)}
              </span>
              <span className="chatbot-attach-chip__name" title={pendingFile.name}>
                {pendingFile.name}
              </span>
              <button
                type="button"
                className="chatbot-attach-chip__remove"
                aria-label="첨부 제거"
                onClick={() => setPendingFile(null)}
              >
                ×
              </button>
            </div>
          </div>
        ) : null}

        <footer className="chatbot-inputbar">
          <button
            type="button"
            className="chatbot-attach-btn"
            aria-label="파일 첨부"
            disabled={isLoading}
            onClick={() => fileInputRef.current?.click()}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden>
              <path
                d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
          <input
            className="chatbot-input"
            placeholder={
              pendingFile
                ? "첨부와 함께 보낼 메시지를 입력하세요 (선택)"
                : "발표 관련 궁금한 점을 물어보세요!"
            }
            value={text}
            disabled={isLoading}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                if (canSend) handleSend();
              }
            }}
          />
          <button
            type="button"
            className="chatbot-send"
            aria-label="보내기"
            onClick={handleSend}
            disabled={!canSend}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
              <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" fill="currentColor" />
            </svg>
          </button>
        </footer>
      </div>
    </div>
  );
}
