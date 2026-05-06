import { useEffect, useId, useRef, useState } from "react";
import { Link } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import {
  addDoc,
  collection,
  deleteDoc,
  doc,
  getDocs,
  onSnapshot,
  orderBy,
  query,
  serverTimestamp,
  setDoc,
  writeBatch,
  type Timestamp,
} from "firebase/firestore";
import { db } from "../firebase/config";
import { useAuth } from "../context/AuthContext";
import "./Chatbot.css";

type ChatRole = "user" | "bot";

type ChatMessage = {
  id: string;
  role: ChatRole;
  text: string;
  attachedFileName?: string;
};

type ChatThreadRow = {
  id: string;
  title: string;
  preview: string;
  updatedAt?: Timestamp;
};

const WELCOME_ID = "welcome-msg";
const WELCOME_MSG: ChatMessage = {
  id: WELCOME_ID,
  role: "bot",
  text: "발표 준비를 돕는 AI 코치입니다. **파일을 끌어다 놓거나 첨부**하고 메시지와 함께 보내면 도와 드립니다.",
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

function formatThreadTime(ts?: Timestamp): string {
  if (!ts?.toDate) return "";
  const d = ts.toDate();
  const now = new Date();
  const diff = now.getTime() - d.getTime();
  if (diff < 86_400_000 && d.getDate() === now.getDate()) {
    return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  }
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/** 업로드 상한 (브라우저·서버 부하 완화). 더 크게 필요하면 조정하세요. */
const MAX_ATTACHMENT_BYTES = 100 * 1024 * 1024;

const IMAGE_EXT = /\.(png|jpe?g|webp|gif|bmp|svg|tif|tiff|avif|heic)$/i;
const PPT_EXT = /\.(pptx|ppt|key)$/i;
const PDF_EXT = /\.pdf$/i;
const DOC_EXT = /\.(docx?|xlsx?|csv|txt|md|rtf|od[tsp]|hwp|hwpx)$/i;
const VIDEO_EXT = /\.(mp4|webm|mov|mkv|avi|m4v|wmv|mpeg|mpg)$/i;
const AUDIO_EXT = /\.(mp3|wav|m4a|aac|ogg|flac|wma)$/i;
const ARCH_EXT = /\.(zip|rar|7z|tar|gz|bz2)$/i;

function attachmentBadgeLabel(fileName: string): string {
  const n = fileName.toLowerCase();
  if (IMAGE_EXT.test(n)) return "이미지";
  if (PPT_EXT.test(n)) return "PPT";
  if (PDF_EXT.test(n)) return "PDF";
  if (DOC_EXT.test(n)) return "문서";
  if (VIDEO_EXT.test(n)) return "동영상";
  if (AUDIO_EXT.test(n)) return "음성";
  if (ARCH_EXT.test(n)) return "압축";
  return "파일";
}

function attachmentRejectReason(file: File): "oversize" | "invalid" | null {
  if (!file.name?.trim()) return "invalid";
  if (file.size > MAX_ATTACHMENT_BYTES) return "oversize";
  return null;
}

function isAllowedAttachment(file: File): boolean {
  return attachmentRejectReason(file) === null;
}

function maxAttachmentLabelMb(): number {
  return Math.round(MAX_ATTACHMENT_BYTES / (1024 * 1024));
}

export function Chatbot() {
  const { user } = useAuth();
  const uid = user?.uid ?? null;

  const attachInputId = useId();
  const pageRef = useRef<HTMLDivElement>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const bodyDragDepthRef = useRef(0);
  const footerDragDepthRef = useRef(0);
  const creatingBootstrapRef = useRef(false);
  const threadsRef = useRef<ChatThreadRow[]>([]);
  const activeThreadIdRef = useRef<string | null>(null);

  const [text, setText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isDeletingChat, setIsDeletingChat] = useState(false);
  const [bodyDropActive, setBodyDropActive] = useState(false);
  const [footerDropActive, setFooterDropActive] = useState(false);
  const [pendingFile, setPendingFile] = useState<File | null>(null);

  const [threads, setThreads] = useState<ChatThreadRow[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>(() => [WELCOME_MSG]);
  const [firestoreError, setFirestoreError] = useState<string | null>(null);
  const [persistHint, setPersistHint] = useState<string | null>(null);

  threadsRef.current = threads;
  activeThreadIdRef.current = activeThreadId;

  useEffect(() => {
    if (!uid) setFirestoreError(null);
  }, [uid]);

  /** 로그인 사용자: 스레드 목록 구독 + 비어 있으면 첫 스레드 생성 */
  useEffect(() => {
    if (!db || !uid) {
      setThreads([]);
      setActiveThreadId(null);
      return;
    }

    const colRef = collection(db, "users", uid, "chatThreads");
    const qRef = query(colRef, orderBy("updatedAt", "desc"));

    const unsub = onSnapshot(
      qRef,
      (snap) => {
        setFirestoreError(null);
      const items: ChatThreadRow[] = snap.docs.map((d) => {
        const data = d.data();
        return {
          id: d.id,
          title: typeof data.title === "string" ? data.title : "새 대화",
          preview: typeof data.preview === "string" ? data.preview : "",
          updatedAt: data.updatedAt as Timestamp | undefined,
        };
      });
      setThreads(items);

      if (snap.empty) {
        if (!creatingBootstrapRef.current) {
          creatingBootstrapRef.current = true;
          const newRef = doc(colRef);
          void setDoc(newRef, {
            title: "새 대화",
            preview: "",
            createdAt: serverTimestamp(),
            updatedAt: serverTimestamp(),
          }).finally(() => {
            creatingBootstrapRef.current = false;
          });
        }
        return;
      }

      setActiveThreadId((prev) => {
        if (prev && items.some((t) => t.id === prev)) return prev;
        return items[0]?.id ?? null;
      });
      },
      (err) => {
        console.error("[chatThreads]", err);
        const code = "code" in err ? String((err as { code: string }).code) : "";
        setFirestoreError(
          code === "permission-denied"
            ? "대화 목록을 불러올 수 없습니다. Firestore 규칙에서 본인 users/chatThreads 경로 쓰기가 허용되는지 확인해 주세요."
            : `대화 목록 오류: ${err instanceof Error ? err.message : String(err)}`
        );
      }
    );

    return () => unsub();
  }, [uid]);

  /** 활성 스레드 메시지 구독 */
  useEffect(() => {
    if (!uid || !db) {
      setMessages([WELCOME_MSG]);
      return;
    }
    if (!activeThreadId) {
      setMessages([WELCOME_MSG]);
      return;
    }

    const qRef = query(
      collection(db, "users", uid, "chatThreads", activeThreadId, "messages"),
      orderBy("createdAt", "asc")
    );

    const unsub = onSnapshot(
      qRef,
      (snap) => {
        setFirestoreError(null);
      const list: ChatMessage[] = [];
      snap.forEach((docSnap) => {
        const d = docSnap.data();
        list.push({
          id: docSnap.id,
          role: d.role === "assistant" ? "bot" : "user",
          text: typeof d.text === "string" ? d.text : "",
          attachedFileName:
            typeof d.attachedFileName === "string" ? d.attachedFileName : undefined,
        });
      });
      setMessages(list.length === 0 ? [WELCOME_MSG] : list);
      },
      (err) => {
        console.error("[messages]", err);
        const code = "code" in err ? String((err as { code: string }).code) : "";
        setFirestoreError(
          code === "permission-denied"
            ? "메시지를 불러오거나 저장할 수 없습니다. Firestore 규칙을 게시했는지 확인해 주세요."
            : `메시지 오류: ${err instanceof Error ? err.message : String(err)}`
        );
      }
    );

    return () => unsub();
  }, [uid, activeThreadId]);

  useEffect(() => {
    const el = bodyRef.current;
    if (!el) return;
    requestAnimationFrame(() => {
      el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    });
  }, [messages, isLoading]);

  /** 채팅 화면 안에서 파일 붙여넣기 (Ctrl+V / ⌘V) */
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const root = pageRef.current;
      if (!root?.contains(e.target as Node)) return;
      const cd = e.clipboardData;
      if (!cd) return;

      const attachFromClipboard = (file: File) => {
        e.preventDefault();
        const why = attachmentRejectReason(file);
        if (why === "oversize") {
          setMessages((prev) => [
            ...prev,
            { id: newId(), role: "user", text: `${file.name} 붙여넣기` },
            {
              id: newId(),
              role: "bot",
              text: `첨부 용량은 **${maxAttachmentLabelMb()}MB** 이하만 가능합니다.`,
            },
          ]);
          return;
        }
        if (why === "invalid") {
          setMessages((prev) => [
            ...prev,
            { id: newId(), role: "user", text: "파일 붙여넣기" },
            {
              id: newId(),
              role: "bot",
              text: "파일 이름을 확인할 수 없어 첨부하지 못했습니다.",
            },
          ]);
          return;
        }
        setPendingFile(file);
      };

      for (let i = 0; i < cd.items.length; i++) {
        const item = cd.items[i];
        if (item.kind !== "file") continue;
        const blob = item.getAsFile();
        if (!blob) continue;
        const named =
          blob.name?.trim().length > 0
            ? blob
            : (() => {
                let ext = blob.type.split("/")[1] || "bin";
                ext = ext.replace("+xml", "").replace("jpeg", "jpg");
                if (ext.includes(";")) ext = ext.split(";")[0];
                return new File([blob], `붙여넣기-${Date.now()}.${ext}`, {
                  type: blob.type || "application/octet-stream",
                });
              })();
        attachFromClipboard(named);
        return;
      }

      const files = cd.files;
      if (files?.length) {
        for (let i = 0; i < files.length; i++) {
          const f = files[i];
          if (isAllowedAttachment(f)) {
            attachFromClipboard(f);
            return;
          }
        }
        const first = files[0];
        attachFromClipboard(first);
      }
    };

    document.addEventListener("paste", onPaste);
    return () => document.removeEventListener("paste", onPaste);
  }, []);

  const canSend = !isLoading && (!!text.trim() || !!pendingFile);

  /** 전송 직전 스레드 ID 확보 (목록 로딩 지연 시 activeThreadId가 비어 저장이 스킵되던 문제 방지) */
  async function resolveThreadIdForSave(): Promise<string | null> {
    if (!uid || !db) return null;
    let tid = activeThreadIdRef.current;
    if (tid) return tid;
    const list = threadsRef.current;
    if (list[0]?.id) {
      tid = list[0].id;
      setActiveThreadId(tid);
      activeThreadIdRef.current = tid;
      return tid;
    }
    // onSnapshot 부트스트랩 스레드 생성 직후 반영될 때까지 짧게 대기 (중복 스레드 생성 완화)
    for (let i = 0; i < 20; i++) {
      tid = activeThreadIdRef.current;
      if (tid) return tid;
      const first = threadsRef.current[0]?.id;
      if (first) {
        setActiveThreadId(first);
        activeThreadIdRef.current = first;
        return first;
      }
      await new Promise((r) => setTimeout(r, 80));
    }
    const colRef = collection(db, "users", uid, "chatThreads");
    const newRef = doc(colRef);
    await setDoc(newRef, {
      title: "새 대화",
      preview: "",
      createdAt: serverTimestamp(),
      updatedAt: serverTimestamp(),
    });
    tid = newRef.id;
    setActiveThreadId(tid);
    activeThreadIdRef.current = tid;
    return tid;
  }

  async function persistMessage(
    threadId: string,
    role: "user" | "assistant",
    textContent: string,
    attachedFileName?: string | null,
    options?: { threadTitleFrom?: string }
  ) {
    if (!db || !uid) return;
    const messagesCol = collection(db, "users", uid, "chatThreads", threadId, "messages");
    await addDoc(messagesCol, {
      role,
      text: textContent,
      attachedFileName: attachedFileName ?? null,
      createdAt: serverTimestamp(),
    });

    const threadRef = doc(db, "users", uid, "chatThreads", threadId);
    const preview =
      textContent.trim().slice(0, 200) || (attachedFileName ? `(첨부: ${attachedFileName})` : " ");
    const patch: Record<string, unknown> = {
      updatedAt: serverTimestamp(),
      preview,
    };
    if (options?.threadTitleFrom?.trim()) {
      patch.title = options.threadTitleFrom.trim().slice(0, 80);
    }
    await setDoc(threadRef, patch, { merge: true });
  }

  async function handleNewChat() {
    if (!db || !uid) return;
    const colRef = collection(db, "users", uid, "chatThreads");
    const newRef = doc(colRef);
    await setDoc(newRef, {
      title: "새 대화",
      preview: "",
      createdAt: serverTimestamp(),
      updatedAt: serverTimestamp(),
    });
    setActiveThreadId(newRef.id);
  }

  async function deleteAllMessagesInThread(threadId: string): Promise<void> {
    if (!db || !uid) return;
    const messagesCol = collection(db, "users", uid, "chatThreads", threadId, "messages");
    const snap = await getDocs(messagesCol);
    if (snap.empty) return;

    let batch = writeBatch(db);
    let count = 0;
    for (const m of snap.docs) {
      batch.delete(m.ref);
      count += 1;
      if (count >= 400) {
        await batch.commit();
        batch = writeBatch(db);
        count = 0;
      }
    }
    if (count > 0) {
      await batch.commit();
    }
  }

  async function handleDeleteActiveChat() {
    if (!db || !uid || !activeThreadId || isDeletingChat) return;
    const ok = window.confirm("현재 대화 기록을 삭제할까요? 삭제 후 복구할 수 없습니다.");
    if (!ok) return;

    setPersistHint(null);
    setIsDeletingChat(true);
    try {
      await deleteAllMessagesInThread(activeThreadId);
      await deleteDoc(doc(db, "users", uid, "chatThreads", activeThreadId));
      const colRef = collection(db, "users", uid, "chatThreads");
      const newRef = doc(colRef);
      await setDoc(newRef, {
        title: "새 대화",
        preview: "",
        createdAt: serverTimestamp(),
        updatedAt: serverTimestamp(),
      });
      setActiveThreadId(newRef.id);
      setMessages([WELCOME_MSG]);
      setPersistHint("대화를 삭제하고 새 채팅방으로 이동했습니다.");
    } catch (e: unknown) {
      console.error(e);
      const code =
        e && typeof e === "object" && "code" in e ? String((e as { code: string }).code) : "";
      setPersistHint(
        code === "permission-denied"
          ? "대화 삭제 권한이 없습니다. Firestore 규칙을 확인해 주세요."
          : "대화 삭제에 실패했습니다. 네트워크 상태를 확인해 주세요."
      );
    } finally {
      setIsDeletingChat(false);
    }
  }

  function warnBadFile(name: string) {
    setMessages((prev) => [
      ...prev,
      { id: newId(), role: "user", text: `${name} 첨부 시도` },
      {
        id: newId(),
        role: "bot",
        text: "이 파일은 첨부할 수 없습니다. 이름이 비어 있는지 확인해 주세요.",
      },
    ]);
  }

  function warnOversize(name: string) {
    setMessages((prev) => [
      ...prev,
      { id: newId(), role: "user", text: `${name} 첨부 시도` },
      {
        id: newId(),
        role: "bot",
        text: `첨부 용량은 **${maxAttachmentLabelMb()}MB** 이하만 가능합니다.`,
      },
    ]);
  }

  function stageFile(file: File | undefined | null) {
    if (!file) return;
    const why = attachmentRejectReason(file);
    if (why === "oversize") {
      warnOversize(file.name || "파일");
      return;
    }
    if (why === "invalid") {
      warnBadFile(file.name || "(이름 없음)");
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
    if (dt?.files?.length) {
      const first = dt.files[0];
      if (attachmentRejectReason(first) === "oversize") warnOversize(first.name || "파일");
      else warnBadFile(first.name || "(이름 없음)");
    }
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

    setPersistHint(null);
    const resolvedThreadId = uid && db ? await resolveThreadIdForSave() : null;
    const canPersistNow = Boolean(uid && db && resolvedThreadId);

    const historyForBackend = messages
      .filter((m) => m.id !== WELCOME_ID)
      .map((m) => ({
        role: m.role === "bot" ? "assistant" : "user",
        content: m.role === "user" ? userMessageContentForApi(m) : m.text,
      }));

    const apiMessage = buildOutgoingApiMessage(raw, file);

    const threadId = resolvedThreadId;
    const isFirstPersistedTurn =
      canPersistNow &&
      threadId &&
      messages.filter((m) => m.id !== WELCOME_ID).length === 0;

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

    if (canPersistNow && threadId) {
      try {
        await persistMessage(threadId, "user", raw, file?.name ?? null, {
          threadTitleFrom: isFirstPersistedTurn ? raw || file?.name || "새 대화" : undefined,
        });
      } catch (e: unknown) {
        console.error(e);
        const code =
          e && typeof e === "object" && "code" in e ? String((e as { code: string }).code) : "";
        setPersistHint(
          code === "permission-denied"
            ? "대화 저장이 거부되었습니다. Firebase 콘솔에서 Firestore 규칙을 게시했는지 확인해 주세요."
            : "대화를 저장하지 못했습니다. 네트워크와 Firebase 설정을 확인해 주세요."
        );
      }
    }

    try {
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

        const botText = lastAiMessage.content;
        setMessages((prev) => [...prev, { id: newId(), role: "bot", text: botText }]);

        if (canPersistNow && threadId) {
          try {
            await persistMessage(threadId, "assistant", botText, null);
          } catch (e: unknown) {
            console.error(e);
            const code =
              e && typeof e === "object" && "code" in e ? String((e as { code: string }).code) : "";
            if (code === "permission-denied") {
              setPersistHint(
                "답변 저장이 거부되었습니다. Firestore 규칙에서 messages 경로 쓰기를 허용했는지 확인해 주세요."
              );
            }
          }
        }
      } else {
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
        const botMessageId = newId();

        setMessages((prev) => [...prev, { id: botMessageId, role: "bot", text: "" }]);
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

        if (canPersistNow && threadId && accumulatedText.trim()) {
          try {
            await persistMessage(threadId, "assistant", accumulatedText, null);
          } catch (e: unknown) {
            console.error(e);
            const code =
              e && typeof e === "object" && "code" in e ? String((e as { code: string }).code) : "";
            if (code === "permission-denied") {
              setPersistHint(
                "답변 저장이 거부되었습니다. Firestore 규칙에서 messages 경로 쓰기를 허용했는지 확인해 주세요."
              );
            }
          }
        }
      }
    } catch (error) {
      console.error(error);
      const errMsg =
        "서버와 연결할 수 없습니다. (FastAPI 서버가 켜져 있는지 확인해주세요!)";
      setMessages((prev) => [...prev, { id: newId(), role: "bot", text: errMsg }]);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="chatbot-page" ref={pageRef}>
      <div className="chatbot-layout">
        {uid ? (
          <aside className="chatbot-sidebar" aria-label="대화 목록">
            <div className="chatbot-sidebar__top">
              <button type="button" className="chatbot-sidebar__new" onClick={() => void handleNewChat()}>
                새 대화
              </button>
              <button
                type="button"
                className="chatbot-sidebar__delete"
                onClick={() => void handleDeleteActiveChat()}
                disabled={!activeThreadId || isDeletingChat}
              >
                {isDeletingChat ? "삭제 중..." : "현재 대화 삭제"}
              </button>
            </div>
            <ul className="chatbot-sidebar__list">
              {threads.map((t) => (
                <li key={t.id}>
                  <button
                    type="button"
                    className={
                      "chatbot-sidebar__item" +
                      (t.id === activeThreadId ? " chatbot-sidebar__item--active" : "")
                    }
                    onClick={() => setActiveThreadId(t.id)}
                  >
                    <span className="chatbot-sidebar__item-title">{t.title}</span>
                    <span className="chatbot-sidebar__item-meta">
                      {formatThreadTime(t.updatedAt)}
                    </span>
                    {t.preview ? (
                      <span className="chatbot-sidebar__item-preview">{t.preview}</span>
                    ) : null}
                  </button>
                </li>
              ))}
            </ul>
          </aside>
        ) : null}

        <div className="chatbot-main">
          <header className="chatbot-page__head">
            <h1 className="chatbot-page__title">챗봇</h1>
            <p className="chatbot-page__sub">
              현재 챗봇에서는 발표 자료를 <strong>PDF 형식으로 첨부한 경우에만</strong> 분석이 가능합니다.
              <br />
              PPT 파일은 직접 분석되지 않을 수 있으니, 발표 자료를 먼저 PDF로 변환한 뒤 업로드해 주세요.
            </p>
            {!uid ? (
              <p className="chatbot-page__hint">
                로그인하면 대화가 계정에 저장되고, 새로고침 후에도 이어집니다.{" "}
                <Link to="/login">로그인</Link>
              </p>
            ) : null}
            {uid && firestoreError ? (
              <p className="chatbot-page__hint chatbot-page__hint--error" role="alert">
                {firestoreError}
              </p>
            ) : null}
            {uid && persistHint ? (
              <p className="chatbot-page__hint chatbot-page__hint--warn" role="status">
                {persistHint}
              </p>
            ) : null}
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
              accept="*/*"
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
                    if (canSend) void handleSend();
                  }
                }}
              />
              <button
                type="button"
                className="chatbot-send"
                aria-label="보내기"
                onClick={() => void handleSend()}
                disabled={!canSend}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
                  <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" fill="currentColor" />
                </svg>
              </button>
            </footer>
          </div>
        </div>
      </div>
    </div>
  );
}
