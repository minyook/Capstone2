import type { FolderFileKind, FolderFileRecord, FolderSubmission, SubmissionFile } from "./folderFileTypes";

export type { FolderFileKind, FolderSubmission, SubmissionFile } from "./folderFileTypes";

const STORAGE_KEY = "overnight-folder-submissions-v1";
const LEGACY_KEY = "overnight-folder-files-v1";

type Store = Record<string, FolderSubmission[]>;

function newId(): string {
  return typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `ff-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function isSubmissionFile(x: unknown): x is SubmissionFile {
  if (!x || typeof x !== "object") return false;
  const o = x as Record<string, unknown>;
  return (
    typeof o.id === "string" &&
    typeof o.name === "string" &&
    o.name.trim().length > 0 &&
    (o.kind === "ppt" || o.kind === "video")
  );
}

function isFolderSubmission(x: unknown): x is FolderSubmission {
  if (!x || typeof x !== "object") return false;
  const o = x as Record<string, unknown>;
  if (typeof o.id !== "string" || typeof o.folderId !== "string" || typeof o.submittedAt !== "string") return false;
  if (!Array.isArray(o.files)) return false;
  return o.files.every(isSubmissionFile);
}

function isLegacyFileRecord(x: unknown): x is FolderFileRecord {
  if (!x || typeof x !== "object") return false;
  const o = x as Record<string, unknown>;
  return (
    typeof o.id === "string" &&
    typeof o.folderId === "string" &&
    typeof o.name === "string" &&
    o.name.trim().length > 0 &&
    (o.kind === "ppt" || o.kind === "video") &&
    typeof o.createdAt === "string"
  );
}

function parseStore(raw: unknown): Store {
  if (!raw || typeof raw !== "object") return {};
  const out: Store = {};
  for (const [folderId, arr] of Object.entries(raw as Record<string, unknown>)) {
    if (!Array.isArray(arr)) continue;
    const subs = arr.filter(isFolderSubmission).filter((s) => s.folderId === folderId);
    if (subs.length) out[folderId] = subs;
  }
  return out;
}

/** 예전 flat 배열 → 제출 1건으로 묶어 병합 */
function mergeLegacyIntoStore(store: Store): Store {
  const legacyRaw = localStorage.getItem(LEGACY_KEY);
  if (!legacyRaw) return store;

  try {
    const parsed = JSON.parse(legacyRaw) as unknown;
    if (!parsed || typeof parsed !== "object") return store;

    const next: Store = { ...store };
    for (const [folderId, arr] of Object.entries(parsed as Record<string, unknown>)) {
      if (!Array.isArray(arr)) continue;
      const legacyFiles = arr.filter(isLegacyFileRecord).filter((f) => f.folderId === folderId);
      if (!legacyFiles.length) continue;

      const sorted = [...legacyFiles].sort((a, b) => a.createdAt.localeCompare(b.createdAt));
      const submission: FolderSubmission = {
        id: newId(),
        folderId,
        submittedAt: sorted[0].createdAt,
        files: sorted.map((f) => ({ id: f.id, name: f.name, kind: f.kind })),
      };
      const existing = next[folderId] ?? [];
      next[folderId] = [...existing, submission];
    }
    localStorage.removeItem(LEGACY_KEY);
    return next;
  } catch {
    return store;
  }
}

function loadAll(): Store {
  let store: Store = {};
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) store = parseStore(JSON.parse(raw) as unknown);
  } catch {
    store = {};
  }

  if (localStorage.getItem(LEGACY_KEY)) {
    store = mergeLegacyIntoStore(store);
    saveAll(store);
  }
  return store;
}

function saveAll(store: Store): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
}

/** 최신 제출이 위로 오도록 */
export function listFolderSubmissions(folderId: string): FolderSubmission[] {
  const all = loadAll();
  const list = all[folderId] ?? [];
  return [...list].sort((a, b) => b.submittedAt.localeCompare(a.submittedAt));
}

/** 제출 ID로 검색 (URL·채점 화면 연동용) */
export function findSubmissionById(submissionId: string): FolderSubmission | null {
  if (!submissionId) return null;
  const all = loadAll();
  for (const subs of Object.values(all)) {
    const found = subs.find((s) => s.id === submissionId);
    if (found) return found;
  }
  return null;
}

export function submissionPrimaryFileName(sub: FolderSubmission): string {
  const video = sub.files.find((f) => f.kind === "video");
  const ppt = sub.files.find((f) => f.kind === "ppt");
  return video?.name ?? ppt?.name ?? "제출";
}

/**
 * 발표 평가에서 「채점 시작」 시 호출 — 한 번의 제출로 묶어 문서·발표 기록「저장 파일」에 쌓습니다.
 */
export function registerFolderFiles(
  folderId: string,
  files: { pptName?: string | null; videoName?: string | null }
): FolderSubmission | null {
  if (!folderId) return null;

  const submissionFiles: SubmissionFile[] = [];
  const push = (name: string | null | undefined, kind: FolderFileKind) => {
    const n = name?.trim();
    if (!n) return;
    submissionFiles.push({ id: newId(), name: n, kind });
  };
  push(files.pptName, "ppt");
  push(files.videoName, "video");
  if (submissionFiles.length === 0) return null;

  const all = loadAll();
  const submission: FolderSubmission = {
    id: newId(),
    folderId,
    submittedAt: new Date().toISOString(),
    files: submissionFiles,
  };
  const list = all[folderId] ?? [];
  all[folderId] = [submission, ...list];
  saveAll(all);
  return submission;
}
