import type { FolderFileKind, FolderFileRecord, FolderSubmission, SubmissionFile } from "./folderFileTypes";

export type { FolderFileKind, FolderSubmission, SubmissionFile } from "./folderFileTypes";

import { foldersUseFirestore } from "./folderFirestore";
import { getSubmissionsCache, registerSubmissionInFirestore } from "./folderSubmissionsFirestore";

const LEGACY_SUBMISSIONS_KEY = "overnight-folder-submissions-v1";
const LEGACY_FILES_KEY = "overnight-folder-files-v1";

function storeKey(scopeId: string): string {
  return `overnight-folder-submissions-v2:${scopeId}`;
}

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
  if (o.presentationTitle !== undefined && typeof o.presentationTitle !== "string") return false;
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

function mergeLegacyIntoStore(scopeId: string, store: Store): Store {
  const legacyRaw = localStorage.getItem(LEGACY_FILES_KEY);
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
    localStorage.removeItem(LEGACY_FILES_KEY);
    saveAllLocal(scopeId, next);
    return next;
  } catch {
    return store;
  }
}

function migrateLegacySubmissionsGuest(scopeId: string): Store | null {
  if (scopeId !== "__guest__") return null;
  const raw = localStorage.getItem(LEGACY_SUBMISSIONS_KEY);
  if (!raw) return null;
  try {
    const store = parseStore(JSON.parse(raw) as unknown);
    localStorage.removeItem(LEGACY_SUBMISSIONS_KEY);
    saveAllLocal(scopeId, store);
    return store;
  } catch {
    localStorage.removeItem(LEGACY_SUBMISSIONS_KEY);
    return {};
  }
}

function loadAllLocal(scopeId: string): Store {
  let store: Store = {};
  try {
    const raw = localStorage.getItem(storeKey(scopeId));
    if (raw) {
      store = parseStore(JSON.parse(raw) as unknown);
    } else {
      const migrated = migrateLegacySubmissionsGuest(scopeId);
      if (migrated !== null) store = migrated;
    }
  } catch {
    store = {};
  }

  if (localStorage.getItem(LEGACY_FILES_KEY)) {
    store = mergeLegacyIntoStore(scopeId, store);
  }
  return store;
}

function saveAllLocal(scopeId: string, store: Store): void {
  localStorage.setItem(storeKey(scopeId), JSON.stringify(store));
}

function loadAll(scopeId: string): Store {
  if (foldersUseFirestore(scopeId)) {
    return getSubmissionsCache();
  }
  return loadAllLocal(scopeId);
}

/** 최신 제출이 위로 오도록 */
export function listFolderSubmissions(scopeId: string, folderId: string): FolderSubmission[] {
  const all = loadAll(scopeId);
  const list = all[folderId] ?? [];
  return [...list].sort((a, b) => b.submittedAt.localeCompare(a.submittedAt));
}

/** 제출 ID로 검색 */
export function findSubmissionById(scopeId: string, submissionId: string): FolderSubmission | null {
  if (!submissionId) return null;
  const all = loadAll(scopeId);
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
 * 발표 평가 「채점 시작」 시 호출.
 * 로그인: Firestore · 게스트: localStorage
 */
export async function registerFolderFiles(
  scopeId: string,
  folderId: string,
  files: { pptName?: string | null; videoName?: string | null; presentationTitle?: string | null }
): Promise<FolderSubmission | null> {
  if (!folderId) return null;

  const presentationTitle = (files.presentationTitle ?? "").trim();

  if (foldersUseFirestore(scopeId)) {
    return registerSubmissionInFirestore(scopeId, folderId, {
      ...files,
      presentationTitle: presentationTitle || undefined,
    });
  }

  const submissionFiles: SubmissionFile[] = [];
  const push = (name: string | null | undefined, kind: FolderFileKind) => {
    const n = name?.trim();
    if (!n) return;
    submissionFiles.push({ id: newId(), name: n, kind });
  };
  push(files.pptName, "ppt");
  push(files.videoName, "video");
  if (submissionFiles.length === 0) return null;

  const all = loadAllLocal(scopeId);
  const submission: FolderSubmission = {
    id: newId(),
    folderId,
    submittedAt: new Date().toISOString(),
    ...(presentationTitle ? { presentationTitle } : {}),
    files: submissionFiles,
  };
  const list = all[folderId] ?? [];
  all[folderId] = [submission, ...list];
  saveAllLocal(scopeId, all);
  return submission;
}
