import {
  Timestamp,
  collection,
  doc,
  getDocs,
  onSnapshot,
  orderBy,
  query,
  serverTimestamp,
  setDoc,
  writeBatch,
  type Firestore,
  type QuerySnapshot,
} from "firebase/firestore";
import { db } from "../firebase/config";
import type { FolderSubmission, SubmissionFile } from "./folderFileTypes";
import type { FolderFileKind } from "./folderFileTypes";

type Store = Record<string, FolderSubmission[]>;

let cache: Store = {};
let unsub: (() => void) | undefined;

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

function rebuildCacheFromSnap(snap: QuerySnapshot): void {
  const byFolder: Store = {};
  snap.forEach((d) => {
    const data = d.data();
    const filesRaw = data.files;
    const files = Array.isArray(filesRaw) ? filesRaw.filter(isSubmissionFile) : [];
    const sub: FolderSubmission = {
      id: d.id,
      folderId: String(data.folderId ?? ""),
      submittedAt:
        data.submittedAt instanceof Timestamp
          ? data.submittedAt.toDate().toISOString()
          : typeof data.submittedAt === "string"
            ? data.submittedAt
            : new Date().toISOString(),
      ...(typeof data.presentationTitle === "string" && data.presentationTitle.trim()
        ? { presentationTitle: data.presentationTitle.trim() }
        : {}),
      files,
    };
    if (!sub.folderId) return;
    if (!byFolder[sub.folderId]) byFolder[sub.folderId] = [];
    byFolder[sub.folderId].push(sub);
  });
  for (const k of Object.keys(byFolder)) {
    byFolder[k].sort((a, b) => b.submittedAt.localeCompare(a.submittedAt));
  }
  cache = byFolder;
}

function parseLocalStore(raw: unknown): Store {
  if (!raw || typeof raw !== "object") return {};
  const out: Store = {};
  for (const [folderId, arr] of Object.entries(raw as Record<string, unknown>)) {
    if (!Array.isArray(arr)) continue;
    const subs = arr.filter((x): x is FolderSubmission => {
      if (!x || typeof x !== "object") return false;
      const o = x as Record<string, unknown>;
      return (
        typeof o.id === "string" &&
        typeof o.folderId === "string" &&
        typeof o.submittedAt === "string" &&
        Array.isArray(o.files)
      );
    });
    if (subs.length) out[folderId] = subs;
  }
  return out;
}

async function migrateLocalSubmissions(uid: string): Promise<void> {
  if (!db) return;
  const col = collection(db, "users", uid, "submissions");
  const snap = await getDocs(col);
  if (!snap.empty) return;

  const key = `overnight-folder-submissions-v2:${uid}`;
  const raw = localStorage.getItem(key);
  if (!raw) return;

  let store: Store = {};
  try {
    store = parseLocalStore(JSON.parse(raw) as unknown);
  } catch {
    return;
  }

  const all = Object.values(store).flat();
  if (all.length === 0) return;

  const fs = db as Firestore;
  let batch = writeBatch(fs);
  let n = 0;
  for (const arr of Object.values(store)) {
    for (const sub of arr) {
      const ref = doc(fs, "users", uid, "submissions", sub.id);
      batch.set(ref, {
        folderId: sub.folderId,
        submittedAt: Timestamp.fromDate(new Date(sub.submittedAt)),
        files: sub.files,
        ...(sub.presentationTitle?.trim() ? { presentationTitle: sub.presentationTitle.trim() } : {}),
      });
      n += 1;
      if (n >= 400) {
        await batch.commit();
        batch = writeBatch(fs);
        n = 0;
      }
    }
  }
  if (n > 0) await batch.commit();
  localStorage.removeItem(key);
}

export function getSubmissionsCache(): Store {
  return cache;
}

/** 로그인 사용자 제출 목록 실시간 동기화 */
export function startSubmissionsSync(uid: string, onChange: () => void): () => void {
  if (!db) return () => {};

  unsub?.();
  unsub = undefined;
  cache = {};
  let cancelled = false;

  void (async () => {
    try {
      await migrateLocalSubmissions(uid);
    } catch (e) {
      console.error("[submissions migrate]", e);
    }
    if (cancelled || !db) return;

    const q = query(collection(db, "users", uid, "submissions"), orderBy("submittedAt", "desc"));
    unsub = onSnapshot(
      q,
      (snap) => {
        rebuildCacheFromSnap(snap);
        onChange();
      },
      (err) => console.error("[submissions]", err)
    );
  })();

  return () => {
    cancelled = true;
    unsub?.();
    unsub = undefined;
    cache = {};
  };
}

export async function registerSubmissionInFirestore(
  uid: string,
  folderId: string,
  files: { pptName?: string | null; videoName?: string | null; presentationTitle?: string | null }
): Promise<FolderSubmission | null> {
  if (!db || !folderId) return null;

  const submissionFiles: SubmissionFile[] = [];
  const newFileId = () =>
    typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID()
      : `ff-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;

  const push = (name: string | null | undefined, kind: FolderFileKind) => {
    const n = name?.trim();
    if (!n) return;
    submissionFiles.push({ id: newFileId(), name: n, kind });
  };
  push(files.pptName, "ppt");
  push(files.videoName, "video");
  if (submissionFiles.length === 0) return null;

  const titleTrimmed = (files.presentationTitle ?? "").trim();

  const ref = doc(collection(db, "users", uid, "submissions"));
  const submittedAtIso = new Date().toISOString();
  await setDoc(ref, {
    folderId,
    submittedAt: serverTimestamp(),
    files: submissionFiles,
    ...(titleTrimmed ? { presentationTitle: titleTrimmed } : {}),
  });

  return {
    id: ref.id,
    folderId,
    submittedAt: submittedAtIso,
    ...(titleTrimmed ? { presentationTitle: titleTrimmed } : {}),
    files: submissionFiles,
  };
}
