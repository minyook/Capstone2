import type { RubricCategoryId } from "./rubric";
import { RUBRIC } from "./rubric";
import { foldersUseFirestore } from "./folderFirestore";
import { getAnalysisScoresCache, saveAnalysisScoresToFirestore } from "./analysisScoresFirestore";
import type { StoredRubricScores } from "./storedRubricScores";
import { parseStoredRubricScores } from "./storedRubricScores";

export type { StoredRubricScores } from "./storedRubricScores";
export { parseStoredRubricScores } from "./storedRubricScores";

const STORAGE_KEY = "overnight-analysis-result-v1";
const BY_SUBMISSION_KEY = "overnight-analysis-by-submission-v1";

function loadAnalysisResultLocal(): StoredRubricScores | null {
  try {
    const s = localStorage.getItem(STORAGE_KEY);
    if (!s) return null;
    return parseStoredRubricScores(JSON.parse(s));
  } catch {
    return null;
  }
}

function loadSubmissionScoreMapLocal(): Record<string, StoredRubricScores> {
  try {
    const raw = localStorage.getItem(BY_SUBMISSION_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object") return {};
    const out: Record<string, StoredRubricScores> = {};
    for (const [id, val] of Object.entries(parsed)) {
      const scores = parseStoredRubricScores(val);
      if (scores) out[id] = scores;
    }
    return out;
  } catch {
    return {};
  }
}

function loadScoresForViewGuest(submissionIdFromUrl: string | null): StoredRubricScores | null {
  if (submissionIdFromUrl) {
    const map = loadSubmissionScoreMapLocal();
    return map[submissionIdFromUrl] ?? null;
  }
  return loadAnalysisResultLocal();
}

/**
 * 로그인: Firestore 캐시 (`getAnalysisScoresCache`). 리렌더는 `useFirestoreSyncRevision()` 포함.
 * 게스트: localStorage.
 */
export function loadScoresForView(
  scopeId: string | null,
  submissionIdFromUrl: string | null
): StoredRubricScores | null {
  if (!scopeId || !foldersUseFirestore(scopeId)) {
    return loadScoresForViewGuest(submissionIdFromUrl);
  }
  const c = getAnalysisScoresCache();
  if (submissionIdFromUrl) {
    return c.bySubmission[submissionIdFromUrl] ?? null;
  }
  return c.global;
}

export function saveAnalysisResult(scopeId: string | null, scores: StoredRubricScores): void {
  if (scopeId && foldersUseFirestore(scopeId)) {
    void saveAnalysisScoresToFirestore(scopeId, null, scores);
    return;
  }
  localStorage.setItem(STORAGE_KEY, JSON.stringify(scores));
}

export function saveAnalysisResultForSubmission(
  scopeId: string | null,
  submissionId: string,
  scores: StoredRubricScores
): void {
  if (!submissionId) return;
  if (scopeId && foldersUseFirestore(scopeId)) {
    void saveAnalysisScoresToFirestore(scopeId, submissionId, scores);
    return;
  }
  const map = loadSubmissionScoreMapLocal();
  map[submissionId] = scores;
  localStorage.setItem(BY_SUBMISSION_KEY, JSON.stringify(map));
}

export function clearAnalysisResult(): void {
  localStorage.removeItem(STORAGE_KEY);
}

export function totalFromScores(scores: StoredRubricScores): number {
  return Math.round(
    RUBRIC.reduce((sum, cat) => sum + scores[cat.id as RubricCategoryId].category, 0)
  );
}
