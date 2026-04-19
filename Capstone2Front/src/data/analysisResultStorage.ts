import type { CategoryScores } from "./firestoreModel";
import type { RubricCategoryId } from "./rubric";
import { RUBRIC } from "./rubric";

/** 루브릭 3영역 점수 — Firestore `GradingScores.scores` 와 동일 구조 */
export type StoredRubricScores = Record<RubricCategoryId, CategoryScores>;

const STORAGE_KEY = "overnight-analysis-result-v1";

/** 제출(세션)별 채점 — `submissionId` → 점수 */
const BY_SUBMISSION_KEY = "overnight-analysis-by-submission-v1";

function isCategoryScores(raw: unknown, itemCount: number): raw is CategoryScores {
  if (!raw || typeof raw !== "object") return false;
  const o = raw as Record<string, unknown>;
  if (typeof o.category !== "number") return false;
  if (!Array.isArray(o.items)) return false;
  if (o.items.length !== itemCount) return false;
  return o.items.every((x) => typeof x === "number");
}

export function parseStoredRubricScores(raw: unknown): StoredRubricScores | null {
  if (!raw || typeof raw !== "object") return null;
  const obj = raw as Record<string, unknown>;
  const out: Partial<StoredRubricScores> = {};
  for (const cat of RUBRIC) {
    const c = obj[cat.id];
    if (!isCategoryScores(c, cat.items.length)) return null;
    out[cat.id] = c;
  }
  return out as StoredRubricScores;
}

export function loadAnalysisResult(): StoredRubricScores | null {
  try {
    const s = localStorage.getItem(STORAGE_KEY);
    if (!s) return null;
    return parseStoredRubricScores(JSON.parse(s));
  } catch {
    return null;
  }
}

function loadSubmissionScoreMap(): Record<string, StoredRubricScores> {
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

/**
 * 분석 화면용: URL에 `submissionId`가 있으면 해당 제출 점수만, 없으면 예전 전역 저장(하위 호환).
 */
export function loadScoresForView(submissionIdFromUrl: string | null): StoredRubricScores | null {
  if (submissionIdFromUrl) {
    const map = loadSubmissionScoreMap();
    return map[submissionIdFromUrl] ?? null;
  }
  return loadAnalysisResult();
}

/** 채점 API·Firestore 동기화 후 호출하면 Analysis 화면에 반영됩니다. */
export function saveAnalysisResult(scores: StoredRubricScores): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(scores));
}

/** 특정 제출에 대한 채점 결과 저장 — 발표 기록에서 제출별로 조회할 때 사용 */
export function saveAnalysisResultForSubmission(submissionId: string, scores: StoredRubricScores): void {
  if (!submissionId) return;
  const map = loadSubmissionScoreMap();
  map[submissionId] = scores;
  localStorage.setItem(BY_SUBMISSION_KEY, JSON.stringify(map));
}

export function clearAnalysisResult(): void {
  localStorage.removeItem(STORAGE_KEY);
}

export function totalFromScores(scores: StoredRubricScores): number {
  return Math.round(
    RUBRIC.reduce((sum, cat) => sum + scores[cat.id].category, 0) / RUBRIC.length
  );
}
