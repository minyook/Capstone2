/** PPT ↔ 영상 슬라이드 일치 검증 결과 (sessionStorage) */

const STORAGE_KEY = "overnight-slide-verify-by-submission-v1";

export type SlideVerifyVerdict = "match" | "partial" | "mismatch";

export interface SlideVerifyIssue {
  time_sec: number;
  message: string;
  ppt_slide_index: number;
  confidence: number;
  issue_type?: string;
}

export interface SlideVerifyResult {
  verdict: SlideVerifyVerdict;
  overall_match_percent: number;
  visual_match_percent: number;
  slide_coverage_percent: number;
  sequence_complete: boolean;
  order_matches_ppt: boolean;
  missing_slides: number[];
  sequence_gaps: Array<{
    from_page: number;
    to_page: number;
    missing_pages: number[];
  }>;
  detected_slide_sequence: number[];
  issues: SlideVerifyIssue[];
  video_type?: string;
  render_mode?: string;
  diagnostics?: {
    slide_present_rate?: number | null;
    perspective_ok_rate?: number | null;
    screen_detection_rate?: number | null;
  };
  note?: string;
}

export function saveSlideVerifyForSubmission(
  submissionId: string,
  result: SlideVerifyResult
): void {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    const map = raw ? (JSON.parse(raw) as Record<string, SlideVerifyResult>) : {};
    map[submissionId] = result;
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(map));
  } catch {
    /* ignore quota */
  }
}

export function loadSlideVerifyForSubmission(
  submissionId: string | null
): SlideVerifyResult | null {
  if (!submissionId) return null;
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const map = JSON.parse(raw) as Record<string, SlideVerifyResult>;
    return map[submissionId] ?? null;
  } catch {
    return null;
  }
}

export function verdictLabelKo(verdict: string): string {
  if (verdict === "match") return "일치";
  if (verdict === "partial") return "부분 일치";
  return "불일치";
}

/** 이슈 유형별 요약 (낮은 일치율 원인 안내) */
export function summarizeLowMatchReasons(result: SlideVerifyResult): string[] {
  const reasons: string[] = [];
  const types = new Set((result.issues ?? []).map((i) => i.issue_type).filter(Boolean));

  if (types.has("missing_slide")) {
    reasons.push("PPT 일부 슬라이드가 영상에 나오지 않았습니다.");
  }
  if (types.has("skipped_sequence")) {
    reasons.push("슬라이드 순서가 건너뛰어진 구간이 있습니다.");
  }
  if (types.has("audience_no_screen")) {
    reasons.push("강의실 촬영에서 프로젝터 스크린이 잡히지 않은 구간이 많습니다.");
  }
  if (types.has("audience_low_perspective")) {
    reasons.push("스크린 각도·원근 보정이 불안정합니다.");
  }
  if (types.has("low_confidence")) {
    reasons.push("일부 구간에서 화면과 PPT 유사도가 낮습니다 (조명·가림·압축).");
  }
  if (types.has("order")) {
    reasons.push("슬라이드 진행 순서가 PPT와 맞지 않습니다.");
  }
  if (result.slide_coverage_percent < 80 && !types.has("missing_slide")) {
    reasons.push(`슬라이드 커버리지가 ${result.slide_coverage_percent}%로 낮습니다.`);
  }
  if (result.visual_match_percent < 50 && !types.has("low_confidence")) {
    reasons.push(`화면 유사도가 ${result.visual_match_percent}%입니다.`);
  }
  if (reasons.length === 0 && result.overall_match_percent < 60) {
    reasons.push("영상 화질·촬영 각도 또는 PPT 렌더 품질을 확인해 주세요.");
  }
  return reasons;
}
