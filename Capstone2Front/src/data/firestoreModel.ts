/**
 * Firestore 데이터 모델 (Overnight — 폴더 · 발표 제출 · 멀티모달 채점)
 *
 * 컬렉션/필드 이름은 실제 write 시 그대로 맞추면 됩니다.
 * 보안 규칙: 아래 모든 문서는 ownerUid(또는 users/{uid})와 request.auth.uid 일치 시만 read/write.
 */

import type { RubricCategoryId } from "./rubric";

// ——— 컬렉션 ID ———

export const COL_USERS = "users";
export const COL_FOLDERS = "folders";
export const COL_GRADING_SESSIONS = "gradingSessions";

// ——— Storage 경로 패턴 (예시) ———
// gradingSessions/{sessionId}/uploads/ppt   + 확장자
// gradingSessions/{sessionId}/uploads/video + 확장자
// 또는 users/{uid}/grading/{sessionId}/...

/** users/{uid} — 이미 AuthContext·Register에서 쓰는 프로필과 동일 계열 */
export type FirestoreUserProfile = {
  email: string | null;
  displayName: string | null;
  photoURL: string | null;
  birthDate?: string | null;
  phone?: string | null;
  createdAt?: import("firebase/firestore").Timestamp;
  updatedAt?: import("firebase/firestore").Timestamp;
};

/**
 * folders/{folderId}
 * 문서(Projects) 화면의 “폴더” 한 개. 로컬 overnight-folders-v1 과 동기화 시 이 구조로 올리면 됨.
 */
export type FirestoreFolder = {
  /** Firebase Auth uid — 반드시 설정 (본인 폴더만 조회) */
  ownerUid: string;
  name: string;
  createdAt: import("firebase/firestore").Timestamp;
  /** 휴지통: 설정 시 문서 목록에서 숨기고 복구/영구삭제 플로우에 사용 */
  deletedAt?: import("firebase/firestore").Timestamp | null;
  /** 선택: 클라이언트가 쓰던 id 와 동일하게 두려면 문서 ID = 기존 folder id */
  legacyLocalId?: string;
};

/** 채점 파이프라인 상태 */
export type GradingStatus = "draft" | "queued" | "processing" | "completed" | "failed";

/**
 * 항목별 점수 — `analysisResultStorage` / Analysis UI 와 동일한 형태
 * 백엔드(STT·비전·융합)가 채운다.
 */
export type CategoryScores = {
  category: number;
  items: number[];
};

export type GradingScores = Record<RubricCategoryId, CategoryScores> & {
  /** 선택: 총점 캐시 */
  total?: number;
};

/**
 * gradingSessions/{sessionId}
 * “한 번의 발표 제출·채점” = 폴더 안에 쌓이는 단위
 */
export type FirestoreGradingSession = {
  ownerUid: string;
  folderId: string;

  /** 표시용 (파일명 또는 사용자 입력) */
  label?: string;

  status: GradingStatus;

  /** 제출 메타 — 실제 바이너리는 Storage */
  input: {
    pptFileName: string;
    videoFileName: string;
    pptStoragePath?: string;
    videoStoragePath?: string;
  };

  /** 처리/알고리즘 버전 (재현·디버깅) */
  pipeline?: {
    version: string;
    notes?: string;
  };

  /** 완료 후 채점 결과 — Analysis 페이지 데이터 소스 */
  scores?: GradingScores;

  errorMessage?: string;

  createdAt: import("firebase/firestore").Timestamp;
  updatedAt: import("firebase/firestore").Timestamp;
  completedAt?: import("firebase/firestore").Timestamp | null;
};

/** 편의: 경로 빌더 */
export const paths = {
  user: (uid: string) => `${COL_USERS}/${uid}`,
  folder: (folderId: string) => `${COL_FOLDERS}/${folderId}`,
  gradingSession: (sessionId: string) => `${COL_GRADING_SESSIONS}/${sessionId}`,
};
