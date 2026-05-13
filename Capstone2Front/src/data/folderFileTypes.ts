export type FolderFileKind = "ppt" | "video";

/** 한 번의 발표 평가 제출에 포함된 파일 */
export type SubmissionFile = {
  id: string;
  name: string;
  kind: FolderFileKind;
};

/** 폴더 안에서 한 번 채점 시작을 눌렀을 때의 묶음 (같은 폴더에 발표를 여러 번 할 수 있음) */
export type FolderSubmission = {
  id: string;
  folderId: string;
  /** 제출 시각 ISO 8601 */
  submittedAt: string;
  /** 발표 평가에서 입력한 소제목 — 문서 화면에서 하위 그룹으로 묶음 */
  presentationTitle?: string;
  files: SubmissionFile[];
};

/** @deprecated 마이그레이션용 — 예전 flat 목록 */
export type FolderFileRecord = {
  id: string;
  folderId: string;
  name: string;
  kind: FolderFileKind;
  createdAt: string;
};
