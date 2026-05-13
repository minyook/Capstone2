import { useEffect, useMemo, useState } from "react";

import { Link } from "react-router-dom";

import { useFirestoreSyncRevision } from "../context/FirestoreSyncContext";
import { useFolders } from "../context/FoldersContext";
import { formatFolderDate } from "../data/folderStorage";
import type { FolderSubmission, SubmissionFile } from "../data/folderFileTypes";
import { listFolderSubmissions } from "../data/folderFilesStorage";

import "./Notes.css";
import "./Projects.css";

const SESSION_VIDEO_PREVIEW_KEY = "overnight-video-preview-by-submission-v1";
const SESSION_PPT_BLOB_KEY = "overnight-ppt-blob-by-submission-v1";

function readSubmissionUrlMap(key: string): Record<string, string> {
  try {
    const raw = sessionStorage.getItem(key);
    return raw ? (JSON.parse(raw) as Record<string, string>) : {};
  } catch {
    return {};
  }
}

type FileViewerState = {
  submissionId: string;
  file: SubmissionFile;
  url: string | null;
} | null;

function groupSubmissionsByPresentationTitle(subs: FolderSubmission[]): {
  title: string;
  submissions: FolderSubmission[];
}[] {
  const order: string[] = [];
  const map = new Map<string, FolderSubmission[]>();
  for (const s of subs) {
    const label = s.presentationTitle?.trim() ? s.presentationTitle.trim() : "(제목 없음)";
    if (!map.has(label)) {
      map.set(label, []);
      order.push(label);
    }
    map.get(label)!.push(s);
  }
  return order.map((title) => ({ title, submissions: map.get(title)! }));
}



type SortKey = "latest" | "name";



export function Projects() {

  const { folders, createFolder, removeFolder, scopeId } = useFolders();

  const fsRevision = useFirestoreSyncRevision();

  const [newFolderOpen, setNewFolderOpen] = useState(false);

  const [folderName, setFolderName] = useState("");

  const [folderError, setFolderError] = useState("");

  const [sort, setSort] = useState<SortKey>("latest");

  const [fileSheetFolderId, setFileSheetFolderId] = useState<string | null>(null);

  /** null: 제목(하위) 목록 / 문자열: 해당 제목 안의 제출·파일 */
  const [fileSheetGroupTitle, setFileSheetGroupTitle] = useState<string | null>(null);

  const [fileViewer, setFileViewer] = useState<FileViewerState>(null);

  const sorted = useMemo(() => {

    const copy = [...folders];

    if (sort === "name") {

      copy.sort((a, b) => a.name.localeCompare(b.name, "ko"));

    } else {

      copy.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());

    }

    return copy;

  }, [folders, sort]);

  const fileSheetFolder = useMemo(
    () => (fileSheetFolderId ? folders.find((x) => x.id === fileSheetFolderId) : undefined),
    [fileSheetFolderId, folders]
  );

  const fileSheetSubmissions = useMemo(
    () => (fileSheetFolderId ? listFolderSubmissions(scopeId, fileSheetFolderId) : []),
    [scopeId, fileSheetFolderId, fsRevision]
  );

  const submissionGroups = useMemo(
    () => groupSubmissionsByPresentationTitle(fileSheetSubmissions),
    [fileSheetSubmissions]
  );

  const activeSubmissionGroup = useMemo(() => {
    if (!fileSheetGroupTitle) return null;
    return submissionGroups.find((g) => g.title === fileSheetGroupTitle) ?? null;
  }, [fileSheetGroupTitle, submissionGroups]);

  useEffect(() => {
    if (!fileSheetFolderId) {
      setFileSheetGroupTitle(null);
    }
  }, [fileSheetFolderId]);

  useEffect(() => {
    if (fileSheetFolderId && !folders.some((f) => f.id === fileSheetFolderId)) {
      setFileSheetFolderId(null);
    }
  }, [folders, fileSheetFolderId]);

  useEffect(() => {
    if (!fileViewer) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setFileViewer(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [fileViewer]);

  const handleCreate = async () => {

    setFolderError("");

    const ok = await createFolder(folderName);

    if (ok) {

      setFolderName("");

      setNewFolderOpen(false);

    } else {

      setFolderError("1~40자 공백이 아닌 이름을 입력해 주세요.");

    }

  };



  const handleDelete = (id: string, name: string) => {

    if (
      window.confirm(
        `「${name}」 폴더를 휴지통으로 옮길까요?\n휴지통에서 복구하거나 영구 삭제할 수 있습니다.`
      )
    ) {
      removeFolder(id);
    }

  };

  const openFileViewer = (submissionId: string, file: SubmissionFile) => {
    const vMap = readSubmissionUrlMap(SESSION_VIDEO_PREVIEW_KEY);
    const pMap = readSubmissionUrlMap(SESSION_PPT_BLOB_KEY);
    const url = file.kind === "video" ? vMap[submissionId] ?? null : pMap[submissionId] ?? null;
    setFileViewer({ submissionId, file, url });
  };



  return (

    <div className="doc-page">

      <header className="doc-toolbar">

        <h1 className="doc-toolbar__title">문서</h1>

        <div className="doc-toolbar__actions">
          <label className="doc-tool-sort">
            <span className="visually-hidden">정렬</span>
            <select
              className="doc-tool-select"
              value={sort}
              onChange={(e) => setSort(e.target.value as SortKey)}
            >
              <option value="latest">최신순</option>
              <option value="name">이름순</option>
            </select>
          </label>
        </div>

      </header>



      <p className="doc-lead">

        발표 영상·PPT·채점 결과를 주제별 폴더에 모읍니다. 폴더 목록은 <strong>이 기기</strong>에만 저장되며, 다른 기기와는
        자동으로 맞춰지지 않을 수 있습니다.

      </p>



      <ul className="doc-grid">

        <li>
          <button type="button" className="doc-card doc-card--new" onClick={() => setNewFolderOpen(true)}>
            <div className="doc-card__thumb doc-card__thumb--new" aria-hidden>
              <span className="doc-card__new-plus">+</span>
            </div>
            <div className="doc-card__body doc-card__body--new">
              <span className="doc-card__name">새 폴더</span>
              <span className="doc-card__meta">이름을 정하고 추가</span>
            </div>
          </button>
        </li>

        {sorted.map((f) => (

          <li key={f.id}>

            <article
              className="doc-card doc-card--clickable"
              role="button"
              tabIndex={0}
              onClick={() => {
                setFileSheetGroupTitle(null);
                setFileSheetFolderId(f.id);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  setFileSheetGroupTitle(null);
                  setFileSheetFolderId(f.id);
                }
              }}
              aria-label={`${f.name} 폴더 — 저장 파일 보기`}
            >

              <div className="doc-card__thumb">

                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" aria-hidden>

                  <path

                    d="M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V7z"

                    fill="currentColor"

                  />

                </svg>

              </div>

              <div className="doc-card__body">

                <h2 className="doc-card__name">{f.name}</h2>

                <time className="doc-card__date" dateTime={f.createdAt}>

                  {formatFolderDate(f.createdAt)}

                </time>

                <button
                  type="button"
                  className="doc-card__menu"
                  aria-label={`${f.name} 폴더 삭제`}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDelete(f.id, f.name);
                  }}
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
                    <path
                      d="M4 7h16M9 7V5a2 2 0 012-2h2a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V7h14zM10 11v6M14 11v6"
                      stroke="currentColor"
                      strokeWidth="1.75"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </button>

              </div>

            </article>

          </li>

        ))}

      </ul>



      <aside className="doc-foot" aria-label="다음 단계">
        <p className="doc-foot__text">
          <Link to="/evaluate">발표 평가</Link>에서 폴더를 선택한 뒤 녹화·업로드할 수 있습니다.{" "}
          <Link to="/notes">기록 상세</Link>에서 메모를 확인하세요.
        </p>
      </aside>



      {newFolderOpen && (

        <div className="modal-root" role="dialog" aria-modal="true" aria-labelledby="nf-title">

          <div className="modal-backdrop" onClick={() => { setNewFolderOpen(false); setFolderError(""); }} />

          <div className="modal-card">

            <div className="modal-card__head">

              <span className="modal-card__icon">📁</span>

              <div>

                <h2 id="nf-title" className="modal-card__title">

                  새 폴더

                </h2>

                <p className="modal-card__sub">이 폴더에 녹화·PPT·채점 기록이 함께 저장됩니다.</p>

              </div>

            </div>

            <input

              className="modal-input"

              placeholder="예: 중간발표, 캡스톤 최종"

              maxLength={40}

              value={folderName}

              onChange={(e) => {

                setFolderName(e.target.value);

                setFolderError("");

              }}

              onKeyDown={(e) => {

                if (e.key === "Enter") handleCreate();

              }}

            />

            <div className="modal-meta">{folderName.length}/40</div>

            {folderError ? <p className="modal-error">{folderError}</p> : null}

            <div className="modal-actions">

              <button

                type="button"

                className="modal-link"

                onClick={() => {

                  setNewFolderOpen(false);

                  setFolderError("");

                }}

              >

                취소

              </button>

              <button type="button" className="modal-link" onClick={handleCreate}>

                만들기

              </button>

            </div>

          </div>

        </div>

      )}

      {fileSheetFolderId && fileSheetFolder && (
        <div className="sheet-root" role="dialog" aria-modal="true" aria-labelledby="projects-sheet-files-title">
          <button
            type="button"
            className="sheet-backdrop"
            onClick={() => setFileSheetFolderId(null)}
            aria-label="닫기"
          />
          <div className="sheet-panel">
            <div className="sheet-handle" />
            <div className="sheet-head">
              <span className="sheet-head__icon" aria-hidden>
                📄
              </span>
              <div>
                <h2 id="projects-sheet-files-title" className="sheet-head__title">
                  저장 파일
                </h2>
                <p className="sheet-head__sub">
                  {fileSheetGroupTitle && activeSubmissionGroup
                    ? `「${fileSheetFolder.name}」 / ${activeSubmissionGroup.title}`
                    : `「${fileSheetFolder.name}」 — 발표 제목(하위)을 누르면 PPT·영상을 볼 수 있습니다.`}
                </p>
              </div>
            </div>
            <hr className="sheet-rule" />
            {fileSheetSubmissions.length === 0 ? (
              <p className="notes-file-hint doc-projects-file-empty">
                저장된 파일이 없습니다.{" "}
                <Link to="/evaluate" className="notes-empty__link" onClick={() => setFileSheetFolderId(null)}>
                  발표 평가
                </Link>
                에서 PPT·영상을 올리면 여기에 표시됩니다.
              </p>
            ) : !fileSheetGroupTitle ? (
              <ul className="doc-projects-group-picker">
                {submissionGroups.map((group) => (
                  <li key={group.title}>
                    <button
                      type="button"
                      className="doc-projects-group-card"
                      onClick={() => setFileSheetGroupTitle(group.title)}
                    >
                      <span className="doc-projects-group-card__icon" aria-hidden>
                        📂
                      </span>
                      <span className="doc-projects-group-card__body">
                        <strong className="doc-projects-group-card__name">{group.title}</strong>
                        <span className="doc-projects-group-card__meta">
                          제출 {group.submissions.length}회
                        </span>
                      </span>
                      <span className="doc-projects-group-card__chev" aria-hidden>
                        ›
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            ) : activeSubmissionGroup ? (
              <>
                <button
                  type="button"
                  className="doc-projects-sheet-back"
                  onClick={() => setFileSheetGroupTitle(null)}
                >
                  ← 발표 제목 목록
                </button>
                <div className="doc-projects-submission-groups doc-projects-submission-groups--nested">
                  <ul className="doc-projects-submission-list">
                    {activeSubmissionGroup.submissions.map((submission) => (
                      <li key={submission.id} className="doc-projects-submission">
                        <p className="doc-projects-submission__label" id={`sub-${submission.id}`}>
                          제출 · {formatFolderDate(submission.submittedAt)}
                        </p>
                        <ul
                          className="doc-projects-file-list doc-projects-file-list--in-submission"
                          aria-labelledby={`sub-${submission.id}`}
                        >
                          {submission.files.map((file) => (
                            <li key={file.id}>
                              <button
                                type="button"
                                className="doc-projects-file-row doc-projects-file-row--clickable"
                                onClick={() => openFileViewer(submission.id, file)}
                              >
                                <span className="sheet-item__icon" aria-hidden>
                                  {file.kind === "ppt" ? "📊" : "🎬"}
                                </span>
                                <div className="doc-projects-file-row__text">
                                  <strong>{file.name}</strong>
                                  <span className="sheet-item__meta">
                                    {file.kind === "ppt"
                                      ? "프레젠테이션 · 탭하여 저장"
                                      : "영상 · 탭하여 재생"}
                                  </span>
                                </div>
                              </button>
                            </li>
                          ))}
                        </ul>
                      </li>
                    ))}
                  </ul>
                </div>
              </>
            ) : (
              <div className="notes-file-hint doc-projects-file-empty">
                <p>이 제목의 자료를 찾을 수 없습니다.</p>
                <button type="button" className="doc-projects-sheet-back" onClick={() => setFileSheetGroupTitle(null)}>
                  ← 목록으로
                </button>
              </div>
            )}
            <button
              type="button"
              className="sheet-file-row"
              onClick={() => {
                setFileSheetGroupTitle(null);
                setFileSheetFolderId(null);
              }}
            >
              닫기
            </button>
          </div>
        </div>
      )}

      {fileViewer ? (
        <div className="doc-file-viewer-root" role="dialog" aria-modal="true" aria-labelledby="doc-file-viewer-title">
          <button
            type="button"
            className="doc-file-viewer-backdrop"
            aria-label="닫기"
            onClick={() => setFileViewer(null)}
          />
          <div className="doc-file-viewer-panel">
            <h2 id="doc-file-viewer-title" className="doc-file-viewer-title">
              {fileViewer.file.kind === "ppt" ? "프레젠테이션" : "영상"} · {fileViewer.file.name}
            </h2>
            {fileViewer.url && fileViewer.file.kind === "video" ? (
              <video className="doc-file-viewer-video" src={fileViewer.url} controls playsInline />
            ) : null}
            {fileViewer.url && fileViewer.file.kind === "ppt" ? (
              <div className="doc-file-viewer-ppt">
                <p className="doc-file-viewer-note">아래 버튼으로 이 기기에 파일을 저장할 수 있습니다.</p>
                <div className="doc-file-viewer-actions">
                  <a
                    className="doc-file-viewer-link doc-file-viewer-link--primary doc-file-viewer-link--block"
                    href={fileViewer.url}
                    download={fileViewer.file.name}
                  >
                    파일로 저장
                  </a>
                </div>
              </div>
            ) : null}
            {!fileViewer.url ? (
              <p className="doc-file-viewer-miss">
                이 기기에서 <strong>발표 평가를 제출한 직후</strong> 같은 브라우저에만 미리보기가 남습니다. 다른 기기나
                나중에 다시 열면 파일을 다시 볼 수 없을 수 있습니다.
              </p>
            ) : null}
            <button type="button" className="doc-file-viewer-close" onClick={() => setFileViewer(null)}>
              닫기
            </button>
          </div>
        </div>
      ) : null}

    </div>

  );

}

