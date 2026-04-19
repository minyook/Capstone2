import { useEffect, useMemo, useState } from "react";

import { Link, useLocation, useNavigate } from "react-router-dom";

import { IconArrowLeft } from "../components/Icons";

import { useFolders } from "../context/FoldersContext";

import { formatFolderDate } from "../data/folderStorage";
import {
  listFolderSubmissions,
  submissionPrimaryFileName,
} from "../data/folderFilesStorage";

import "./Notes.css";



const NOTES_FOLDER_KEY = "overnight-notes-selected-folder";

const NOTES_SUB_BY_FOLDER = "overnight-notes-submission-by-folder";

function readSubmissionByFolder(): Record<string, string> {
  try {
    const raw = localStorage.getItem(NOTES_SUB_BY_FOLDER);
    return raw ? (JSON.parse(raw) as Record<string, string>) : {};
  } catch {
    return {};
  }
}

function writeSubmissionForFolder(folderId: string, submissionId: string): void {
  const m = readSubmissionByFolder();
  m[folderId] = submissionId;
  try {
    localStorage.setItem(NOTES_SUB_BY_FOLDER, JSON.stringify(m));
  } catch {
    /* ignore */
  }
}



export function Notes() {

  const { folders } = useFolders();

  const navigate = useNavigate();

  const location = useLocation();

  const [folderSheet, setFolderSheet] = useState(false);

  const [fileSheet, setFileSheet] = useState(false);

  const [selectedSubmissionId, setSelectedSubmissionId] = useState<string | null>(null);

  const [selectedFolderId, setSelectedFolderId] = useState<string>(() => {

    try {

      return localStorage.getItem(NOTES_FOLDER_KEY) ?? "";

    } catch {

      return "";

    }

  });



  useEffect(() => {

    if (folders.length === 0) {

      setSelectedFolderId("");

      try {

        localStorage.removeItem(NOTES_FOLDER_KEY);

      } catch {

        /* ignore */

      }

      return;

    }

    const exists = folders.some((f) => f.id === selectedFolderId);

    if (!selectedFolderId || !exists) {

      setSelectedFolderId(folders[0].id);

    }

  }, [folders, selectedFolderId]);



  useEffect(() => {

    if (!selectedFolderId) return;

    try {

      localStorage.setItem(NOTES_FOLDER_KEY, selectedFolderId);

    } catch {

      /* ignore */

    }

  }, [selectedFolderId]);



  useEffect(() => {

    if (!selectedFolderId) {

      setSelectedSubmissionId(null);

      return;

    }

    const map = readSubmissionByFolder();

    const saved = map[selectedFolderId];

    const subs = listFolderSubmissions(selectedFolderId);

    if (saved && subs.some((s) => s.id === saved)) {

      setSelectedSubmissionId(saved);

    } else {

      setSelectedSubmissionId(null);

    }

  }, [selectedFolderId]);



  const folderSubmissions = useMemo(

    () => (selectedFolderId ? listFolderSubmissions(selectedFolderId) : []),

    [selectedFolderId, fileSheet, location.key]

  );



  const selected = folders.find((f) => f.id === selectedFolderId);

  const pickedSubmission =

    selectedSubmissionId && selected

      ? folderSubmissions.find((s) => s.id === selectedSubmissionId)

      : undefined;



  return (

    <div className="page notes-page">

      <div className="page-inner">

        <Link to="/projects" className="notes-back" aria-label="뒤로">

          <IconArrowLeft />

        </Link>

        <h1 className="notes-title">발표 기록</h1>

        <p className="notes-sub">폴더와 파일을 골라 녹화·채점 내역을 확인합니다</p>



        {folders.length === 0 ? (

          <p className="notes-empty">

            폴더가 없습니다.{" "}

            <Link to="/projects" className="notes-empty__link">

              문서에서 폴더 만들기

            </Link>

          </p>

        ) : (

          <>

            <button type="button" className="notes-select" onClick={() => setFolderSheet(true)}>

              <span className="notes-select__label">저장 폴더</span>

              <span className="notes-select__row">

                <span>{selected?.name ?? "선택"}</span>

                <span className="notes-select__ico" aria-hidden>

                  📁

                </span>

              </span>

            </button>



            <button type="button" className="notes-select" onClick={() => setFileSheet(true)}>

              <span className="notes-select__label">저장 파일 (제출별)</span>

              <span className="notes-select__row">

                <span className="notes-select__value">

                  {pickedSubmission

                    ? `${submissionPrimaryFileName(pickedSubmission)} · ${formatFolderDate(pickedSubmission.submittedAt)}`

                    : folderSubmissions.length === 0

                      ? "아직 제출 없음"

                      : "기록을 선택해 주세요"}

                </span>

                <span className="notes-select__ico" aria-hidden>

                  📄

                </span>

              </span>

            </button>

          </>

        )}

      </div>



      {folderSheet && (

        <div className="sheet-root" role="dialog" aria-modal="true" aria-labelledby="sheet-f-title">

          <button type="button" className="sheet-backdrop" onClick={() => setFolderSheet(false)} aria-label="닫기" />

          <div className="sheet-panel">

            <div className="sheet-handle" />

            <div className="sheet-head">

              <span className="sheet-head__icon">📁</span>

              <div>

                <h2 id="sheet-f-title" className="sheet-head__title">

                  폴더 선택

                </h2>

                <p className="sheet-head__sub">발표 영상·자료가 저장된 폴더를 고릅니다.</p>

              </div>

            </div>

            <hr className="sheet-rule" />

            {folders.map((f) => (

              <button

                key={f.id}

                type="button"

                className="sheet-item"

                onClick={() => {

                  setSelectedFolderId(f.id);

                  setFolderSheet(false);

                }}

              >

                <span className="sheet-item__icon">📁</span>

                <div>

                  <strong>{f.name}</strong>

                  <span className="sheet-item__meta">{formatFolderDate(f.createdAt)}</span>

                </div>

              </button>

            ))}

          </div>

        </div>

      )}



      {fileSheet && (

        <div className="sheet-root" role="dialog" aria-modal="true" aria-labelledby="sheet-file-title">

          <button type="button" className="sheet-backdrop" onClick={() => setFileSheet(false)} aria-label="닫기" />

          <div className="sheet-panel">

            <div className="sheet-handle" />

            <div className="sheet-head">

              <span className="sheet-head__icon" aria-hidden>

                📄

              </span>

              <div>

                <h2 id="sheet-file-title" className="sheet-head__title">

                  발표 파일 · 제출 선택

                </h2>

                <p className="sheet-head__sub">

                  {selected

                    ? `「${selected.name}」폴더에서 채점 시작할 때마다 쌓인 제출입니다. 고르면 채점 결과 화면으로 이동합니다.`

                    : "폴더를 선택해 주세요."}

                </p>

              </div>

            </div>

            <hr className="sheet-rule" />

            {selected && folderSubmissions.length === 0 ? (

              <p className="notes-file-hint">

                이 폴더에 아직 제출이 없습니다.{" "}

                <Link to="/evaluate" className="notes-empty__link" onClick={() => setFileSheet(false)}>

                  발표 평가

                </Link>

                에서 자료를 올리면 여기에 표시됩니다.

              </p>

            ) : selected ? (

              <ul className="notes-submission-sheet-list">

                {folderSubmissions.map((sub) => (

                  <li key={sub.id}>

                    <button

                      type="button"

                      className="sheet-item notes-submission-sheet-item"

                      onClick={() => {

                        if (!selectedFolderId) return;

                        writeSubmissionForFolder(selectedFolderId, sub.id);

                        setSelectedSubmissionId(sub.id);

                        setFileSheet(false);

                        navigate(`/analysis?submissionId=${encodeURIComponent(sub.id)}`);

                      }}

                    >

                      <span className="sheet-item__icon" aria-hidden>

                        📋

                      </span>

                      <div>

                        <strong>{submissionPrimaryFileName(sub)}</strong>

                        <span className="sheet-item__meta">

                          제출 {formatFolderDate(sub.submittedAt)} · 채점 결과 보기

                        </span>

                      </div>

                    </button>

                  </li>

                ))}

              </ul>

            ) : (

              <p className="notes-file-hint">폴더를 먼저 선택해 주세요.</p>

            )}

            <button type="button" className="sheet-file-row" onClick={() => setFileSheet(false)}>

              닫기

            </button>

          </div>

        </div>

      )}

    </div>

  );

}

