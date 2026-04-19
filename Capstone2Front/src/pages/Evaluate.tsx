import { useState } from "react";

import { Link, useNavigate } from "react-router-dom";

import { useFolders } from "../context/FoldersContext";
import { registerFolderFiles } from "../data/folderFilesStorage";

import "./Evaluate.css";



type Step = 1 | 2 | 3;



export function Evaluate() {

  const navigate = useNavigate();

  const { folders } = useFolders();

  const [step, setStep] = useState<Step>(1);

  const [folderId, setFolderId] = useState<string>("");

  const [pptName, setPptName] = useState<string | null>(null);

  const [videoName, setVideoName] = useState<string | null>(null);



  const hasFolders = folders.length > 0;

  const canAnalyze = Boolean(folderId && pptName && videoName && hasFolders);



  return (

    <div className="page evaluate">

      <div className="page-inner page-inner--wide">

        <p className="evaluate-kicker">STT · 비전 · 융합 채점 파이프라인</p>

        <h1 className="evaluate-title">발표 자료 제출</h1>

        <p className="evaluate-lead">

          발표 음성은 글자로 옮겨 내용을 보고, 영상에서는 시선·제스처·표정을 함께 봅니다. <strong>저장 폴더</strong>를 고른

          뒤 PPT와 영상을 올리면 항목별 점수와 PDF·Excel 리포트로 이어집니다.

        </p>



        <section className="evaluate-panel evaluate-panel--folder" aria-labelledby="sf">

          <h2 id="sf" className="evaluate-panel__title">

            저장할 발표 폴더

          </h2>

          <p className="evaluate-panel__desc">

            녹화·업로드한 영상과 PPT, 이후 채점 결과가 이 폴더에 정리됩니다. 폴더는 <strong>문서</strong> 화면에서

            추가·삭제할 수 있습니다.

          </p>

          <div className="evaluate-folder-row">

            <label className="visually-hidden" htmlFor="evaluate-folder">

              발표 폴더 선택

            </label>

            {hasFolders ? (

              <select

                id="evaluate-folder"

                className="evaluate-folder-select"

                value={folderId}

                onChange={(e) => setFolderId(e.target.value)}

              >

                <option value="">저장할 폴더를 선택하세요</option>

                {folders.map((f) => (

                  <option key={f.id} value={f.id}>

                    {f.name}

                  </option>

                ))}

              </select>

            ) : (

              <p className="evaluate-folder-empty">

                등록된 폴더가 없습니다.{" "}

                <Link to="/projects" className="evaluate-folder-link">

                  문서에서 폴더 만들기

                </Link>

              </p>

            )}

            <Link to="/projects" className="evaluate-folder-link">

              폴더 관리

            </Link>

          </div>

        </section>



        <ol className="evaluate-steps" aria-label="진행 단계">

          <li className={step >= 1 ? "evaluate-steps__item--active" : ""}>

            <span className="evaluate-steps__num">1</span>

            PPT 업로드

          </li>

          <li className={step >= 2 ? "evaluate-steps__item--active" : ""}>

            <span className="evaluate-steps__num">2</span>

            발표 영상

          </li>

          <li className={step >= 3 ? "evaluate-steps__item--active" : ""}>

            <span className="evaluate-steps__num">3</span>

            분석 실행

          </li>

        </ol>



        <section className="evaluate-panel" aria-labelledby="s1">

          <h2 id="s1" className="evaluate-panel__title">

            ① PPT 파일

          </h2>

          <p className="evaluate-panel__desc">채점 기준과 비교할 슬라이드 자료를 올려주세요. (.pptx 등)</p>

          <label className="evaluate-drop">

            <input

              type="file"

              accept=".ppt,.pptx,application/vnd.ms-powerpoint,application/vnd.openxmlformats-officedocument.presentationml.presentation"

              className="evaluate-drop__input"

              onChange={(e) => {

                const f = e.target.files?.[0];

                setPptName(f?.name ?? null);

                if (f) setStep((s) => (s < 2 ? 2 : s));

              }}

            />

            <span className="evaluate-drop__ui">

              {pptName ? (

                <>

                  <strong>{pptName}</strong>

                  <span className="evaluate-drop__hint">다른 파일로 바꾸려면 클릭</span>

                </>

              ) : (

                <>

                  <span className="evaluate-drop__icon" aria-hidden>

                    📊

                  </span>

                  클릭하여 PPT 선택

                </>

              )}

            </span>

          </label>

        </section>



        <section className="evaluate-panel" aria-labelledby="s2">

          <h2 id="s2" className="evaluate-panel__title">

            ② 발표 영상

          </h2>

          <p className="evaluate-panel__desc">

            아래에서 녹화하거나 파일을 업로드하세요. 음성·얼굴이 나와야 태도·음성 항목을 채점할 수 있습니다.

          </p>

          <div className="evaluate-preview" role="region" aria-label="카메라 미리보기">

            카메라 미리보기와 녹화는 곧 이 화면에서 이용할 수 있도록 준비 중입니다. 지금은 아래에서 영상 파일을 선택해
            주세요.

          </div>

          <div className="evaluate-row">

            <button type="button" className="evaluate-btn evaluate-btn--secondary">

              녹화 시작

            </button>

            <label className="evaluate-btn evaluate-btn--ghost">

              영상 파일 선택

              <input

                type="file"

                accept="video/*"

                className="evaluate-drop__input"

                onChange={(e) => {

                  const f = e.target.files?.[0];

                  setVideoName(f?.name ?? null);

                  if (f) setStep((s) => (s < 3 ? 3 : s));

                }}

              />

            </label>

          </div>

          {videoName && (

            <p className="evaluate-filetag">

              선택됨: <strong>{videoName}</strong>

            </p>

          )}

        </section>



        <section className="evaluate-panel evaluate-panel--last" aria-labelledby="s3">

          <h2 id="s3" className="evaluate-panel__title">

            ③ 분석

          </h2>

          <p className="evaluate-panel__desc">

            준비가 되면 서버로 전송해 PPT·발표 대조 및 멀티모달 채점을 실행합니다.

          </p>

          <button

            type="button"

            className="evaluate-btn evaluate-btn--primary evaluate-btn--block"

            disabled={!canAnalyze}

            onClick={() => {
              const submission = registerFolderFiles(folderId, { pptName, videoName });
              navigate(
                submission
                  ? `/analysis?submissionId=${encodeURIComponent(submission.id)}`
                  : "/analysis"
              );
            }}

          >

            {canAnalyze

              ? "채점 시작하기"

              : !hasFolders

                ? "폴더를 먼저 만드세요"

                : !folderId || !pptName || !videoName

                  ? "폴더·PPT·영상을 모두 준비해 주세요"

                  : "채점 시작하기"}

          </button>

          {!canAnalyze && (

            <p className="evaluate-note">

              ※ <strong>폴더</strong>, PPT, 영상을 모두 준비하면 채점을 시작할 수 있습니다. 폴더 목록은 문서 화면과

              문서 화면과 같이 이 기기에만 저장됩니다.

            </p>

          )}

        </section>



        <p className="evaluate-foot">

          <Link to="/guide">채점 항목 상세 보기</Link>

          <span aria-hidden> · </span>

          <Link to="/">홈으로</Link>

        </p>

      </div>

    </div>

  );

}

