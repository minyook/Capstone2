import { useMemo } from "react";
import { Link } from "react-router-dom";
import { useFolders } from "../context/FoldersContext";
import { formatFolderDate } from "../data/folderStorage";
import { IconArrowLeft } from "../components/Icons";
import "./Trash.css";

export function Trash() {
  const { trashFolders, restoreFolder, purgeFolder } = useFolders();

  const sorted = useMemo(() => {
    const copy = [...trashFolders];
    copy.sort((a, b) => new Date(b.deletedAt).getTime() - new Date(a.deletedAt).getTime());
    return copy;
  }, [trashFolders]);

  return (
    <div className="page trash-page">
      <div className="page-inner page-inner--wide">
        <Link to="/projects" className="trash-back" aria-label="문서로 돌아가기">
          <IconArrowLeft />
        </Link>
        <h1 className="trash-title">휴지통</h1>
        <p className="trash-sub">
          문서에서 삭제한 폴더가 여기에 보관됩니다. 복구하면 다시 목록에 나타나고, 영구 삭제하면 이 기기에서 완전히
          지워집니다.
        </p>

        {sorted.length === 0 ? (
          <div className="trash-empty">
            <span className="trash-empty__icon" aria-hidden>
              🗑️
            </span>
            <p className="trash-empty__main">휴지통이 비어 있습니다</p>
            <p className="trash-empty__sub">
              문서 화면에서 폴더를 삭제하면 이곳으로 이동합니다.
            </p>
          </div>
        ) : (
          <ul className="trash-list">
            {sorted.map((f) => (
              <li key={f.id} className="trash-row">
                <div className="trash-row__main">
                  <span className="trash-row__name">{f.name}</span>
                  <span className="trash-row__meta">
                    만들어짐 {formatFolderDate(f.createdAt)} · 삭제 {formatFolderDate(f.deletedAt)}
                  </span>
                </div>
                <div className="trash-row__actions">
                  <button type="button" className="trash-btn trash-btn--primary" onClick={() => restoreFolder(f.id)}>
                    복구
                  </button>
                  <button
                    type="button"
                    className="trash-btn trash-btn--danger"
                    onClick={() => {
                      if (window.confirm(`「${f.name}」을(를) 영구 삭제할까요? 이 작업은 되돌릴 수 없습니다.`)) {
                        purgeFolder(f.id);
                      }
                    }}
                  >
                    영구 삭제
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
