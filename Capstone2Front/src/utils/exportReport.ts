import { jsPDF } from "jspdf";
import autoTable from "jspdf-autotable";
import * as XLSX from "xlsx";
import type { StoredRubricScores } from "../data/analysisResultStorage";
import { RUBRIC } from "../data/rubric";
import type { SlideVerifyResult } from "../data/slideVerifyStorage";
import { verdictLabelKo } from "../data/slideVerifyStorage";
import type { FolderSubmission } from "../data/folderFilesStorage";
import { submissionPrimaryFileName } from "../data/folderFilesStorage";

// Convert ArrayBuffer to base64 asynchronously using native FileReader to prevent stack overflows
function arrayBufferToBase64(buffer: ArrayBuffer): Promise<string> {
  return new Promise((resolve, reject) => {
    const blob = new Blob([buffer], { type: "application/octet-stream" });
    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target?.result as string;
      if (!dataUrl) {
        reject(new Error("Failed to convert array buffer to base64"));
        return;
      }
      const base64 = dataUrl.split(",")[1];
      resolve(base64);
    };
    reader.onerror = (err) => reject(err);
    reader.readAsDataURL(blob);
  });
}

// Fetch NanumGothic TTF locally from Vite public folder for 100% unblocked reliable loading
async function fetchKoreanFont(): Promise<string | null> {
  const fontUrl = "/NanumGothic.ttf";
  try {
    const response = await fetch(fontUrl);
    if (!response.ok) throw new Error("Failed to fetch local Korean font file");
    const buffer = await response.arrayBuffer();
    return await arrayBufferToBase64(buffer);
  } catch (err) {
    console.error("Font loading error:", err);
    return null;
  }
}

// Format Date string nicely
function formatDate(isoString?: string): string {
  if (!isoString) return "-";
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return isoString;
    return `${d.getFullYear()}년 ${d.getMonth() + 1}월 ${d.getDate()}일 ${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
  } catch {
    return isoString;
  }
}

// Helper to clean up raw markdown text, strip redundant tables/scorecards and format nicely
function cleanMarkdown(text: string | null): string {
  if (!text) return "";
  
  let cleanText = text;
  
  // 1. Strip the scorecard dashboard section at the end (redundant and ugly in exports)
  const scorecardIndicators = [
    "## 📊 AI 초정밀 정량 채점표",
    "📊 AI 초정밀 정량 채점표",
    "AI 초정밀 정량 채점표",
    "5. AI 초정밀 정량 채점표",
    "[CRITICAL] 5."
  ];
  
  for (const indicator of scorecardIndicators) {
    const index = cleanText.indexOf(indicator);
    if (index !== -1) {
      cleanText = cleanText.substring(0, index).trim();
      break;
    }
  }
  
  // 2. Strip any raw markdown tables (lines starting with '|' and containing '|')
  const lines = cleanText.split("\n");
  const filteredLines = lines.filter(line => {
    const trimmed = line.trim();
    if (trimmed.startsWith("|") && trimmed.endsWith("|")) {
      return false;
    }
    if (trimmed.startsWith("|") && (trimmed.includes("---") || trimmed.includes(":::"))) {
      return false;
    }
    return true;
  });
  
  cleanText = filteredLines.join("\n");
  
  // 3. Remove raw markdown formatting symbols cleanly
  cleanText = cleanText
    .replace(/[#*`_~]/g, "") // remove hashes, stars, backticks, underscores, tildes
    .replace(/^[-\s]*-\s+/gm, "• ") // replace markdown bullet points with neat round bullets
    .replace(/\n{3,}/g, "\n\n") // compress excessive newlines to double newlines
    .trim();
    
  return cleanText;
}

// Helper to render overallFeedback as a beautiful Notion-style document in the PDF (headings, dividers, callouts, lists)
function renderNotionStyleText(
  doc: jsPDF,
  rawText: string | null,
  startY: number,
  fontName: string
): number {
  if (!rawText) return startY;
  
  // 1. Strip the scorecard dashboard section at the end (redundant in exports)
  let cleanText = rawText;
  const scorecardIndicators = [
    "## 📊 AI 초정밀 정량 채점표",
    "📊 AI 초정밀 정량 채점표",
    "AI 초정밀 정량 채점표",
    "5. AI 초정밀 정량 채점표",
    "[CRITICAL] 5."
  ];
  for (const indicator of scorecardIndicators) {
    const index = cleanText.indexOf(indicator);
    if (index !== -1) {
      cleanText = cleanText.substring(0, index).trim();
      break;
    }
  }

  // Define page metrics
  const marginX = 20;
  const maxBodyY = 270; // page break trigger
  const contentWidth = 210 - (marginX * 2); // 170mm

  let currentY = startY;

  // Helper to ensure page space and handle automatic page breaks
  const ensureSpace = (neededHeight: number) => {
    if (currentY + neededHeight > maxBodyY) {
      doc.addPage();
      
      // Page number and standard headers will be styled globally at the end
      currentY = 20; // restart at margin top
    }
  };

  // Split lines
  const lines = cleanText.split("\n");
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) {
      currentY += 4; // Paragraph spacer
      continue;
    }

    // Skip raw markdown tables
    if (line.startsWith("|") && (line.endsWith("|") || line.includes("---") || line.includes(":::"))) {
      continue;
    }

    // Check if it's a heading
    const headingMatch = line.match(/^([1-9]\.\s+.*|##+\s+.*|\[CRITICAL\]\s+\d+\..*)/i);
    const isHeading = headingMatch && (line.length < 50 || line.startsWith("##") || line.startsWith("[CRITICAL]"));
    if (isHeading) {
      const headingText = line
        .replace(/[#*`_~]/g, "")
        .replace(/^\[CRITICAL\]/i, "⚠️ [CRITICAL]")
        .trim();

      ensureSpace(15);
      
      // Notion Heading 2 style: Bold, Size 12.5, dark charcoal
      doc.setFont(fontName, "bold");
      doc.setFontSize(12.5);
      doc.setTextColor(31, 41, 55); // Gray 800
      
      currentY += 4;
      doc.text(headingText, marginX, currentY);
      
      // Notion Heading Divider line
      currentY += 2;
      doc.setDrawColor(229, 231, 235); // Gray 200 (Notion style)
      doc.setLineWidth(0.3);
      doc.line(marginX, currentY, marginX + contentWidth, currentY);
      
      currentY += 5; // spacing after heading
      continue;
    }

    // Check if it's a Callout or Quote Block (starts with ">" or is a warning/notice)
    const isQuote = line.startsWith(">") || line.startsWith("[Action") || line.startsWith("⚠️") || line.startsWith("• [Action");
    let cleanedLine = line
      .replace(/[#*`_~]/g, "")
      .replace(/^>\s*/, "") // remove leading >
      .trim();

    if (isQuote) {
      // Split text into wrapped lines for callout
      doc.setFont(fontName, "normal");
      doc.setFontSize(10);
      const wrappedLines = doc.splitTextToSize(cleanedLine, contentWidth - 10);
      const blockHeight = (wrappedLines.length * 5) + 6;

      ensureSpace(blockHeight + 4);

      // Draw light gray Notion callout/quote background box
      doc.setFillColor(249, 250, 251) as any; // Gray 50
      doc.setDrawColor(229, 231, 235); // Gray 200
      doc.roundedRect(marginX, currentY, contentWidth, blockHeight, 1.5, 1.5, "FD");

      // Draw left solid purple border accent (Notion Quote style)
      doc.setFillColor(123, 97, 255) as any; // Purple accent
      doc.rect(marginX, currentY, 1.5, blockHeight, "F");

      // Draw the text inside callout
      doc.setTextColor(55, 65, 81); // Gray 700
      let textY = currentY + 5;
      wrappedLines.forEach((wLine: string) => {
        doc.text(wLine, marginX + 6, textY);
        textY += 5;
      });

      currentY += blockHeight + 4;
      continue;
    }

    // Check if it's a list item (bullet points)
    const isBullet = line.startsWith("-") || line.startsWith("•") || line.startsWith("*");
    if (isBullet) {
      const bulletText = line
        .replace(/^[-*•]\s+/, "")
        .replace(/[#*`_~]/g, "")
        .trim();

      doc.setFont(fontName, "normal");
      doc.setFontSize(10);
      
      // Wrap bullet point text with indent width
      const wrappedBulletLines = doc.splitTextToSize(bulletText, contentWidth - 8);
      const bulletHeight = wrappedBulletLines.length * 5.5;

      ensureSpace(bulletHeight + 2);

      // Draw clean Notion bullet point
      doc.setTextColor(156, 163, 175); // Gray 400
      doc.text("•", marginX + 2, currentY + 3.5);
      
      doc.setTextColor(55, 65, 81); // Gray 700
      let textY = currentY + 3.5;
      wrappedBulletLines.forEach((wLine: string) => {
        doc.text(wLine, marginX + 8, textY);
        textY += 5.5;
      });

      currentY += bulletHeight + 2.5;
      continue;
    }

    // Otherwise, it's a standard paragraph
    doc.setFont(fontName, "normal");
    doc.setFontSize(10.5);
    doc.setTextColor(55, 65, 81); // Gray 700

    const wrappedParaLines = doc.splitTextToSize(cleanedLine, contentWidth);
    const paraHeight = wrappedParaLines.length * 5.5;

    ensureSpace(paraHeight + 2);

    let textY = currentY + 3.5;
    wrappedParaLines.forEach((wLine: string) => {
      doc.text(wLine, marginX, textY);
      textY += 5.5;
    });

    currentY += paraHeight + 3;
  }

  return currentY;
}


export async function exportPDF(
  submissionMeta: FolderSubmission | null,
  scores: StoredRubricScores | null,
  voiceBlock: {
    metrics: Record<string, unknown>;
    scores: Record<string, number>;
  } | null,
  slideVerify: SlideVerifyResult | null,
  overallFeedback: string | null,
  onProgress?: (progress: string) => void
): Promise<void> {
  onProgress?.("한글 폰트 로드 중...");
  const fontBase64 = await fetchKoreanFont();

  if (!fontBase64) {
    throw new Error("한글 폰트를 서버에서 가져오지 못했습니다. 로컬 서버 상태를 확인해 주세요.");
  }

  onProgress?.("PDF 생성 준비 중...");
  const doc = new jsPDF({
    orientation: "portrait",
    unit: "mm",
    format: "a4",
    compress: false
  });

  try {
    doc.addFileToVFS("NanumGothic.ttf", fontBase64);
    doc.addFont("NanumGothic.ttf", "NanumGothic", "normal");
    doc.addFont("NanumGothic.ttf", "NanumGothic", "bold");
    doc.setFont("NanumGothic");
  } catch (e) {
    console.error("Failed to register custom font", e);
    throw new Error("한글 폰트 시스템 가상 등록 중 예외가 발생했습니다.");
  }

  const fontName = "NanumGothic";

  // Header Helper Function
  const addHeader = (title: string) => {
    doc.setFont(fontName, "normal");
    doc.setFontSize(8);
    doc.setTextColor(150, 150, 150);
    doc.text(`Overnight.AI 발표 분석 상세 보고서 - ${title}`, 14, 10);
    doc.text(formatDate(new Date().toISOString()), 196, 10, { align: "right" });
    doc.setDrawColor(230, 230, 230);
    doc.setLineWidth(0.2);
    doc.line(14, 12, 196, 12);
  };

  // Footer Helper Function
  const addFooter = (pageNum: number) => {
    doc.setFont(fontName, "normal");
    doc.setFontSize(8);
    doc.setTextColor(150, 150, 150);
    doc.text(`Page ${pageNum}`, 105, 287, { align: "center" });
  };

  // --- PAGE 1: COVER ---
  doc.setFillColor(123, 97, 255); // Premium Purple (#7b61ff)
  doc.rect(0, 0, 210, 85, "F");

  doc.setTextColor(255, 255, 255);
  doc.setFont(fontName, "normal");
  doc.setFontSize(28);
  doc.text("Overnight.AI", 20, 35);
  
  doc.setFontSize(18);
  doc.text("발표 종합 분석 리포트", 20, 48);

  doc.setFontSize(10);
  doc.setTextColor(220, 220, 255);
  doc.text("인공지능 기반 발표 자세, 발화 지표 및 자료 정렬 분석 결과보고서", 20, 58);

  // Metadata block
  doc.setFillColor(255, 255, 255);
  doc.setDrawColor(220, 220, 220);
  doc.setLineWidth(0.3);
  doc.roundedRect(14, 95, 182, 90, 3, 3, "FD");

  doc.setTextColor(60, 60, 60);
  doc.setFontSize(13);
  doc.text("📋 발표 및 제출 정보", 25, 110);
  doc.setDrawColor(240, 240, 240);
  doc.line(25, 115, 185, 115);

  doc.setFontSize(10);
  let metadataY = 125;
  const metadataItems = [
    { label: "발표자/제출명", value: submissionMeta ? submissionPrimaryFileName(submissionMeta) : "-" },
    { label: "제출 일시", value: submissionMeta ? formatDate(submissionMeta.submittedAt) : "-" },
    { label: "분석 완료일", value: formatDate(new Date().toISOString()) },
    { label: "슬라이드 상태", value: slideVerify ? `${verdictLabelKo(slideVerify.verdict)} (종합 일치율 ${slideVerify.overall_match_percent}%)` : "분석 미제공" }
  ];

  metadataItems.forEach(item => {
    doc.setTextColor(110, 110, 110);
    doc.text(item.label, 28, metadataY);
    doc.setTextColor(50, 50, 50);
    doc.text(String(item.value), 65, metadataY);
    metadataY += 12;
  });

  // Big score badge on cover
  let totalScore = 0;
  if (scores) {
    totalScore = RUBRIC.reduce((sum, cat) => sum + (scores[cat.id]?.category ?? 0), 0);
  }

  doc.setFillColor(245, 243, 255);
  doc.roundedRect(14, 200, 182, 60, 3, 3, "F");

  doc.setTextColor(123, 97, 255);
  doc.setFontSize(12);
  doc.text("✨ 발표 최종 종합 평가 점수", 25, 215);

  doc.setFontSize(42);
  const scoreWidth = doc.getTextWidth(String(totalScore));
  doc.text(String(totalScore), 25, 247);
  doc.setFontSize(16);
  doc.setTextColor(150, 150, 150);
  doc.text("/ 100 점", 25 + scoreWidth + 4, 247);

  doc.setFontSize(9);
  doc.setTextColor(100, 100, 100);
  doc.text("* 본 평가는 자료 시각화(50점), 전달 음성(30점), 시각 비언어 자세(20점)를 종합 정량한 지표입니다.", 25, 254);

  addFooter(1);

  // --- PAGE 2: DETAILED RUBRIC SCORES ---
  doc.addPage();
  addHeader("상세 평가 채점표");

  doc.setTextColor(40, 40, 40);
  doc.setFontSize(14);
  doc.text("📊 영역별 세부 항목 채점 내역", 14, 22);

  const tableBody: any[] = [];
  
  RUBRIC.forEach(cat => {
    const catScore = scores?.[cat.id];
    
    tableBody.push([
      { content: cat.title, colSpan: 3, styles: { fillColor: [240, 240, 255], fontStyle: "bold", textColor: [123, 97, 255], font: fontName } },
      { content: `${catScore?.category ?? 0} / ${cat.maxScore}점`, styles: { fillColor: [240, 240, 255], fontStyle: "bold", textColor: [123, 97, 255], halign: "right", font: fontName } }
    ]);

    cat.items.forEach((item, idx) => {
      const scoreVal = catScore?.items[idx] ?? 0;
      const maxVal = cat.itemMaxes[idx];
      const rate = maxVal > 0 ? Math.round((scoreVal / maxVal) * 100) : 0;
      tableBody.push([
        "",
        item.split(" (")[0],
        `${scoreVal} / ${maxVal}점`,
        `${rate}%`
      ]);
    });
  });

  tableBody.push([
    { content: "총점 합계 (Total Score)", colSpan: 2, styles: { fillColor: [123, 97, 255], textColor: [255, 255, 255], fontStyle: "bold", font: fontName } },
    { content: `${totalScore} / 100점`, colSpan: 2, styles: { fillColor: [123, 97, 255], textColor: [255, 255, 255], fontStyle: "bold", halign: "right", font: fontName } }
  ]);

  autoTable(doc, {
    startY: 28,
    head: [["영역", "평가 항목", "취득 점수", "달성률"]],
    body: tableBody,
    theme: "striped",
    styles: {
      font: fontName,
      fontSize: 9,
      cellPadding: 4,
      valign: "middle"
    },
    headStyles: {
      fillColor: [100, 100, 100],
      textColor: [255, 255, 255],
      font: fontName
    },
    columnStyles: {
      0: { cellWidth: 50 },
      1: { cellWidth: 80 },
      2: { cellWidth: 32, halign: "center" },
      3: { cellWidth: 20, halign: "center" }
    }
  });

  const lastY = (doc as any).lastAutoTable.finalY || 150;
  doc.setFillColor(250, 250, 250);
  doc.setDrawColor(230, 230, 230);
  doc.roundedRect(14, lastY + 10, 182, 35, 2, 2, "FD");

  doc.setTextColor(80, 80, 80);
  doc.setFontSize(10);
  doc.text("💡 평가 영역 설명:", 20, lastY + 17);
  doc.setFontSize(8.5);
  doc.setTextColor(110, 110, 110);
  doc.text("- 내용 및 시각화: 대조된 PPT 문서의 완성도, 텍스트 가독성, 전달하는 데이터 신뢰성 대조 점수입니다.", 20, lastY + 23);
  doc.text("- 전달의 안정성: 오디오 및 프라트 기반의 발화 안정 지수, 불필요한 필러워드 비중 제어 점수입니다.", 20, lastY + 28);
  doc.text("- 시각적 비언어: 영상 분석을 기반으로 제스처, 풍부한 미소 표정, 아이컨택 시선 유지 긍정 지수입니다.", 20, lastY + 33);

  addFooter(2);

  // --- PAGE 3: VOICE & SLIDE VERIFICATION DETAIL ---
  doc.addPage();
  addHeader("음성 및 슬라이드 상세 분석");

  doc.setTextColor(40, 40, 40);
  doc.setFontSize(13);
  doc.text("🎙️ 목소리 및 언어적 분석 지표 (Voice Quant)", 14, 22);

  const voiceMetrics = voiceBlock?.metrics || {};
  const voiceBody = [
    ["발화 중 말버릇 (필러워드 총합)", `${voiceMetrics.filler_total ?? 0}회 (분당 ${Number(voiceMetrics.fillers_per_minute ?? 0).toFixed(1)}회)`],
    ["반복 구절 발생 건수", `${voiceMetrics.repeated_phrase_hits ?? 0}건`],
    ["의도치 않은 긴 무음 (정적)", `${voiceMetrics.silence_pause_count ?? 0}회 (총 ${Number(voiceMetrics.silence_total_sec ?? 0).toFixed(1)}초)`],
    ["평균 말하는 속도 (Cps)", `${Number(voiceMetrics.avg_speech_rate_cps ?? 0).toFixed(2)} 글자/초`]
  ];

  autoTable(doc, {
    startY: 26,
    head: [["지표 항목", "측정 수치 및 상세 분석"]],
    body: voiceBody,
    theme: "grid",
    styles: { font: fontName, fontSize: 9, cellPadding: 4 },
    headStyles: { fillColor: [80, 80, 100], textColor: [255, 255, 255], font: fontName },
    columnStyles: {
      0: { cellWidth: 70 },
      1: { cellWidth: 112 }
    }
  });

  const voiceY = (doc as any).lastAutoTable.finalY || 80;

  doc.setTextColor(40, 40, 40);
  doc.setFontSize(13);
  doc.text("🖥️ PPT 슬라이드 일치 및 영상 분석 (Visual Match)", 14, voiceY + 12);

  const slideBody = [
    ["종합 일치율 (Overall Match)", slideVerify ? `${slideVerify.overall_match_percent}%` : "기록 없음"],
    ["화면 시각 유사도 (Visual Similarity)", slideVerify ? `${slideVerify.visual_match_percent}%` : "기록 없음"],
    ["슬라이드 페이지 커버리지", slideVerify ? `${slideVerify.slide_coverage_percent}%` : "기록 없음"],
    ["촬영/렌더링 모드 구분", slideVerify ? `${slideVerify.video_type === "audience" ? "강의실 스크린 감지" : "직접 렌더링"} (${slideVerify.render_mode ?? "Normal"})` : "기록 없음"],
    ["영상 내 슬라이드 장표 진행 순서", slideVerify?.detected_slide_sequence?.length ? slideVerify.detected_slide_sequence.join(" → ") + " 장" : "-"]
  ];

  autoTable(doc, {
    startY: voiceY + 16,
    head: [["검증 항목", "검사 및 매칭 스코어"]],
    body: slideBody,
    theme: "grid",
    styles: { font: fontName, fontSize: 9, cellPadding: 4 },
    headStyles: { fillColor: [80, 100, 80], textColor: [255, 255, 255], font: fontName },
    columnStyles: {
      0: { cellWidth: 70 },
      1: { cellWidth: 112 }
    }
  });

  const slideY = (doc as any).lastAutoTable.finalY || 160;

  const issues = slideVerify?.issues || [];
  if (issues.length > 0) {
    doc.setFontSize(11);
    doc.setTextColor(200, 50, 50);
    doc.text("⚠️ 슬라이드 불일치 감지 리스트", 14, slideY + 12);

    const issuesBody = issues.slice(0, 5).map(issue => [
      `${issue.time_sec}초`,
      `${issue.ppt_slide_index}번째 슬라이드`,
      issue.message,
      `${Math.round(issue.confidence * 100)}%`
    ]);

    autoTable(doc, {
      startY: slideY + 16,
      head: [["검출 시간", "대상 장표", "이슈 탐지 내용", "신뢰도"]],
      body: issuesBody,
      theme: "striped",
      styles: { font: fontName, fontSize: 8.5, cellPadding: 3.5 },
      headStyles: { fillColor: [200, 80, 80], textColor: [255, 255, 255], font: fontName },
      columnStyles: {
        0: { cellWidth: 25, halign: "center" },
        1: { cellWidth: 35, halign: "center" },
        2: { cellWidth: 97 },
        3: { cellWidth: 25, halign: "center" }
      }
    });
  }

  addFooter(3);

  // --- PAGE 4: AI EXPERT FEEDBACK ---
  doc.addPage();
  addHeader("AI 심층 전문가 피드백");

  doc.setTextColor(40, 40, 40);
  doc.setFontSize(13);
  doc.text("🤖 AI 인공지능 종합 심층 피드백", 14, 22);

  // Render feedback using a beautiful Notion-style document layout!
  renderNotionStyleText(doc, overallFeedback, 26, fontName);

  const pageCount = (doc as any).internal.getNumberOfPages();
  for (let i = 4; i <= pageCount; i++) {
    doc.setPage(i);
    addHeader("AI 심층 전문가 피드백 (계속)");
    addFooter(i);
  }

  onProgress?.("PDF 파일 다운로드 중...");
  doc.save(`Overnight_AI_발표_리포트_${submissionMeta ? submissionMeta.id.slice(0, 6) : "analysis"}.pdf`);
  onProgress?.("");
}

export function exportExcel(
  submissionMeta: FolderSubmission | null,
  scores: StoredRubricScores | null,
  voiceBlock: {
    metrics: Record<string, unknown>;
    scores: Record<string, number>;
  } | null,
  slideVerify: SlideVerifyResult | null,
  overallFeedback: string | null
): void {
  const wb = XLSX.utils.book_new();
  const cleanFeedback = cleanMarkdown(overallFeedback);

  let totalScore = 0;
  if (scores) {
    totalScore = RUBRIC.reduce((sum, cat) => sum + (scores[cat.id]?.category ?? 0), 0);
  }

  // ==========================================
  // --- SHEET 0: 종합 보고서 (통합 단일 시트) ---
  // ==========================================
  const sheetUnifiedData: any[] = [
    ["📢 Overnight.AI 발표 분석 종합 대시보드 보고서 (통합)"],
    [],
    ["[1. 발표 기본 제출 정보]"],
    ["발표 자료 및 영상명", submissionMeta ? submissionPrimaryFileName(submissionMeta) : "-"],
    ["제출 및 분석 시각", submissionMeta ? formatDate(submissionMeta.submittedAt) : "-"],
    ["슬라이드 종합 매칭 판정", slideVerify ? `${verdictLabelKo(slideVerify.verdict)} (종합 일치율 ${slideVerify.overall_match_percent}%)` : "분석 기록 없음"],
    ["최종 종합 발표 스코어", `${totalScore} / 100 점`],
    [],
    ["[2. 영역별 정량 평가 요약]"],
    ["평가 부문 (Category)", "취득 점수", "배점 만점", "달성 비율 (%)"]
  ];

  RUBRIC.forEach(cat => {
    const catScore = scores?.[cat.id];
    const scoreVal = catScore?.category ?? 0;
    const maxVal = cat.maxScore;
    const rate = maxVal > 0 ? `${Math.round((scoreVal / maxVal) * 100)}%` : "0%";
    sheetUnifiedData.push([cat.title, String(scoreVal), String(maxVal), rate]);
  });
  sheetUnifiedData.push(["최종 종합 합계", String(totalScore), "100점 만점", `${totalScore}%`]);

  sheetUnifiedData.push(
    [],
    ["[3. 루브릭 평가 항목별 세부 채점 내역]"],
    ["평가 부문", "평가 대상 세부 요소", "취득 점수", "배점 만점", "항목 달성비율 (%)"]
  );

  RUBRIC.forEach(cat => {
    const catScore = scores?.[cat.id];
    cat.items.forEach((item, idx) => {
      const scoreVal = catScore?.items[idx] ?? 0;
      const maxVal = cat.itemMaxes[idx];
      const rate = maxVal > 0 ? `${Math.round((scoreVal / maxVal) * 100)}%` : "0%";
      sheetUnifiedData.push([
        cat.title.split(" (")[0],
        item.split(" (")[0],
        String(scoreVal),
        String(maxVal),
        rate
      ]);
    });
  });

  const voiceMetrics = voiceBlock?.metrics || {};
  sheetUnifiedData.push(
    [],
    ["[4. 목소리 및 발화 분석 정량 지표]"],
    ["지표 측정 요소", "정량 측정 수치", "상세 단위"],
    ["발화 중 말버릇 (필러워드 총합)", String(voiceMetrics.filler_total ?? 0), "회"],
    ["분당 필러워드 사용 횟수", String(Number(voiceMetrics.fillers_per_minute ?? 0).toFixed(2)), "회 / 분"],
    ["반복 구절 발생 빈도", String(voiceMetrics.repeated_phrase_hits ?? 0), "건"],
    ["의도치 않은 무음(정적) 횟수", String(voiceMetrics.silence_pause_count ?? 0), "회"],
    ["무음 누적 시간", String(Number(voiceMetrics.silence_total_sec ?? 0).toFixed(1)), "초"],
    ["평균 말 속도", String(Number(voiceMetrics.avg_speech_rate_cps ?? 0).toFixed(2)), "글자 / 초"]
  );

  sheetUnifiedData.push(
    [],
    ["[5. PPT 슬라이드 매칭 분석 상세]"],
    ["검증 분석 지표", "결과 수치", "단위"]
  );
  sheetUnifiedData.push(
    ["종합 슬라이드 일치율", slideVerify ? String(slideVerify.overall_match_percent) : "-", "%"],
    ["화면 시각 유사성", slideVerify ? String(slideVerify.visual_match_percent) : "-", "%"],
    ["장표 페이지 커버리지", slideVerify ? String(slideVerify.slide_coverage_percent) : "-", "%"],
    ["영상 내 슬라이드 진행 장표 순서", slideVerify?.detected_slide_sequence?.length ? slideVerify.detected_slide_sequence.join(" -> ") : "-", "장표 순서"]
  );

  const issues = slideVerify?.issues || [];
  if (issues.length > 0) {
    sheetUnifiedData.push(
      [],
      ["[5-1. 슬라이드 불일치 실시간 이슈 로그]"],
      ["탐지 시간", "대상 장표 번호", "불일치 사유 내용", "매칭 신뢰도"]
    );
    issues.forEach(issue => {
      sheetUnifiedData.push([
        `${issue.time_sec}초`,
        `${issue.ppt_slide_index}번째 장표`,
        issue.message,
        `${Math.round(issue.confidence * 100)}%`
      ]);
    });
  }

  sheetUnifiedData.push(
    [],
    ["[6. AI 전문가 인공지능 종합 심층 피드백]"],
    [cleanFeedback]
  );

  const wsUnified = XLSX.utils.aoa_to_sheet(sheetUnifiedData);
  // Set gorgeous wide auto-fit column widths so that it renders beautifully on one screen!
  wsUnified["!cols"] = [
    { wch: 32 }, // Col A (Category / Metaname) - Widened
    { wch: 48 }, // Col B (Item / Metavalue) - Widened
    { wch: 15 }, // Col C (Score)
    { wch: 15 }, // Col D (Max Score)
    { wch: 22 }  // Col E (Rate %)
  ];
  XLSX.utils.book_append_sheet(wb, wsUnified, "발표 종합 분석 보고서");

  // ==========================================
  // --- SHEET 1: 종합 성적표 (Summary Scores) ---
  // ==========================================
  const sheet1Data = [
    ["Overnight.AI 발표 분석 종합 성적표"],
    [],
    ["[제출 정보]"],
    ["발표 자료 및 영상명", submissionMeta ? submissionPrimaryFileName(submissionMeta) : "-"],
    ["제출 및 분석 시각", submissionMeta ? formatDate(submissionMeta.submittedAt) : "-"],
    ["종합 결과 평정", slideVerify ? verdictLabelKo(slideVerify.verdict) : "기록 없음"],
    [],
    ["[평가 영역별 취득 점수]"],
    ["평가 부문 (Category)", "취득 점수", "배점 만점", "달성비율 (%)"]
  ];

  RUBRIC.forEach(cat => {
    const catScore = scores?.[cat.id];
    const scoreVal = catScore?.category ?? 0;
    const maxVal = cat.maxScore;
    const rate = maxVal > 0 ? `${Math.round((scoreVal / maxVal) * 100)}%` : "0%";
    sheet1Data.push([cat.title, String(scoreVal), String(maxVal), rate]);
  });
  sheet1Data.push(["최종 종합 점수 (Total Score)", String(totalScore), "100", `${totalScore}%`]);

  const ws1 = XLSX.utils.aoa_to_sheet(sheet1Data);
  ws1["!cols"] = [
    { wch: 32 },
    { wch: 32 },
    { wch: 15 },
    { wch: 18 }
  ];
  XLSX.utils.book_append_sheet(wb, ws1, "종합 성적표");

  // ==========================================
  // --- SHEET 2: 세부 항목 채점표 (Detailed Rubric) ---
  // ==========================================
  const sheet2Data = [
    ["평가 영역 (Category)", "세부 평가 요소 (Item)", "취득 점수", "배점 만점", "달성률 (%)"]
  ];

  RUBRIC.forEach(cat => {
    const catScore = scores?.[cat.id];
    cat.items.forEach((item, idx) => {
      const scoreVal = catScore?.items[idx] ?? 0;
      const maxVal = cat.itemMaxes[idx];
      const rate = maxVal > 0 ? `${Math.round((scoreVal / maxVal) * 100)}%` : "0%";
      sheet2Data.push([
        cat.title.split(" (")[0],
        item.split(" (")[0],
        String(scoreVal),
        String(maxVal),
        rate
      ]);
    });
  });

  const ws2 = XLSX.utils.aoa_to_sheet(sheet2Data);
  ws2["!cols"] = [
    { wch: 28 },
    { wch: 40 },
    { wch: 15 },
    { wch: 15 },
    { wch: 18 }
  ];
  XLSX.utils.book_append_sheet(wb, ws2, "세부 채점 내역");

  // ==========================================
  // --- SHEET 3: 음성 및 슬라이드 분석 (Voice & Slide Analysis) ---
  // ==========================================
  const sheet3Data = [
    ["Overnight.AI 음성 및 슬라이드 정량 지표 리포트"],
    [],
    ["[1. 목소리 및 발화 분석 지표]"],
    ["지표 요소", "측정 수치", "상세 단위"],
    ["말버릇(필러워드) 총합", String(voiceBlock?.metrics?.filler_total ?? 0), "회"],
    ["분당 필러워드 사용 횟수", String(Number(voiceBlock?.metrics?.fillers_per_minute ?? 0).toFixed(2)), "회 / 분"],
    ["반복 구절 발생 빈도", String(voiceBlock?.metrics?.repeated_phrase_hits ?? 0), "건"],
    ["의도치 않은 무음(정적) 횟수", String(voiceBlock?.metrics?.silence_pause_count ?? 0), "회"],
    ["무음 누적 시간", String(Number(voiceBlock?.metrics?.silence_total_sec ?? 0).toFixed(1)), "초"],
    ["평균 말 속도", String(Number(voiceBlock?.metrics?.avg_speech_rate_cps ?? 0).toFixed(2)), "글자/초"],
    [],
    ["[2. PPT 슬라이드 매칭 분석]"],
    ["검증 지표", "결과 수치"],
    ["종합 슬라이드 일치도", slideVerify ? `${slideVerify.overall_match_percent}%` : "-"],
    ["화면 시각 유사성", slideVerify ? `${slideVerify.visual_match_percent}%` : "-"],
    ["장표 페이지 커버리지", slideVerify ? `${slideVerify.slide_coverage_percent}%` : "-"],
    ["슬라이드 진행 순서", slideVerify?.detected_slide_sequence?.length ? slideVerify.detected_slide_sequence.join(" -> ") : "-"]
  ];

  if (issues.length > 0) {
    sheet3Data.push([], ["[3. 슬라이드 불일치 이슈 로그]"], ["검출 시간", "슬라이드 인덱스", "이슈 탐지 내용", "신뢰도"]);
    issues.forEach(issue => {
      sheet3Data.push([
        `${issue.time_sec}초`,
        `${issue.ppt_slide_index}번째 장표`,
        issue.message,
        `${Math.round(issue.confidence * 100)}%`
      ]);
    });
  }

  const ws3 = XLSX.utils.aoa_to_sheet(sheet3Data);
  ws3["!cols"] = [
    { wch: 35 },
    { wch: 35 },
    { wch: 20 },
    { wch: 15 }
  ];
  XLSX.utils.book_append_sheet(wb, ws3, "음성 및 PPT 분석");

  // ==========================================
  // --- SHEET 4: AI 전문가 심층 피드백 (AI Expert Feedback) ---
  // ==========================================
  const sheet4Data = [
    ["🤖 AI 인공지능 종합 심층 피드백 리포트"],
    [],
    [cleanFeedback]
  ];

  const ws4 = XLSX.utils.aoa_to_sheet(sheet4Data);
  ws4["!cols"] = [
    { wch: 120 }
  ];
  XLSX.utils.book_append_sheet(wb, ws4, "AI 심층 피드백");

  // Trigger Excel file download
  XLSX.writeFile(wb, `Overnight_AI_발표_분석_엑셀_${submissionMeta ? submissionMeta.id.slice(0, 6) : "report"}.xlsx`);
}
