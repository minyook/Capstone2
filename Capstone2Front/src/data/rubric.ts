/** 발표 자동 채점 — 팀 기준 (PPT·발표 비교 멀티모달 평가) */

export type RubricCategoryId = "content" | "voice" | "attitude";

export interface RubricCategory {
  id: RubricCategoryId;
  title: string;
  subtitle: string;
  items: string[];
  /** UI용 짧은 설명 */
  summary: string;
  /** 카테고리별 만점 */
  maxScore: number;
  /** 각 세부 항목별 만점 배열 */
  itemMaxes: number[];
}

export const RUBRIC: RubricCategory[] = [
  {
    id: "content",
    title: "내용 및 시각화 (Content)",
    subtitle: "PPT 일치율 및 내용 구성의 완성도",
    summary: "업로드한 PPT와 실제 발표를 비교해 내용 일치도, 가독성, 데이터 신뢰성을 종합 평가합니다.",
    maxScore: 50,
    itemMaxes: [15, 10, 15, 10],
    items: [
      "논리 구조 (15점)",
      "헤드라인 전략 (10점)",
      "시각적 가독성 (15점)",
      "데이터 신뢰성 (10점)",
    ],
  },
  {
    id: "voice",
    title: "전달의 안정성 (Voice)",
    subtitle: "음성 안정성, 신체 평정심, 언어 유창성",
    summary: "오디오와 비전 분석으로 목소리 떨림, 신체 평정심 및 언어적 유창성을 평가합니다.",
    maxScore: 30,
    itemMaxes: [10, 10, 10],
    items: [
      "음성 안정도 (10점)",
      "신체 평정심 (10점)",
      "언어적 유창성 (10점)",
    ],
  },
  {
    id: "attitude",
    title: "시각적 비언어 (Attitude)",
    subtitle: "시선 처리 및 표정/제스처 긍정 척도",
    summary: "영상 분석으로 발표자의 시선 응시 비율과 긍정적인 비언어적 커뮤니케이션을 평가합니다.",
    maxScore: 20,
    itemMaxes: [10, 10],
    items: [
      "시선 처리 (10점)",
      "제스처/표정 (10점)",
    ],
  },
];

export function totalRubricItems(): number {
  return RUBRIC.reduce((n, c) => n + c.items.length, 0);
}
