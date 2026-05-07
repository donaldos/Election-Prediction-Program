"use client";

import type { Verdict } from "@/lib/types";

const VERDICT_STYLE: Record<string, { bg: string; text: string; label: string }> = {
  우세: { bg: "bg-red-100", text: "text-red-700", label: "우세" },
  균형: { bg: "bg-yellow-100", text: "text-yellow-700", label: "균형" },
  열세: { bg: "bg-blue-100", text: "text-blue-700", label: "열세" },
};

interface Props {
  verdict: Verdict;
}

export default function VerdictCard({ verdict }: Props) {
  const sorted = [...verdict.candidates].sort(
    (a, b) => b.win_probability - a.win_probability
  );

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-bold text-gray-900">
          {verdict.district_name}
        </h2>
        <span className="text-sm text-gray-500">
          {new Date(verdict.date).toLocaleString("ko-KR")}
        </span>
      </div>

      <div className="space-y-4">
        {sorted.map((c) => {
          const style = VERDICT_STYLE[c.verdict] || VERDICT_STYLE["균형"];
          const pct = Math.round(c.win_probability * 100);

          return (
            <div key={c.candidate} className="space-y-1">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-gray-900">
                    {c.candidate}
                  </span>
                  <span className="text-sm text-gray-500">{c.party}</span>
                  <span
                    className={`text-xs px-2 py-0.5 rounded-full font-medium ${style.bg} ${style.text}`}
                  >
                    {style.label}
                  </span>
                </div>
                <span className="text-lg font-bold text-gray-900">{pct}%</span>
              </div>

              <div className="w-full bg-gray-100 rounded-full h-2.5">
                <div
                  className="h-2.5 rounded-full transition-all duration-500"
                  style={{
                    width: `${pct}%`,
                    backgroundColor:
                      c.verdict === "우세"
                        ? "#ef4444"
                        : c.verdict === "열세"
                          ? "#3b82f6"
                          : "#f59e0b",
                  }}
                />
              </div>

              {typeof c.reasoning === "string" ? (
                <p className="text-sm text-gray-600">{c.reasoning}</p>
              ) : (
                <div className="text-sm space-y-1 mt-1">
                  <p className="text-green-700"><span className="font-medium">장점:</span> {c.reasoning.strengths}</p>
                  <p className="text-red-700"><span className="font-medium">단점:</span> {c.reasoning.weaknesses}</p>
                  <p className="text-blue-700"><span className="font-medium">예측:</span> {c.reasoning.forecast}</p>
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="mt-4 pt-4 border-t border-gray-100">
        <p className="text-sm text-gray-700">{verdict.summary}</p>
        <p className="text-xs text-gray-400 mt-1">
          분석 청크: {verdict.total_chunks_analyzed}건
        </p>
      </div>
    </div>
  );
}
