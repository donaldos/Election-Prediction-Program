"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import type { QueryResponse } from "@/lib/types";

export default function QueryBox() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await api.query(query.trim());
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "질의 실패");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <h2 className="text-lg font-bold text-gray-900 mb-4">
        자유 질의
      </h2>

      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="예: 조국의 평택을 지지율 변화는?"
          className="flex-1 border border-gray-300 rounded-lg px-4 py-2 text-sm text-black focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          disabled={loading}
        />
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
        >
          {loading ? "분석 중..." : "질의"}
        </button>
      </form>

      {error && (
        <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-3 text-red-700 text-sm">
          {error}
        </div>
      )}

      {loading && (
        <div className="mt-4 text-center py-8 text-gray-500 text-sm">
          VectorDB 검색 + LLM 분석 중... (20~40초 소요)
        </div>
      )}

      {result && (
        <div className="mt-4 space-y-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm font-medium text-gray-500 mb-2">답변</p>
            <div className="text-sm text-gray-900 whitespace-pre-wrap leading-relaxed">
              {result.answer}
            </div>
          </div>

          {result.sources.length > 0 && (
            <div>
              <p className="text-sm font-medium text-gray-500 mb-2">
                참고 뉴스 ({result.chunk_count}건)
              </p>
              <div className="space-y-2">
                {result.sources.map((s, i) => (
                  <div
                    key={i}
                    className="border border-gray-100 rounded-lg p-3 text-xs"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-medium text-gray-900">
                        {s.title}
                      </span>
                      <span className="text-gray-400 ml-2 shrink-0">
                        {s.score.toFixed(3)}
                      </span>
                    </div>
                    <div className="text-gray-500">
                      {s.source}
                      {s.published_at &&
                        ` · ${new Date(s.published_at).toLocaleDateString("ko-KR")}`}
                    </div>
                    <p className="text-gray-600 mt-1">{s.text_preview}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
