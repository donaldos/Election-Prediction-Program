"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { District, PipelineStatus, PollEntry, RAGConfig, VectorDBStats } from "@/lib/types";

interface PollRow {
  candidate: string;
  party: string;
  support: string;
}

interface PollFormState {
  district_id: string;
  pollster: string;
  survey_date: string;
  rows: PollRow[];
}

export default function AdminPage() {
  const [stats, setStats] = useState<VectorDBStats | null>(null);
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus | null>(null);
  const [ragConfig, setRagConfig] = useState<RAGConfig | null>(null);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const [lookbackDays, setLookbackDays] = useState("");
  const [topK, setTopK] = useState("");
  const [minScore, setMinScore] = useState("");
  const [purgeDays, setPurgeDays] = useState("");
  const [scraperType, setScraperType] = useState("all");
  const [scraperDays, setScraperDays] = useState("");
  const [purgeInput, setPurgeInput] = useState("60");

  const [districts, setDistricts] = useState<District[]>([]);
  const [pollHistory, setPollHistory] = useState<PollEntry[]>([]);
  const [pollForm, setPollForm] = useState<PollFormState>({
    district_id: "",
    pollster: "",
    survey_date: new Date().toISOString().slice(0, 10),
    rows: [],
  });

  const refresh = async () => {
    try {
      const [s, p, r] = await Promise.all([
        api.getVectorDBStats(),
        api.getPipelineStatus(),
        api.getRAGConfig(),
      ]);
      setStats(s);
      setPipelineStatus(p);
      setRagConfig(r);
      setLookbackDays(String(r.retriever.lookback_days ?? ""));
      setTopK(String(r.retriever.top_k ?? ""));
      setMinScore(String(r.reranker.min_score ?? ""));
      setPurgeDays(String(r.purge_days ?? ""));
    } catch (e) {
      setMessage(`오류: ${(e as Error).message}`);
    }
  };

  const refreshPolls = async () => {
    try {
      const [d, polls] = await Promise.all([
        api.getDistricts(),
        api.getPolls(),
      ]);
      setDistricts(d);
      setPollHistory(polls.entries);
      if (d.length > 0 && !pollForm.district_id) {
        const first = d[0];
        setPollForm((prev) => ({
          ...prev,
          district_id: first.id,
          rows: first.candidates.map((c) => ({
            candidate: c.name,
            party: c.party,
            support: "",
          })),
        }));
      }
    } catch (e) {
      setMessage(`오류: ${(e as Error).message}`);
    }
  };

  useEffect(() => {
    refresh();
    refreshPolls();
  }, []);

  const showMessage = (msg: string) => {
    setMessage(msg);
    setTimeout(() => setMessage(""), 5000);
  };

  const handleRunPipeline = async () => {
    setLoading(true);
    try {
      const res = await api.runPipeline(
        scraperType,
        scraperDays ? parseInt(scraperDays) : undefined
      );
      showMessage(res.message);
    } catch (e) {
      showMessage(`오류: ${(e as Error).message}`);
    }
    setLoading(false);
  };

  const handleRebuild = async () => {
    if (!confirm("VectorDB를 삭제하고 전체 파이프라인을 재실행합니다. 계속하시겠습니까?")) return;
    setLoading(true);
    try {
      const res = await api.rebuildVectorDB();
      showMessage(res.message);
    } catch (e) {
      showMessage(`오류: ${(e as Error).message}`);
    }
    setLoading(false);
  };

  const handlePurge = async () => {
    setLoading(true);
    try {
      const res = await api.purgeVectorDB(parseInt(purgeInput));
      showMessage(res.message);
      refresh();
    } catch (e) {
      showMessage(`오류: ${(e as Error).message}`);
    }
    setLoading(false);
  };

  const handleSaveRAG = async () => {
    setLoading(true);
    try {
      const data: Record<string, unknown> = {};
      if (lookbackDays) data.lookback_days = parseInt(lookbackDays);
      if (topK) data.top_k = parseInt(topK);
      if (minScore) data.min_score = parseFloat(minScore);
      if (purgeDays) data.purge_days = parseInt(purgeDays);

      await api.updateRAGConfig(data);
      showMessage("RAG 설정이 저장되었습니다.");
      refresh();
    } catch (e) {
      showMessage(`오류: ${(e as Error).message}`);
    }
    setLoading(false);
  };

  const handleDistrictChange = (districtId: string) => {
    const district = districts.find((d) => d.id === districtId);
    if (!district) return;
    setPollForm((prev) => ({
      ...prev,
      district_id: districtId,
      rows: district.candidates.map((c) => ({
        candidate: c.name,
        party: c.party,
        support: "",
      })),
    }));
  };

  const handlePollSupportChange = (index: number, value: string) => {
    setPollForm((prev) => {
      const rows = [...prev.rows];
      rows[index] = { ...rows[index], support: value };
      return { ...prev, rows };
    });
  };

  const handleSavePolls = async () => {
    const entries = pollForm.rows
      .filter((r) => r.support !== "")
      .map((r) => ({
        district_id: pollForm.district_id,
        candidate: r.candidate,
        party: r.party,
        support: parseFloat(r.support),
        pollster: pollForm.pollster,
        survey_date: pollForm.survey_date,
      }));

    if (entries.length === 0) {
      showMessage("지지율을 1명 이상 입력하세요.");
      return;
    }
    if (!pollForm.pollster.trim()) {
      showMessage("조사기관을 입력하세요.");
      return;
    }

    setLoading(true);
    try {
      await api.savePolls(entries);
      showMessage(`여론조사 ${entries.length}건 저장 완료. 다음 판정 시 반영됩니다.`);
      refreshPolls();
    } catch (e) {
      showMessage(`오류: ${(e as Error).message}`);
    }
    setLoading(false);
  };

  const handleDeletePoll = async (entryId: string) => {
    setLoading(true);
    try {
      await api.deletePoll(entryId);
      refreshPolls();
    } catch (e) {
      showMessage(`오류: ${(e as Error).message}`);
    }
    setLoading(false);
  };

  const filteredHistory = pollHistory.filter(
    (e) => !pollForm.district_id || e.district_id === pollForm.district_id
  );

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-xl font-bold text-gray-900">관리자</h1>
          <a
            href="/"
            className="text-sm text-gray-500 hover:text-gray-900 transition-colors"
          >
            대시보드
          </a>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-8 space-y-6">
        {message && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-blue-700 text-sm">
            {message}
          </div>
        )}

        {/* 여론조사 입력 */}
        <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-bold text-gray-900 mb-4">여론조사 입력</h2>

          <div className="grid grid-cols-3 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-1">선거구</label>
              <select
                value={pollForm.district_id}
                onChange={(e) => handleDistrictChange(e.target.value)}
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm text-gray-900 w-full"
              >
                {districts.map((d) => (
                  <option key={d.id} value={d.id}>{d.name}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-1">조사기관</label>
              <input
                type="text"
                value={pollForm.pollster}
                onChange={(e) => setPollForm((prev) => ({ ...prev, pollster: e.target.value }))}
                placeholder="예: 한국갤럽"
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm text-gray-900 w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-1">조사일</label>
              <input
                type="date"
                value={pollForm.survey_date}
                onChange={(e) => setPollForm((prev) => ({ ...prev, survey_date: e.target.value }))}
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm text-gray-900 w-full"
              />
            </div>
          </div>

          {/* 후보별 지지율 테이블 */}
          <div className="border border-gray-200 rounded-lg overflow-hidden mb-4">
            <table className="w-full text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="text-left px-4 py-2 font-medium text-gray-900">후보</th>
                  <th className="text-left px-4 py-2 font-medium text-gray-900">정당</th>
                  <th className="text-left px-4 py-2 font-medium text-gray-900">지지율 (%)</th>
                </tr>
              </thead>
              <tbody>
                {pollForm.rows.map((row, i) => (
                  <tr key={row.candidate} className="border-t border-gray-100">
                    <td className="px-4 py-2 text-gray-900">{row.candidate}</td>
                    <td className="px-4 py-2 text-gray-600">{row.party}</td>
                    <td className="px-4 py-2">
                      <input
                        type="number"
                        step="0.1"
                        min="0"
                        max="100"
                        value={row.support}
                        onChange={(e) => handlePollSupportChange(i, e.target.value)}
                        placeholder="0.0"
                        className="border border-gray-300 rounded px-2 py-1 text-sm text-gray-900 w-24"
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <button
            onClick={handleSavePolls}
            disabled={loading}
            className="bg-indigo-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-indigo-700 disabled:opacity-50"
          >
            적용
          </button>
        </section>

        {/* 여론조사 이력 */}
        <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-bold text-gray-900 mb-4">여론조사 이력</h2>
          {filteredHistory.length === 0 ? (
            <p className="text-sm text-gray-500">등록된 여론조사가 없습니다.</p>
          ) : (
            <div className="border border-gray-200 rounded-lg overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="text-left px-3 py-2 font-medium text-gray-900">조사일</th>
                    <th className="text-left px-3 py-2 font-medium text-gray-900">조사기관</th>
                    <th className="text-left px-3 py-2 font-medium text-gray-900">후보</th>
                    <th className="text-left px-3 py-2 font-medium text-gray-900">정당</th>
                    <th className="text-right px-3 py-2 font-medium text-gray-900">지지율</th>
                    <th className="px-3 py-2"></th>
                  </tr>
                </thead>
                <tbody>
                  {filteredHistory.map((entry) => (
                    <tr key={entry.id} className="border-t border-gray-100">
                      <td className="px-3 py-2 text-gray-900">{entry.survey_date}</td>
                      <td className="px-3 py-2 text-gray-600">{entry.pollster}</td>
                      <td className="px-3 py-2 text-gray-900">{entry.candidate}</td>
                      <td className="px-3 py-2 text-gray-600">{entry.party}</td>
                      <td className="px-3 py-2 text-right text-gray-900 font-medium">{entry.support}%</td>
                      <td className="px-3 py-2 text-right">
                        <button
                          onClick={() => handleDeletePoll(entry.id)}
                          className="text-red-500 hover:text-red-700 text-xs"
                        >
                          삭제
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {/* VectorDB 상태 */}
        <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-bold text-gray-900 mb-4">VectorDB 상태</h2>
          {stats && (
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center">
                <p className="text-2xl font-bold text-gray-900">{stats.count}</p>
                <p className="text-sm text-gray-500">저장 벡터 수</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-gray-900">{stats.type}</p>
                <p className="text-sm text-gray-500">DB 타입</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-gray-900">{stats.collection}</p>
                <p className="text-sm text-gray-500">컬렉션</p>
              </div>
            </div>
          )}
        </section>

        {/* 파이프라인 실행 */}
        <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-bold text-gray-900 mb-4">파이프라인 실행</h2>
          {pipelineStatus && (
            <div className="mb-4 p-3 bg-gray-50 rounded-lg text-sm">
              <span className="font-medium">현재 상태:</span>{" "}
              <span
                className={
                  pipelineStatus.status === "running"
                    ? "text-yellow-600 font-medium"
                    : pipelineStatus.status === "completed"
                      ? "text-green-600 font-medium"
                      : "text-gray-600"
                }
              >
                {pipelineStatus.status}
              </span>
              {pipelineStatus.started_at && (
                <span className="text-gray-400 ml-2">
                  시작: {new Date(pipelineStatus.started_at).toLocaleString("ko-KR")}
                </span>
              )}
            </div>
          )}

          <div className="flex items-end gap-3 mb-4">
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-1">스크레이퍼</label>
              <select
                value={scraperType}
                onChange={(e) => setScraperType(e.target.value)}
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm text-gray-900"
              >
                <option value="all">전체</option>
                <option value="naver">네이버</option>
                <option value="political">정치 매체</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-1">기간 (일)</label>
              <input
                type="number"
                value={scraperDays}
                onChange={(e) => setScraperDays(e.target.value)}
                placeholder="기본값"
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm text-gray-900 w-24"
              />
            </div>
            <button
              onClick={handleRunPipeline}
              disabled={loading}
              className="bg-gray-900 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-gray-800 disabled:opacity-50"
            >
              수집 실행
            </button>
            <button
              onClick={handleRebuild}
              disabled={loading}
              className="bg-red-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-red-700 disabled:opacity-50"
            >
              전체 재구축
            </button>
            <button
              onClick={() => refresh()}
              className="text-sm text-gray-500 hover:text-gray-900 px-3 py-2"
            >
              새로고침
            </button>
          </div>
        </section>

        {/* VectorDB 정리 */}
        <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-bold text-gray-900 mb-4">만료 벡터 정리</h2>
          <div className="flex items-end gap-3">
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-1">N일 이전 삭제</label>
              <input
                type="number"
                value={purgeInput}
                onChange={(e) => setPurgeInput(e.target.value)}
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm w-24"
              />
            </div>
            <button
              onClick={handlePurge}
              disabled={loading}
              className="bg-yellow-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-yellow-700 disabled:opacity-50"
            >
              정리 실행
            </button>
          </div>
        </section>

        {/* RAG 설정 */}
        <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-bold text-gray-900 mb-4">RAG 설정</h2>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-1">
                검색 기간 (lookback_days)
              </label>
              <input
                type="number"
                value={lookbackDays}
                onChange={(e) => setLookbackDays(e.target.value)}
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm text-gray-900 w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-1">
                검색 수 (top_k)
              </label>
              <input
                type="number"
                value={topK}
                onChange={(e) => setTopK(e.target.value)}
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm text-gray-900 w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-1">
                유사도 임계값 (min_score)
              </label>
              <input
                type="number"
                step="0.1"
                value={minScore}
                onChange={(e) => setMinScore(e.target.value)}
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm text-gray-900 w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-1">
                만료 정리 (purge_days)
              </label>
              <input
                type="number"
                value={purgeDays}
                onChange={(e) => setPurgeDays(e.target.value)}
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm text-gray-900 w-full"
              />
            </div>
          </div>
          <button
            onClick={handleSaveRAG}
            disabled={loading}
            className="bg-gray-900 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-gray-800 disabled:opacity-50"
          >
            설정 저장
          </button>
        </section>
      </main>
    </div>
  );
}
