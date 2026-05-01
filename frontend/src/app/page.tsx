"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { District, TimeSeries, Verdict } from "@/lib/types";
import DistrictSelector from "@/components/DistrictSelector";
import VerdictCard from "@/components/VerdictCard";
import WinProbChart from "@/components/WinProbChart";

export default function Home() {
  const [districts, setDistricts] = useState<District[]>([]);
  const [selected, setSelected] = useState("");
  const [verdict, setVerdict] = useState<Verdict | null>(null);
  const [timeseries, setTimeseries] = useState<TimeSeries | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .getDistricts()
      .then((data) => {
        setDistricts(data);
        if (data.length > 0) setSelected(data[0].id);
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!selected) return;
    setLoading(true);
    setError("");

    Promise.all([
      api.getLatestVerdict(selected).catch(() => null),
      api.getTimeSeries(selected).catch(() => null),
    ])
      .then(([v, ts]) => {
        setVerdict(v);
        setTimeseries(ts);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [selected]);

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-xl font-bold text-gray-900">
            Election Radar
          </h1>
          <a
            href="/admin"
            className="text-sm text-gray-500 hover:text-gray-900 transition-colors"
          >
            관리자
          </a>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-8 space-y-8">
        {districts.length > 0 && (
          <DistrictSelector
            districts={districts}
            selected={selected}
            onChange={setSelected}
          />
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-sm">
            {error}
          </div>
        )}

        {loading && (
          <div className="text-center py-12 text-gray-500">
            데이터를 불러오는 중...
          </div>
        )}

        {!loading && verdict && <VerdictCard verdict={verdict} />}

        {!loading && timeseries && (
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-lg font-bold text-gray-900 mb-4">
              승률 추이
            </h2>
            <WinProbChart data={timeseries} />
          </div>
        )}

        {!loading && !verdict && !error && (
          <div className="text-center py-12 text-gray-500">
            아직 판정 결과가 없습니다. 관리자 페이지에서 판정을 실행하세요.
          </div>
        )}
      </main>
    </div>
  );
}
