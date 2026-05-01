import type {
  District,
  PipelineStatus,
  RAGConfig,
  TimeSeries,
  Verdict,
  VerdictList,
  VectorDBStats,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

export const api = {
  getDistricts: () => fetchJSON<District[]>("/admin/districts"),

  getLatestVerdict: (districtId: string) =>
    fetchJSON<Verdict>(`/scores/${districtId}/latest`),

  getVerdictHistory: (districtId: string, limit = 50) =>
    fetchJSON<VerdictList>(`/scores/${districtId}/history?limit=${limit}`),

  getTimeSeries: (districtId: string) =>
    fetchJSON<TimeSeries>(`/scores/${districtId}/timeseries`),

  runVerdict: (districtId: string) =>
    fetchJSON<Verdict>(`/scores/${districtId}/run`, { method: "POST" }),

  getVectorDBStats: () => fetchJSON<VectorDBStats>("/admin/vectordb/stats"),

  getPipelineStatus: () => fetchJSON<PipelineStatus>("/admin/pipeline/status"),

  runPipeline: (scraper = "all", days?: number) =>
    fetchJSON<{ task_id: string; status: string; message: string }>(
      "/admin/pipeline/run",
      {
        method: "POST",
        body: JSON.stringify({ scraper, days }),
      }
    ),

  rebuildVectorDB: () =>
    fetchJSON<{ task_id: string; status: string; message: string }>(
      "/admin/pipeline/rebuild",
      { method: "POST" }
    ),

  purgeVectorDB: (purgeDays: number) =>
    fetchJSON<{ deleted: number; remaining: number; message: string }>(
      "/admin/vectordb/purge",
      {
        method: "POST",
        body: JSON.stringify({ purge_days: purgeDays }),
      }
    ),

  getRAGConfig: () => fetchJSON<RAGConfig>("/admin/config/rag"),

  updateRAGConfig: (data: Record<string, unknown>) =>
    fetchJSON<RAGConfig>("/admin/config/rag", {
      method: "PATCH",
      body: JSON.stringify(data),
    }),
};
