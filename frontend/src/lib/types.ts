export interface CandidateReasoning {
  strengths: string;
  weaknesses: string;
  forecast: string;
}

export interface CandidateScore {
  candidate: string;
  party: string;
  verdict: string;
  win_probability: number;
  reasoning: string | CandidateReasoning;
}

export interface Verdict {
  district_id: string;
  district_name: string;
  date: string;
  candidates: CandidateScore[];
  total_chunks_analyzed: number;
  summary: string;
}

export interface VerdictList {
  district_id: string;
  count: number;
  verdicts: Verdict[];
}

export interface TimeSeriesPoint {
  date: string;
  candidates: Record<string, number>;
}

export interface TimeSeries {
  district_id: string;
  district_name: string;
  points: TimeSeriesPoint[];
}

export interface District {
  id: string;
  name: string;
  candidates: { name: string; party: string; keywords: string[] }[];
}

export interface VectorDBStats {
  type: string;
  collection: string;
  count: number;
}

export interface PipelineStatus {
  task_id: string | null;
  status: string;
  started_at: string | null;
  finished_at: string | null;
  result: Record<string, string> | null;
}

export interface RAGConfig {
  retriever: Record<string, number | null>;
  reranker: Record<string, number | boolean>;
  scorer: Record<string, string | number>;
  purge_days: number | null;
}

export interface PollEntry {
  id: string;
  district_id: string;
  candidate: string;
  party: string;
  support: number;
  pollster: string;
  survey_date: string;
  created_at: string;
}

export interface PollList {
  count: number;
  entries: PollEntry[];
}
