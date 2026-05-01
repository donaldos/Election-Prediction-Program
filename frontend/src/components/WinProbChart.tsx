"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { TimeSeries } from "@/lib/types";

const COLORS = ["#ef4444", "#3b82f6", "#f59e0b", "#10b981", "#8b5cf6"];

interface Props {
  data: TimeSeries;
}

export default function WinProbChart({ data }: Props) {
  if (!data.points.length) {
    return <p className="text-gray-500 text-center py-8">판정 이력이 없습니다.</p>;
  }

  const candidates = Object.keys(data.points[0].candidates);

  const chartData = data.points.map((p) => ({
    date: new Date(p.date).toLocaleDateString("ko-KR", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }),
    ...Object.fromEntries(
      Object.entries(p.candidates).map(([k, v]) => [k, Math.round(v * 100)])
    ),
  }));

  return (
    <ResponsiveContainer width="100%" height={350}>
      <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis dataKey="date" fontSize={12} tick={{ fill: "#6b7280" }} />
        <YAxis
          domain={[0, 100]}
          tickFormatter={(v) => `${v}%`}
          fontSize={12}
          tick={{ fill: "#6b7280" }}
        />
        <Tooltip
          formatter={(value, name) => [`${value}%`, name]}
          labelStyle={{ fontWeight: 600 }}
        />
        <Legend />
        {candidates.map((name, i) => (
          <Line
            key={name}
            type="monotone"
            dataKey={name}
            stroke={COLORS[i % COLORS.length]}
            strokeWidth={2}
            dot={{ r: 4 }}
            activeDot={{ r: 6 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
