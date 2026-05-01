"use client";

import type { District } from "@/lib/types";

interface Props {
  districts: District[];
  selected: string;
  onChange: (id: string) => void;
}

export default function DistrictSelector({
  districts,
  selected,
  onChange,
}: Props) {
  return (
    <div className="flex gap-2">
      {districts.map((d) => (
        <button
          key={d.id}
          onClick={() => onChange(d.id)}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            selected === d.id
              ? "bg-gray-900 text-white"
              : "bg-gray-100 text-gray-700 hover:bg-gray-200"
          }`}
        >
          {d.name}
        </button>
      ))}
    </div>
  );
}
