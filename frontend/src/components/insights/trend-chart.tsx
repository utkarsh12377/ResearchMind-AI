"use client";

import { useMemo } from "react";

import type { Progression, YearBucket } from "@/lib/api/types";
import { cn } from "@/lib/utils";

const WIDTH = 720;
const HEIGHT = 220;
const PADDING = { top: 16, right: 16, bottom: 28, left: 40 };

function scale(value: number, min: number, max: number, lower: number, upper: number): number {
  if (max === min) return (lower + upper) / 2;
  return lower + ((value - min) / (max - min)) * (upper - lower);
}

/**
 * Papers per year as a bar chart.
 *
 * Bars rather than a line, because publication years are discrete buckets: a
 * line between 2019 and 2023 implies values for the years in between that the
 * data does not have.
 */
export function TimelineChart({
  buckets,
  className,
}: {
  buckets: YearBucket[];
  className?: string;
}) {
  const maxCount = Math.max(1, ...buckets.map((bucket) => bucket.paper_count));

  if (buckets.length === 0) {
    return (
      <p className={cn("text-sm text-muted-foreground", className)}>
        No papers carry a publication year yet, so there is nothing to place on a timeline.
      </p>
    );
  }

  const plotWidth = WIDTH - PADDING.left - PADDING.right;
  const plotHeight = HEIGHT - PADDING.top - PADDING.bottom;
  const barWidth = Math.min(48, (plotWidth / buckets.length) * 0.7);

  return (
    <div className={cn("overflow-x-auto", className)}>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="h-auto w-full min-w-[520px]"
        role="img"
        aria-label="Papers per publication year"
      >
        <line
          x1={PADDING.left}
          y1={PADDING.top + plotHeight}
          x2={WIDTH - PADDING.right}
          y2={PADDING.top + plotHeight}
          stroke="currentColor"
          className="text-border"
        />

        {buckets.map((bucket, index) => {
          const centre =
            PADDING.left + ((index + 0.5) / buckets.length) * plotWidth;
          const height = (bucket.paper_count / maxCount) * plotHeight;
          return (
            <g key={bucket.year}>
              <rect
                x={centre - barWidth / 2}
                y={PADDING.top + plotHeight - height}
                width={barWidth}
                height={Math.max(height, 1)}
                rx={3}
                className="fill-primary/70"
              >
                <title>{`${bucket.year}: ${bucket.paper_count} paper(s)`}</title>
              </rect>
              <text
                x={centre}
                y={PADDING.top + plotHeight + 18}
                textAnchor="middle"
                className="fill-muted-foreground text-[10px]"
              >
                {bucket.year}
              </text>
              <text
                x={centre}
                y={PADDING.top + plotHeight - height - 5}
                textAnchor="middle"
                className="fill-foreground text-[10px]"
              >
                {bucket.paper_count}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/**
 * Best reported score per year for one dataset and metric.
 *
 * Here a line is right: the points are measurements of the same quantity over
 * time, so the segment between them carries meaning.
 */
export function ProgressionChart({
  progression,
  className,
}: {
  progression: Progression;
  className?: string;
}) {
  const points = progression.points;

  const path = useMemo(() => {
    if (points.length < 2) return "";
    const years = points.map((point) => point.year);
    const values = points.map((point) => point.value);
    const minYear = Math.min(...years);
    const maxYear = Math.max(...years);
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);

    const plotWidth = WIDTH - PADDING.left - PADDING.right;
    const plotHeight = HEIGHT - PADDING.top - PADDING.bottom;

    return points
      .map((point, index) => {
        const x = scale(point.year, minYear, maxYear, PADDING.left, PADDING.left + plotWidth);
        const y = scale(
          point.value,
          minValue,
          maxValue,
          PADDING.top + plotHeight,
          PADDING.top,
        );
        return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(" ");
  }, [points]);

  if (points.length < 2) return null;

  const years = points.map((point) => point.year);
  const values = points.map((point) => point.value);
  const minYear = Math.min(...years);
  const maxYear = Math.max(...years);
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const plotWidth = WIDTH - PADDING.left - PADDING.right;
  const plotHeight = HEIGHT - PADDING.top - PADDING.bottom;

  return (
    <div className={cn("overflow-x-auto", className)}>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="h-auto w-full min-w-[520px]"
        role="img"
        aria-label={`${progression.metric} on ${progression.dataset} over time`}
      >
        <path d={path} fill="none" stroke="currentColor" className="text-primary" strokeWidth={2} />

        {points.map((point) => {
          const x = scale(point.year, minYear, maxYear, PADDING.left, PADDING.left + plotWidth);
          const y = scale(point.value, minValue, maxValue, PADDING.top + plotHeight, PADDING.top);
          return (
            <g key={`${point.year}-${point.value}`}>
              <circle cx={x} cy={y} r={4} className="fill-primary" />
              <text x={x} y={y - 10} textAnchor="middle" className="fill-foreground text-[10px]">
                {point.value}
              </text>
              <text
                x={x}
                y={PADDING.top + plotHeight + 18}
                textAnchor="middle"
                className="fill-muted-foreground text-[10px]"
              >
                {point.year}
              </text>
              <title>{`${point.year}: ${point.value} (${point.model})`}</title>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
