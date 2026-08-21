"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { GraphEdge, GraphNode } from "@/lib/api/types";
import { cn } from "@/lib/utils";

/**
 * A force-directed graph rendered as plain SVG.
 *
 * Written rather than pulled from d3-force or react-flow because the simulation
 * is about forty lines and a chart library is several hundred kilobytes of
 * JavaScript for a single view. It also keeps the node markup ours, which is
 * what makes hover, selection, and the colour-by-kind legend straightforward.
 *
 * The simulation runs for a fixed number of ticks and then stops. A graph that
 * keeps jiggling is harder to read than one that settles, and this view is for
 * reading rather than for dragging things around.
 */

const WIDTH = 900;
const HEIGHT = 560;
const TICKS = 320;
const REPULSION = 9000;
const SPRING = 0.012;
const IDEAL_EDGE_LENGTH = 90;
const CENTER_PULL = 0.006;
const DAMPING = 0.85;
const MIN_RADIUS = 5;
const MAX_RADIUS = 20;

const KIND_COLORS: Record<string, string> = {
  Paper: "var(--chart-1, #6366f1)",
  Author: "var(--chart-2, #14b8a6)",
  Dataset: "var(--chart-3, #f59e0b)",
  Model: "var(--chart-4, #ec4899)",
  Metric: "var(--chart-5, #8b5cf6)",
  Task: "#0ea5e9",
  Method: "#22c55e",
  Benchmark: "#ef4444",
  Venue: "#a855f7",
  Institution: "#64748b",
};

interface Positioned {
  node: GraphNode;
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
}

function colorFor(kind: string): string {
  return KIND_COLORS[kind] ?? "#94a3b8";
}

function radiusFor(weight: number, maxWeight: number): number {
  if (maxWeight <= 1) return MIN_RADIUS + 3;
  const scale = Math.sqrt(weight / maxWeight);
  return MIN_RADIUS + scale * (MAX_RADIUS - MIN_RADIUS);
}

/**
 * Deterministic starting positions.
 *
 * Random seeding makes the same corpus land differently on every render, which
 * reads as the data having changed when it has not. Seeding from the node id
 * means a reload reproduces the same layout.
 */
function seedPosition(id: string, index: number, total: number): { x: number; y: number } {
  let hash = 0;
  for (let i = 0; i < id.length; i += 1) {
    hash = (hash * 31 + id.charCodeAt(i)) | 0;
  }
  const angle = (index / Math.max(total, 1)) * Math.PI * 2;
  const jitter = ((hash % 100) / 100 - 0.5) * 60;
  const ring = Math.min(WIDTH, HEIGHT) * 0.32;
  return {
    x: WIDTH / 2 + Math.cos(angle) * ring + jitter,
    y: HEIGHT / 2 + Math.sin(angle) * ring + jitter,
  };
}

function simulate(nodes: GraphNode[], edges: GraphEdge[]): Positioned[] {
  const maxWeight = Math.max(1, ...nodes.map((n) => n.weight));
  const points: Positioned[] = nodes.map((node, index) => {
    const { x, y } = seedPosition(node.id, index, nodes.length);
    return { node, x, y, vx: 0, vy: 0, radius: radiusFor(node.weight, maxWeight) };
  });

  const byId = new Map(points.map((point) => [point.node.id, point]));
  const links = edges
    .map((edge) => ({ a: byId.get(edge.source), b: byId.get(edge.target) }))
    .filter((link): link is { a: Positioned; b: Positioned } => Boolean(link.a && link.b));

  for (let tick = 0; tick < TICKS; tick += 1) {
    for (let i = 0; i < points.length; i += 1) {
      for (let j = i + 1; j < points.length; j += 1) {
        const a = points[i];
        const b = points[j];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let distanceSquared = dx * dx + dy * dy;

        // Coincident nodes produce a zero-length vector and then NaN. Nudge
        // them apart deterministically rather than dividing by zero.
        if (distanceSquared < 1) {
          dx = (i % 2 === 0 ? 1 : -1) * 0.5;
          dy = (j % 2 === 0 ? 1 : -1) * 0.5;
          distanceSquared = dx * dx + dy * dy;
        }

        const distance = Math.sqrt(distanceSquared);
        const force = REPULSION / distanceSquared;
        const fx = (dx / distance) * force;
        const fy = (dy / distance) * force;
        a.vx -= fx;
        a.vy -= fy;
        b.vx += fx;
        b.vy += fy;
      }
    }

    for (const { a, b } of links) {
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const distance = Math.max(Math.sqrt(dx * dx + dy * dy), 0.01);
      const displacement = (distance - IDEAL_EDGE_LENGTH) * SPRING;
      const fx = (dx / distance) * displacement;
      const fy = (dy / distance) * displacement;
      a.vx += fx;
      a.vy += fy;
      b.vx -= fx;
      b.vy -= fy;
    }

    for (const point of points) {
      point.vx += (WIDTH / 2 - point.x) * CENTER_PULL;
      point.vy += (HEIGHT / 2 - point.y) * CENTER_PULL;
      point.vx *= DAMPING;
      point.vy *= DAMPING;
      point.x = Math.min(WIDTH - point.radius, Math.max(point.radius, point.x + point.vx));
      point.y = Math.min(HEIGHT - point.radius, Math.max(point.radius, point.y + point.vy));
    }
  }

  return points;
}

export function ForceGraph({
  nodes,
  edges,
  onSelect,
  className,
}: {
  nodes: GraphNode[];
  edges: GraphEdge[];
  onSelect?: (node: GraphNode) => void;
  className?: string;
}) {
  const [hovered, setHovered] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [layout, setLayout] = useState<Positioned[]>([]);
  const frame = useRef<number | null>(null);

  // The simulation is O(n^2) per tick, so every path through it is deferred to
  // the next frame -- including the empty case, which keeps the effect free of
  // a synchronous setState and off the paint path either way.
  useEffect(() => {
    frame.current = window.requestAnimationFrame(() =>
      setLayout(nodes.length === 0 ? [] : simulate(nodes, edges)),
    );
    return () => {
      if (frame.current !== null) window.cancelAnimationFrame(frame.current);
    };
  }, [nodes, edges]);

  const positions = useMemo(
    () => new Map(layout.map((point) => [point.node.id, point])),
    [layout],
  );

  const neighbours = useMemo(() => {
    const map = new Map<string, Set<string>>();
    for (const edge of edges) {
      if (!map.has(edge.source)) map.set(edge.source, new Set());
      if (!map.has(edge.target)) map.set(edge.target, new Set());
      map.get(edge.source)!.add(edge.target);
      map.get(edge.target)!.add(edge.source);
    }
    return map;
  }, [edges]);

  const focus = hovered ?? selected;
  const isDimmed = useCallback(
    (id: string) => {
      if (!focus) return false;
      return id !== focus && !neighbours.get(focus)?.has(id);
    },
    [focus, neighbours],
  );

  const kinds = useMemo(
    () => Array.from(new Set(nodes.map((node) => node.kind))).sort(),
    [nodes],
  );

  if (nodes.length === 0) {
    return (
      <div
        className={cn(
          "flex h-[320px] items-center justify-center rounded-lg border border-dashed text-sm text-muted-foreground",
          className,
        )}
      >
        Nothing to plot yet. Upload a few papers so the graph has entities to connect.
      </div>
    );
  }

  return (
    <div className={cn("space-y-3", className)}>
      <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
        {kinds.map((kind) => (
          <span key={kind} className="inline-flex items-center gap-1.5">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: colorFor(kind) }}
            />
            {kind}
          </span>
        ))}
      </div>

      <div className="overflow-x-auto rounded-lg border bg-card">
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className="h-auto w-full min-w-[640px]"
          role="img"
          aria-label={`Knowledge graph with ${nodes.length} nodes and ${edges.length} edges`}
        >
          <g>
            {edges.map((edge, index) => {
              const a = positions.get(edge.source);
              const b = positions.get(edge.target);
              if (!a || !b) return null;
              const dimmed = isDimmed(edge.source) && isDimmed(edge.target);
              return (
                <line
                  key={`${edge.source}-${edge.target}-${index}`}
                  x1={a.x}
                  y1={a.y}
                  x2={b.x}
                  y2={b.y}
                  stroke="currentColor"
                  className="text-muted-foreground"
                  strokeOpacity={dimmed ? 0.05 : 0.25}
                  strokeWidth={Math.min(1 + edge.weight * 0.4, 3)}
                />
              );
            })}
          </g>

          <g>
            {layout.map((point) => {
              const dimmed = isDimmed(point.node.id);
              const isFocus = focus === point.node.id;
              return (
                <g
                  key={point.node.id}
                  transform={`translate(${point.x}, ${point.y})`}
                  opacity={dimmed ? 0.2 : 1}
                  className="cursor-pointer"
                  onMouseEnter={() => setHovered(point.node.id)}
                  onMouseLeave={() => setHovered(null)}
                  onClick={() => {
                    setSelected(point.node.id === selected ? null : point.node.id);
                    onSelect?.(point.node);
                  }}
                >
                  <circle
                    r={point.radius}
                    fill={colorFor(point.node.kind)}
                    stroke="currentColor"
                    className="text-background"
                    strokeWidth={isFocus ? 2.5 : 1}
                  />
                  {(isFocus || point.radius > 11) && (
                    <text
                      y={point.radius + 12}
                      textAnchor="middle"
                      className="fill-foreground text-[10px]"
                    >
                      {point.node.label.length > 28
                        ? `${point.node.label.slice(0, 27)}…`
                        : point.node.label}
                    </text>
                  )}
                  <title>{`${point.node.kind}: ${point.node.label}`}</title>
                </g>
              );
            })}
          </g>
        </svg>
      </div>
    </div>
  );
}
