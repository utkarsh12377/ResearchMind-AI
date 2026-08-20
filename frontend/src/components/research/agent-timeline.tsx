"use client";

import { motion } from "framer-motion";
import {
  BadgeCheck,
  Brain,
  FileSearch,
  Globe,
  ListChecks,
  Loader2,
  Quote,
  RefreshCw,
  ScanSearch,
  Sparkles,
} from "lucide-react";

import type { AgentStep } from "@/lib/api/types";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const AGENT_ICONS: Record<string, typeof Brain> = {
  planner: ListChecks,
  retriever: FileSearch,
  web_search: Globe,
  ranker: ScanSearch,
  reasoner: Brain,
  critic: RefreshCw,
  reflector: RefreshCw,
  verifier: BadgeCheck,
  citation: Quote,
  summarizer: Sparkles,
};

const AGENT_LABELS: Record<string, string> = {
  planner: "Planner",
  retriever: "Retriever",
  web_search: "Web Search",
  ranker: "Ranker",
  reasoner: "Reasoner",
  critic: "Critic",
  reflector: "Reflector",
  verifier: "Verifier",
  citation: "Citation",
  summarizer: "Summarizer",
};

export function AgentTimeline({
  steps,
  isRunning,
}: {
  steps: AgentStep[];
  isRunning: boolean;
}) {
  if (steps.length === 0 && !isRunning) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          Agent activity
          {isRunning && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ol className="space-y-3">
          {steps.map((step, index) => {
            const Icon = AGENT_ICONS[step.agent] ?? Brain;
            const failed = step.summary.endsWith("failed");

            return (
              <motion.li
                key={`${step.agent}-${index}`}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.25 }}
                className="flex gap-3"
              >
                <div
                  className={
                    "mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border " +
                    (failed ? "border-destructive text-destructive" : "text-muted-foreground")
                  }
                >
                  <Icon className="h-3.5 w-3.5" />
                </div>

                <div className="min-w-0 flex-1 space-y-0.5">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">
                      {AGENT_LABELS[step.agent] ?? step.agent}
                    </span>
                    {step.duration_ms > 0 && (
                      <span className="text-xs text-muted-foreground">{step.duration_ms}ms</span>
                    )}
                  </div>
                  <p className="text-sm text-muted-foreground">{step.summary}</p>
                  {step.detail && (
                    <p className="text-xs text-muted-foreground/80">{step.detail}</p>
                  )}
                </div>
              </motion.li>
            );
          })}

          {isRunning && steps.length === 0 && (
            <li className="text-sm text-muted-foreground">Starting the research graph…</li>
          )}
        </ol>
      </CardContent>
    </Card>
  );
}

export function ConfidenceBadge({ confidence }: { confidence: number }) {
  // Thresholds mirror the backend's scoring: a verifier-flagged answer is
  // capped at 0.35, so anything at or below that is explicitly low trust.
  const variant =
    confidence >= 0.7 ? "default" : confidence > 0.35 ? "secondary" : "destructive";
  const label =
    confidence >= 0.7 ? "High confidence" : confidence > 0.35 ? "Moderate" : "Low confidence";

  return (
    <Badge variant={variant}>
      {label} · {(confidence * 100).toFixed(0)}%
    </Badge>
  );
}
