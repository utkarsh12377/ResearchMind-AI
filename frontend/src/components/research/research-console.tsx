"use client";

import { AlertTriangle, Send } from "lucide-react";
import { useRef, useState } from "react";

import { ApiError, streamResearch } from "@/lib/api/client";
import type { AgentStep, ResearchResult } from "@/lib/api/types";
import { AgentTimeline, ConfidenceBadge } from "@/components/research/agent-timeline";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";

export function ResearchConsole() {
  const [question, setQuestion] = useState("");
  const [steps, setSteps] = useState<AgentStep[]>([]);
  const [result, setResult] = useState<ResearchResult | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!question.trim() || isRunning) return;

    // Abandon any in-flight run before starting another.
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setIsRunning(true);
    setSteps([]);
    setResult(null);
    setError(null);

    try {
      for await (const event of streamResearch(question, { signal: controller.signal })) {
        if (event.type === "step") {
          setSteps((current) => [
            ...current,
            {
              agent: event.agent,
              summary: event.summary,
              detail: event.detail,
              duration_ms: event.duration_ms,
            },
          ]);
        } else if (event.type === "result") {
          setResult(event);
        }
      }
    } catch (caught) {
      if (!controller.signal.aborted) {
        setError(caught instanceof ApiError ? caught.message : "The research run failed.");
      }
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardContent className="pt-6">
          <form onSubmit={handleSubmit} className="space-y-3">
            <Textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a research question, request a comparison, or ask for a literature review…"
              rows={3}
              maxLength={4000}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleSubmit(e);
              }}
            />
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">
                Answers are grounded in your uploaded papers and cited.
              </span>
              <Button type="submit" disabled={isRunning || !question.trim()}>
                <Send className="mr-2 h-4 w-4" />
                {isRunning ? "Researching…" : "Research"}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <AgentTimeline steps={steps} isRunning={isRunning} />

      {result && (
        <>
          {result.errors.length > 0 && (
            <Alert variant="destructive">
              <AlertTitle className="flex items-center gap-2">
                <AlertTriangle className="h-4 w-4" />
                Partial answer
              </AlertTitle>
              <AlertDescription>
                {/* An agent can fail without failing the run, so say so
                    explicitly rather than presenting a degraded answer as
                    complete. */}
                Some agents failed: {result.errors.join("; ")}
              </AlertDescription>
            </Alert>
          )}

          <Card>
            <CardHeader>
              <CardTitle className="flex flex-wrap items-center justify-between gap-2 text-base">
                Answer
                <div className="flex items-center gap-2">
                  {result.plan && <Badge variant="secondary">{result.plan.intent}</Badge>}
                  {result.revision_count > 0 && (
                    <Badge variant="secondary">
                      {result.revision_count} revision{result.revision_count > 1 ? "s" : ""}
                    </Badge>
                  )}
                  <ConfidenceBadge confidence={result.confidence} />
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="whitespace-pre-wrap text-sm leading-relaxed">{result.answer}</div>

              {result.verification && !result.verification.supported && (
                <Alert variant="destructive">
                  <AlertTitle>Unverified claims</AlertTitle>
                  <AlertDescription>
                    <p>{result.verification.reason}</p>
                    {result.verification.unsupported_claims.length > 0 && (
                      <ul className="mt-2 list-inside list-disc">
                        {result.verification.unsupported_claims.map((claim) => (
                          <li key={claim}>{claim}</li>
                        ))}
                      </ul>
                    )}
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          {result.sources.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">
                  Sources ({result.sources.length})
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {result.sources.map((source) => (
                  <div
                    key={source.chunk_id}
                    className={
                      "rounded-md border p-3 " +
                      // Cited sources are visually distinguished so it's clear
                      // which ones actually support the answer.
                      (result.citations.includes(source.index) ? "border-primary/50" : "opacity-70")
                    }
                  >
                    <div className="mb-1 flex items-center gap-2">
                      <Badge variant="secondary">[{source.index}]</Badge>
                      <span className="text-sm font-medium">{source.citation}</span>
                    </div>
                    <p className="line-clamp-3 text-xs text-muted-foreground">{source.content}</p>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
