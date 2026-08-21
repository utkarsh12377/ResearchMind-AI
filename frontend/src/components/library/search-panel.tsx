"use client";

import { useMutation } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { Loader2, Search, SlidersHorizontal } from "lucide-react";
import { useState } from "react";

import { api } from "@/lib/api/client";
import type { SearchFilters, SearchResponse, SearchResult } from "@/lib/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";

const KINDS = ["text", "table", "figure"] as const;

/**
 * Explains why a result surfaced.
 *
 * Retrieval here is a fusion of two retrievers plus a reranker, and showing
 * only a score would make the ranking unauditable. The per-retriever ranks are
 * the difference between "trust me" and "here is the reason".
 */
function ProvenanceBadges({ result }: { result: SearchResult }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {result.dense_rank !== null && (
        <Badge variant="secondary" className="text-[10px]">
          semantic #{result.dense_rank}
        </Badge>
      )}
      {result.sparse_rank !== null && (
        <Badge variant="secondary" className="text-[10px]">
          keyword #{result.sparse_rank}
        </Badge>
      )}
      {result.dense_rank !== null && result.sparse_rank !== null && (
        <Badge className="text-[10px]">both retrievers</Badge>
      )}
      {result.rerank_score !== null && (
        <Badge variant="outline" className="text-[10px]">
          reranked {result.rerank_score.toFixed(2)}
        </Badge>
      )}
    </div>
  );
}

function ResultCard({ result, index }: { result: SearchResult; index: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, delay: Math.min(index * 0.03, 0.3) }}
    >
      <Card>
        <CardHeader className="pb-3">
          <div className="flex flex-wrap items-start justify-between gap-2">
            <CardTitle className="text-sm font-medium">
              {result.paper_title ?? "Untitled paper"}
            </CardTitle>
            <span className="text-xs text-muted-foreground">
              score {result.score.toFixed(3)}
            </span>
          </div>
          <p className="text-xs text-muted-foreground">
            {result.section_path ?? "body"}
            {result.page_number !== null && ` · page ${result.page_number}`}
            {result.kind !== "text" && ` · ${result.kind}`}
          </p>
        </CardHeader>
        <CardContent className="space-y-3">
          {result.supporting_sentences.length > 0 ? (
            <ul className="space-y-1.5">
              {result.supporting_sentences.map((sentence) => (
                <li key={sentence} className="border-l-2 border-primary/40 pl-3 text-sm">
                  {sentence}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-muted-foreground">{result.content.slice(0, 400)}</p>
          )}
          <ProvenanceBadges result={result} />
        </CardContent>
      </Card>
    </motion.div>
  );
}

export function SearchPanel() {
  const [query, setQuery] = useState("");
  const [showFilters, setShowFilters] = useState(false);
  const [kinds, setKinds] = useState<string[]>([]);
  const [yearFrom, setYearFrom] = useState("");
  const [yearTo, setYearTo] = useState("");
  const [limit, setLimit] = useState(10);

  const search = useMutation<SearchResponse, Error, void>({
    mutationFn: () => {
      const filters: SearchFilters = {};
      if (kinds.length > 0) filters.kinds = kinds;
      if (yearFrom) filters.year_from = Number(yearFrom);
      if (yearTo) filters.year_to = Number(yearTo);
      return api.search(query.trim(), { limit, filters });
    },
  });

  const toggleKind = (kind: string) => {
    setKinds((current) =>
      current.includes(kind) ? current.filter((k) => k !== kind) : [...current, kind],
    );
  };

  const canSearch = query.trim().length > 0 && !search.isPending;

  return (
    <div className="space-y-5">
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (canSearch) search.mutate();
        }}
        className="space-y-3"
      >
        <div className="flex gap-2">
          <Input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search across every indexed passage…"
            aria-label="Search query"
          />
          <Button type="button" variant="outline" onClick={() => setShowFilters((v) => !v)}>
            <SlidersHorizontal className="h-4 w-4" />
            <span className="sr-only">Toggle filters</span>
          </Button>
          <Button type="submit" disabled={!canSearch}>
            {search.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Search className="h-4 w-4" />
            )}
            <span className="ml-2">Search</span>
          </Button>
        </div>

        {showFilters && (
          <Card>
            <CardContent className="grid gap-4 pt-6 sm:grid-cols-3">
              <div className="space-y-2">
                <Label>Content type</Label>
                <div className="flex flex-wrap gap-1.5">
                  {KINDS.map((kind) => (
                    <Button
                      key={kind}
                      type="button"
                      size="sm"
                      variant={kinds.includes(kind) ? "default" : "outline"}
                      onClick={() => toggleKind(kind)}
                    >
                      {kind}
                    </Button>
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="year-from">Published between</Label>
                <div className="flex items-center gap-2">
                  <Input
                    id="year-from"
                    inputMode="numeric"
                    placeholder="from"
                    value={yearFrom}
                    onChange={(event) => setYearFrom(event.target.value.replace(/\D/g, ""))}
                  />
                  <Input
                    inputMode="numeric"
                    placeholder="to"
                    aria-label="Published up to"
                    value={yearTo}
                    onChange={(event) => setYearTo(event.target.value.replace(/\D/g, ""))}
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="limit">Results</Label>
                <Input
                  id="limit"
                  inputMode="numeric"
                  value={limit}
                  onChange={(event) => {
                    const next = Number(event.target.value.replace(/\D/g, "")) || 1;
                    setLimit(Math.min(50, Math.max(1, next)));
                  }}
                />
              </div>
            </CardContent>
          </Card>
        )}
      </form>

      {search.isError && (
        <p className="text-sm text-destructive">{search.error.message}</p>
      )}

      {search.data && (
        <>
          <div className="flex items-center gap-3">
            <p className="text-sm text-muted-foreground">
              {search.data.total} result{search.data.total === 1 ? "" : "s"} for{" "}
              <span className="text-foreground">{search.data.query}</span>
            </p>
            <Separator className="flex-1" />
          </div>

          {search.data.results.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              Nothing matched. Papers only become searchable once ingestion finishes, so check
              the library if you uploaded recently.
            </p>
          ) : (
            <div className="space-y-3">
              {search.data.results.map((result, index) => (
                <ResultCard key={result.chunk_id} result={result} index={index} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
