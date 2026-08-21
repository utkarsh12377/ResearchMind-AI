"""Render the Buildathon submission document to PDF."""

from pathlib import Path

import pymupdf

OUTPUT = Path(__file__).resolve().parent.parent / "ResearchMind-AI-Submission.pdf"
PAGE = pymupdf.paper_rect("a4")
MARGIN = 52
FRAME = PAGE + (MARGIN, MARGIN, -MARGIN, -MARGIN)

CSS = """
body { font-family: sans-serif; font-size: 10.5px; color: #1a1a1a; line-height: 1.5; }
h1 { font-size: 26px; color: #0b2545; margin-top: 0; margin-bottom: 4px; }
h2 { font-size: 15px; color: #0b2545; margin-top: 20px; margin-bottom: 6px;
     border-bottom: 1px solid #c7d3e0; padding-bottom: 3px; }
h3 { font-size: 11.5px; color: #14508c; margin-top: 13px; margin-bottom: 3px; }
p  { margin-top: 0; margin-bottom: 8px; }
ul { margin-top: 0; margin-bottom: 9px; }
li { margin-bottom: 4px; }
.sub      { font-size: 12px; color: #52657a; margin-bottom: 2px; }
.meta     { font-size: 9.5px; color: #52657a; margin-bottom: 14px; }
.lead     { font-size: 11.5px; color: #23384f; margin-bottom: 12px; }
.field    { font-size: 9px; color: #14508c; margin-bottom: 2px; margin-top: 12px; }
.answer   { background-color: #f2f6fa; padding: 8px; margin-bottom: 4px; }
.mono     { font-family: monospace; font-size: 9.5px; color: #0b2545; }
.note     { font-size: 9.5px; color: #7a5c00; background-color: #fff8e1; padding: 7px;
            margin-bottom: 8px; }
.stat     { font-size: 9.5px; color: #52657a; }
.tag      { font-size: 9px; color: #52657a; }
b         { color: #0b2545; }
"""

HTML = """
<h1>ResearchMind&nbsp;AI</h1>
<p class="sub">Multi-agent research intelligence over a corpus of scientific papers</p>
<p class="meta">
  Razorpay AI Builder Internship 2026 &middot; Open Track<br/>
  Utkarsh &middot; utkarshvats4108@gmail.com<br/>
  <span class="mono">github.com/utkarsh12377/ResearchMind-AI</span>
</p>

<p class="lead">
  A researcher asking &ldquo;which of these approaches actually works best on this
  benchmark, and where do the papers disagree?&rdquo; cannot answer it from any single
  paper. The answer lives in the relationships between them. ResearchMind AI
  ingests a corpus, builds a knowledge graph over it, and answers questions that
  require reading across all of it &mdash; citing every claim and saying so when it is
  not sure.
</p>

<h2>Form answers</h2>

<p class="field">PROJECT NAME / TITLE</p>
<p class="answer">ResearchMind AI &mdash; Multi-Agent Research Intelligence over Scientific Literature</p>

<p class="field">PROJECT OBJECTIVES &mdash; WHAT DOES IT SOLVE?</p>
<div class="answer">
<p>
Literature review does not scale. Answering one comparative question across
200 papers takes weeks, and existing &ldquo;chat with your PDF&rdquo; tools answer from a
single document, cannot tell you when two papers contradict each other, and give
you fluent prose with no way to check it.
</p>
<p>ResearchMind AI targets four specific failures:</p>
<ul>
<li><b>Retrieval that misses.</b> Semantic search alone loses exact identifiers
like <span class="mono">WikiSQL</span>; keyword search alone loses paraphrase.
Both run, get fused by reciprocal rank, and get reranked by a cross-encoder. A
third retriever walks the knowledge graph, because &ldquo;which other papers used this
dataset&rdquo; is a path, not a passage &mdash; no embedding finds it.</li>
<li><b>Answers you cannot check.</b> Every claim carries a citation marker
resolved back to a specific passage. A verifier agent re-reads the answer against
its sources and flags unsupported claims, and a flagged answer is capped at low
confidence no matter how good retrieval looked.</li>
<li><b>Questions no single paper answers.</b> Comparison tables grouped by
dataset and metric, contradiction detection, trend analysis over time, research
gap discovery, and a cited literature review exportable as Markdown and BibTeX.</li>
<li><b>Demos that only work on the demo.</b> Every layer is an interface with an
offline implementation, so the whole system runs and is tested with no
containers, no API keys, and no external services.</li>
</ul>
</div>

<p class="field">GITHUB REPOSITORY URL</p>
<p class="answer mono">https://github.com/utkarsh12377/ResearchMind-AI</p>

<p class="field">5-MIN PITCH VIDEO LINK</p>
<p class="answer mono">[ paste your video link here before submitting ]</p>

<h2>Build challenges &amp; technical obstacles</h2>
<p class="tag">
  The problems below are the ones that actually cost time. Several were found by
  running the system end to end rather than by any unit test, which is the
  through-line: tests written against mocks would have passed for every one of them.
</p>

<h3>1. Half of hybrid search was silently dead</h3>
<p>
Uploading a paper indexed it into the vector store and nowhere else. The BM25
index was only rebuilt at startup, so every paper uploaded after boot was
invisible to keyword retrieval until the next restart. Every test passed &mdash;
they seeded the index directly and never exercised the upload path.
</p>
<p>
Caught by a live run where a freshly uploaded paper returned
<span class="mono">sparse_rank: null</span> on every hit. The fix made the indexer
write to both retrievers atomically. The lesson stuck: the ranks are now returned
in the API response, so the same failure would be visible rather than silent.
</p>

<h3>2. Uploads returned 500 after succeeding</h3>
<p>
<span class="mono">asyncio.run() cannot be called from a running event loop</span>.
Running Celery in eager mode for local development meant the ingestion task
executed inside the API&rsquo;s own loop. The paper was stored, parsed, and indexed
correctly &mdash; then the response blew up.
</p>
<p>
Fixed by detecting a running loop and dispatching to a single-worker thread pool
when one exists. The subtlety is that the bug only appears in the configuration
used for development, so it would have shipped fine and broken every contributor&rsquo;s
first run.
</p>

<h3>3. Letting a model write database queries</h3>
<p>
Natural-language graph search means an LLM generating Cypher. The obvious
mitigation &mdash; block dangerous keywords &mdash; loses to the first spelling you did not
anticipate.
</p>
<p>
Built a whitelist validator instead: allowed clauses, allowed labels, allowed
relationship types, mandatory LIMIT, single statement only. String literals are
blanked before the keyword scan, so a paper legitimately titled &ldquo;Deleting Noisy
Labels&rdquo; is not read as a DELETE &mdash; and equally, a keyword hidden inside quotes
cannot execute. Its tests are written as attempts to get something past it rather
than as happy-path coverage.
</p>

<h3>4. Retrieved text is an injection vector</h3>
<p>
A model reads retrieved passages as if they were instructions. There is no single
fix, so the defence is layered: zero-width and bidirectional control characters
stripped (invisible in a UI, meaningful to a tokenizer), external content wrapped
in explicit UNTRUSTED delimiters, pattern-matching content dropped rather than
forwarded, and tools that take typed arguments so no generated string reaches a
shell or an arbitrary URL.
</p>
<p>
The judgement call: content tripping the detector is discarded, not passed
through with a warning. Forwarding it would make the system&rsquo;s safety a property
of how well the model resists persuasion.
</p>

<h3>5. A tokenizer bug that quietly deflated every quality metric</h3>
<p>
The evaluation metrics tokenize with a character class allowing dots, so
<span class="mono">0.89</span> and <span class="mono">rouge-l</span> survive
intact. It also meant a sentence-final <span class="mono">&ldquo;encoder.&rdquo;</span>
never matched the same word written mid-sentence &mdash; so faithfulness and relevancy
were understated across the board.
</p>
<p>
Found because one assertion produced 0.5 where the arithmetic said 0.75. The
tempting move was to lower the threshold; the right one was to ask why the number
was wrong. Trailing punctuation is now stripped after matching.
</p>

<h3>6. The comparison table was empty on data that was obviously comparable</h3>
<p>
Comparison groups results on the (dataset, metric) pair. Rule-based extraction
set model and metric but never a dataset, so offline every extracted row was
ungroupable and the feature returned nothing on a corpus visibly full of
comparable numbers.
</p>
<p>
Fixed by attributing a dataset from the known-entity lexicon &mdash; but only within
roughly a sentence of the metric. Matching across the whole document was the
tempting version, and is exactly how this kind of extractor ends up confidently
attributing a number to whichever benchmark was named first. Model attribution
deliberately stays with the LLM, because that genuinely does have to be read out
of the prose.
</p>

<h3>7. Authorization that leaks through result counts</h3>
<p>
The natural way to scope retrieval is to rank everything and filter afterwards.
That leaks: result counts, score distributions, and reranker behaviour all shift
measurably in the presence of documents the caller cannot read.
</p>
<p>
The accessible set is resolved first and the candidate set restricted to it, so
another workspace&rsquo;s content is never scored at all. The same rule governs the
insight endpoints &mdash; passing someone else&rsquo;s paper id narrows the scope to nothing
rather than returning a distinguishable error.
</p>

<h3>8. Observability that would have taken down the metrics store</h3>
<p>
Auto-instrumentation labels HTTP metrics by request path, which opens one
Prometheus time series per paper id. Cardinality is the failure mode that kills a
Prometheus install.
</p>
<p>
Instrumentation is hand-written: routes labelled by path template, statuses
bucketed by class, nothing labelled by user input. Latency buckets are tuned for
this workload too &mdash; retrieval lands in tens of milliseconds and a full agent run
takes tens of seconds, so the library default topping out at 10s would have put
every agent run in the overflow bucket. A test asserts that paper ids never
appear in the metrics output.
</p>

<h3>9. Line endings that broke the container</h3>
<p>
Building on Windows produced <span class="mono">bad interpreter: /bin/sh^M</span>
inside a Linux container. A CRLF shebang. The error names the shell rather than
the line ending, which makes it a genuinely confusing hour. Fixed permanently in
<span class="mono">.gitattributes</span> rather than by hand.
</p>

<h3>10. Building against a framework newer than my references</h3>
<p>
Next.js 16 moved several APIs, and shadcn on base-ui replaced the
<span class="mono">asChild</span> composition pattern with a
<span class="mono">render</span> prop. Rather than guessing from memory, I read
the version&rsquo;s own bundled documentation. Its React Compiler lint rules then caught
a real defect of mine &mdash; a synchronous <span class="mono">setState</span> inside
an effect in the graph simulation &mdash; which I fixed rather than suppressed.
</p>

<h2>What got built</h2>
<ul>
<li><b>Ingestion</b> &mdash; layout-aware PDF parsing, OCR routed only to pages with a
thin text layer, table and figure extraction with caption matching, bibliography
parsing with DOI/arXiv resolution, and chunking that follows the paper&rsquo;s own
section structure and carries breadcrumb paths.</li>
<li><b>Retrieval</b> &mdash; dense embeddings and a hand-written BM25 index fused by
reciprocal rank, then cross-encoder reranking over an over-fetched candidate set.
Every result reports which retriever found it and where reranking moved it.</li>
<li><b>Reasoning</b> &mdash; multi-provider LLM gateway with fallback and token
accounting, token-budgeted context packing that compresses rather than truncates,
and the Self-RAG / Corrective-RAG loop: grade retrieval, rewrite and re-retrieve
if weak, answer, verify.</li>
<li><b>Agents</b> &mdash; ten agents in a LangGraph graph with a bounded
critic&#8596;reflector revision cycle, streaming each step to the UI as it completes.</li>
<li><b>Knowledge graph</b> &mdash; closed-schema entity extraction, GraphRAG expansion,
natural-language Cypher behind the validator, and co-authorship, citation, and
shared-entity network views.</li>
<li><b>Cross-paper analysis</b> &mdash; comparison tables, trend timelines, numeric and
claim-level contradiction detection, gap discovery, and cited literature review
generation with Markdown and BibTeX export.</li>
<li><b>Operations</b> &mdash; Prometheus metrics, Grafana dashboards and alert rules, an
evaluation benchmark gating CI on retrieval quality, Kubernetes manifests with a
migration Job, and a deploy pipeline that rolls back on failure.</li>
</ul>

<h2>Engineering decisions worth defending</h2>
<ul>
<li><b>Reciprocal rank fusion over weighted score blending.</b> Cosine similarity
and BM25 scores are on incompatible scales, and normalising them requires knowing
each distribution &mdash; which shifts with the corpus. RRF uses only the ordering,
which is the part that is actually comparable.</li>
<li><b>Over-fetch before reranking.</b> Reranking exactly <i>k</i> results can only
shuffle them. It cannot recover a relevant passage the first stage ranked at
<i>k+1</i>, so the candidate set is four times the requested limit.</li>
<li><b>Validate model output before it becomes state.</b> Extraction is the one
place a hallucination becomes durable, so unknown labels, unknown relation types,
and edges between the wrong kinds of node are dropped rather than written.</li>
<li><b>Bound the agent revision loop.</b> A critic and a reviser left to argue
will burn tokens indefinitely. The cycle is capped inside the critic.</li>
<li><b>Fallback is not a retry.</b> The LLM gateway falls back on provider
outages but never on 4xx &mdash; a malformed request or a content-filter refusal fails
identically everywhere. Streaming never falls back at all, because tokens already
sent cannot be retracted.</li>
<li><b>Skip, do not fail, what cannot be measured.</b> Benchmark cases needing a
real model are skipped offline rather than failed. A CI gate that is permanently
red because of a missing key stops being read.</li>
</ul>

<h2>Verification</h2>
<ul>
<li><b>605 backend tests</b>, all passing, with no containers or API keys required.
Tests build real PDFs with PyMuPDF and push them through the real ingestion,
indexing, and retrieval pipeline &mdash; which is how four of the bugs above were
found.</li>
<li><b>Retrieval benchmark gates CI</b> on faithfulness, context precision, and
citation validity, ingesting a synthetic corpus through the real pipeline so a
regression in chunking or fusion fails even when every unit test passes.</li>
<li><b>Live end-to-end run</b> across 26 checks: ingestion, upload rejection,
request correlation, hybrid search with both retrievers contributing, all three
graph views, comparison, trends, gaps, review generation, the full agent graph,
and metric cardinality.</li>
<li><b>Clean</b> ruff, eslint, <span class="mono">tsc --noEmit</span>, and
<span class="mono">next build</span>; all migrations round-trip up and down.</li>
</ul>

<p class="note">
<b>Stated honestly:</b> Docker was not available on the build machine, so
Postgres, Redis, Qdrant, Neo4j, real LLM providers, and Tesseract OCR are
code-complete and contract-tested but were not smoke-tested against live
instances. Everything verified above ran against SQLite, the offline embedding
and LLM providers, and the in-memory vector and graph stores. The pluggable
interfaces are exactly what makes that substitution possible &mdash; and exactly what
makes the untested paths a real, named risk rather than a hidden one.
</p>
"""


def main() -> None:
    story = pymupdf.Story(html=HTML, user_css=CSS)
    writer = pymupdf.DocumentWriter(OUTPUT)

    more = True
    pages = 0
    while more:
        device = writer.begin_page(PAGE)
        more, _ = story.place(FRAME)
        story.draw(device)
        writer.end_page()
        pages += 1

    writer.close()
    print(f"wrote {OUTPUT} ({pages} pages, {OUTPUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
