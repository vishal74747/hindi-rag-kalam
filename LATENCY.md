# Why the 420 ms latency target can't be met

As of 2026-10-18.

## Bottom line

The full answer cannot reach the user within 420 ms with this setup. The median time to the first answer token was 1,772 ms (p95 2,806 ms), and the Groq LLM alone needs about 830 ms before it starts answering.

What can arrive within 420 ms is the first cited result: the top passage with its page, section and chunk id. Hybrid retrieval took 368 ms median, and 294 to 436 ms for the three Hindi test queries.

## Measured results

Source: the `latency.py` output from 2026-10-18. The machine runs CPU-only PyTorch, the LLM is `openai/gpt-oss-120b` on Groq, and top-k is 4. Loading the index and embedder takes about 29 to 30 s once per process start (30,052 ms and 28,875 ms in two runs), not per query.

| Stage | Median | p95 | Max | Queries |
| --- | --- | --- | --- | --- |
| Retrieval, dense | 387 ms | 719 ms | 1,125 ms | 22 |
| Retrieval, hybrid (no translation) | 368 ms | 565 ms | 573 ms | 22 |
| Retrieval, hybrid + query translation | 959 ms | 2,299 ms | 2,730 ms | 22 |
| of which English queries | 1,523 ms | 1,745 ms | 1,745 ms | 11 |
| LLM first token (any) | 650 ms | 1,159 ms | 1,159 ms | 6 |
| LLM first answer token | 830 ms | 1,348 ms | 1,348 ms | 6 |
| LLM full reply | 924 ms | 1,449 ms | 1,449 ms | 6 |
| End to end, first answer token | 1,772 ms | 2,806 ms | 2,806 ms | 6 |

Per query, for the six queries that went through the LLM. The total is retrieval plus first answer token.

| Query | Language | Retrieval | First answer token | Total |
| --- | --- | --- | --- | --- |
| Q1 | Hindi | 369 ms | 857 ms | 1,226 ms |
| Q2 | Hindi | 294 ms | 793 ms | 1,087 ms |
| Q3 | Hindi | 436 ms | 804 ms | 1,240 ms |
| Q4 | English | 1,513 ms | 791 ms | 2,304 ms |
| Q5 | English | 1,350 ms | 1,015 ms | 2,365 ms |
| Q6 | English | 1,458 ms | 1,348 ms | 2,806 ms |

## Where the time goes

Three costs add up on every English question, and two on every Hindi question.

- **Embedding the question (about 370 ms, every query).** Dense (387 ms) and hybrid (368 ms) retrieval take almost the same time, so the keyword search is not the cost. Most of it is most likely turning the question into a vector with bge-m3 on the CPU, plus the ChromaDB lookup. This is an inference; it was not profiled.
- **Translating English questions (about 1.1 s, English only).** Retrieval took 1,350 to 1,513 ms for English questions Q4 to Q6, against 294 to 436 ms for Hindi questions Q1 to Q3. The difference is the LLM call that translates the question into Hindi for the keyword side.
- **The Groq call (about 830 ms, every answer).** The first token arrived after 437 to 1,159 ms, and the first answer token after 791 to 1,348 ms. The gap between the two, 71 to 445 ms, is the reasoning model thinking before it answers.

## Why 420 ms is out of reach for the full answer

The LLM step alone misses the target, before retrieval is counted.

- The fastest first answer token measured was 791 ms (Q4). Even the fastest first token of any kind, 437 ms (Q3), is above 420 ms.
- Retrieval has to finish first. The shortest measured retrieval was 294 ms (Q2), so the best end-to-end result was 1,087 ms, about 2.6 times the target.
- Each answer needs a network round trip to a hosted model, plus its queue and thinking time. Code on this machine cannot remove that.
- Reaching 420 ms for a full answer would need a different design, such as a small local model on a GPU. That was not tested here, and it would change answer quality and cost.

## What can be reached

A first cited result within 420 ms is reachable. Show the top passage with its page, section and chunk id as soon as retrieval finishes, then stream the LLM answer underneath.

- Hindi questions already retrieve in 294 to 436 ms, so two of three are under 420 ms and Q3 (436 ms) is just over.
- English questions need the translation step removed from the wait (cached, made faster, or skipped) to get near 370 ms. Today they take 1,350 to 1,513 ms.
- The streamed answer then starts about 0.8 to 1.3 s after retrieval ends.

Report this as time to first cited result, not time to first answer token. The two are different measurements, and claiming the second would be wrong.

## Options to cut latency

Ordered by expected impact. None of these has been measured yet. After any change, re-run `latency.py` and the retrieval and answer checks.

| Option | What it cuts | Cost or risk |
| --- | --- | --- |
| Show the cited result before the LLM answers | Time to first visible result drops to retrieval time (about 370 ms) | A change in what is reported, not in the LLM speed |
| Remember translations already done | About 1.1 s on repeated English questions | Helps only questions asked before |
| Use a smaller, faster model for translation only | Part of the 1.1 s on English questions | A weaker translation could lower keyword matches |
| Skip translation and use vector search alone for English | All of the 1.1 s | Brings back the English Padma Bhushan miss (Q10) described in EXPERIMENTS.md |
| Faster question embedding: GPU, quantized or ONNX model, cached vectors | Part of the 370 ms | A smaller embedding model means rebuilding the index and re-running the retrieval checks, and quality may drop |
| Trim the LLM step: smaller model, lower thinking effort, fewer or shorter chunks, warm connection | Part of the 830 ms | Possible drop in answer quality; it will not reach 420 ms |
| Keep the process running as a server | The 29 to 30 s load per start | Needs a long-running process, not a one-off script |

## Caveats

- **Small sample.** The LLM timings come from 6 queries, so the p95 figures there are close to the maximum. Retrieval timings come from 22 queries on one CPU-only machine. A first run without a key gave a dense median of 340 ms, against 387 ms in the second, so expect run-to-run variation, and Groq's speed also varies with load.
- **Where the 420 ms target comes from is unknown.** The assignment brief's scoring criteria contain no latency requirement. If the target means retrieval only, the hybrid median of 368 ms already meets it, though the p95 (565 ms) does not.
- **Inferred, not profiled.** The statement that query embedding dominates the 370 ms rests on dense and hybrid taking the same time.
- **Not measured.** Every option in the table above.
