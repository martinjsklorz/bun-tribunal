# Bun Tribunal UI

Static web page (HTML/CSS/JS, no build step) that sends one photo to all three classifiers from
`../CONTRACT.md` in parallel and compares their answers: the LLM (`:8003`), CLEF-Flash (`:8001`) and the CNN (`:8002`).

## Start it

From the project root:

```bash
./run.sh        # set up + start everything, UI on http://localhost:8080
```

The UI alone is just `python3 -m http.server 8080` in `ui/`.

## Backend URLs

Defaults live in `config.js`. Override them per page load with query params, e.g.
`http://localhost:8080/?llm=http://localhost:9003&clef=http://gpu-box:8001&cnn=http://localhost:8002`.

## Status pills

Each backend's `/health` is polled, so states change without a reload.

| State | Meaning |
|---|---|
| **ready** (green) | `ready: true`. |
| **loading** (pulsing yellow) | `ready: false` without a reason, e.g. CLEF-Flash still loading weights. |
| **not set up** (amber) | `ready: false` with a `detail`/`error` message. The card shows the message and the fix: `./run.sh llm` (LLM provider), `./run.sh download` (CLEF-Flash weights) or `./run.sh train` (CNN). |
| **offline** (red) | The server can't be reached. Start it with `./run.sh`, or check CORS and the URLs. |

A model that isn't ready still gets asked in each round; it simply forfeits.

## Rounds, retry and scoreboard

- Pick a file, drag & drop, paste, use the phone camera, or try one of the four samples.
- A card in an error state (offline, not set up, 502, 503, 504) has a **Retry** button that asks only that model again
  about the same image. With several failures, **Retry N failed** in the head-to-head strip retries them all.
  The round's stats and any ground-truth grade are rolled back and re-applied, so the scoreboard never double-counts.
- "Actually it WAS a hotdog" / "It was NOT" grades all three models. Scores are kept in `localStorage`
  (`bun-tribunal:score`).
- "Pitch deck mode" (or `?pitch=1`) is an easter egg.

Unofficial parody; all artwork is original.
