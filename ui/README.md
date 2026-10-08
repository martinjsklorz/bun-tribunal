# Bun Tribunal UI

Static web page (HTML/CSS/JS, no build step) that sends one photo to all three classifiers in parallel and compares
their answers: the LLM (`:8003`), CLEF-Flash (`:8001`) and the CNN (`:8002`). It speaks only the API in
[`../CONTRACT.md`](../CONTRACT.md).

## Start it

From the project root:

```bash
./run.sh        # set up + start everything, UI on http://localhost:8080
```

The UI alone is just `python3 -m http.server 8080` in `ui/`.

## Configuration

`config.js` holds the defaults. The backend URLs use the page's own host, so a phone on your Wi-Fi works with
`LAN=1 ./run.sh`. Override them per page load with query params, e.g.
`http://localhost:8080/?llm=http://localhost:9003&clef=http://gpu-box:8001&cnn=http://localhost:8002`.

| Setting | Default | |
|---|---|---|
| `healthIntervalMs` / `healthRetryMs` | 15 s / 3 s | `/health` polling, when everything is ready / while something isn't |
| `healthTimeoutMs` | 4 s | per health check |
| `classifyTimeoutMs` | 5 min | browser-side safety net; the card then says **Timed out** and offers **Retry** |
| `uploadMaxSide` / `uploadJpegQuality` | 1280 / 0.9 | larger photos are downscaled once in the browser (EXIF orientation respected) and sent as JPEG |

Health polling pauses while the tab is hidden and refreshes as soon as it is visible again.

## Status pills

| State | Meaning |
|---|---|
| **ready** (green) | `ready: true`. |
| **loading** (pulsing yellow) | `ready: false` without a reason, e.g. CLEF-Flash still loading weights. |
| **not set up** (amber) | `ready: false` with a `detail` (or `error`) message. The card shows it and the fix: `./run.sh llm` (LLM provider), `./run.sh download` (CLEF-Flash weights) or `./run.sh train` (CNN). |
| **offline** (red) | The server can't be reached. Start it with `./run.sh`, or check the URLs. |

A model that isn't ready still gets asked in each round; it simply forfeits.

## Rounds, retry and scoreboard

- Pick a file, drag & drop, paste, use the phone camera, or try one of the four samples.
- The LLM card shows the model's one-line reason, whether its confidence comes from token logprobs or is
  self-reported, and an "N attempts" pill when the server had to retry a busy provider.
- A card in an error state (offline, not set up, 502, 503, 504, timed out) has a **Retry** button that asks only that
  model again about the same image. With several failures, **Retry N failed** in the head-to-head strip retries them
  all. The round's stats and any ground-truth grade are rolled back and re-applied, so the scoreboard never
  double-counts.
- "Actually it WAS a hotdog" / "It was NOT" grades all three models. Scores are kept in `localStorage`
  (`bun-tribunal:score`).
- "Pitch deck mode" (or `?pitch=1`) is an easter egg.
- For automated tests, `window.__bunTribunal.runSample(id)` starts a round with a sample (`hotdog`, `pizza`, `shoe`,
  `dachshund`); its promise resolves once all three verdicts (or errors) are in, or a newer round replaced it.

Unofficial parody; all artwork is original.
