# BAD SMILE — Shopping Copilot (demo UI)

A desktop-first front end for the conversational shopping copilot. It talks to
the **real Python agent** over a small local HTTP bridge, so what the interface
shows is genuine retrieval over the 50,000-product catalog — not canned data.

The UI is a demonstration surface only. Scoring runs headlessly against
`starter.agent.Agent`; nothing here is on that path.

## Run locally

Requires Node.js 20.19+ or 22.12+ (the range the current Vite supports).
Two terminals, and the agent has to start first.

**Terminal 1** — the bridge, from the repository root. It builds the index once
(about 2.5 s) and then serves on `http://localhost:8000`:

```bash
python3 -m scripts.serve
```

**Terminal 2** — the interface:

```bash
cd frontend
npm install
npm run dev
```

Vite prints a URL, usually `http://localhost:5173/`. Open it in a browser.
Stop either process with `Control + C`.

If the bridge is not running the UI says so explicitly rather than failing
silently. `VITE_API_BASE_URL` overrides the bridge address when needed.

## Production check

```bash
npm run build
```

## Behaviour

- Up to ten turns per session, matching the competition protocol.
- Shows the agent's ranked Top 10 for the accumulated conversation.
- Preference chips reflect what the parser actually extracted from the dialogue,
  so you can see the state the agent is reasoning over.
- Each card lists the constraints that product literally matches; the agent
  never claims a match it cannot support.
- When the top results score too closely to be meaningfully ordered, the agent
  says it cannot tell them apart instead of presenting an arbitrary ranking.
- Tiles are abstract category/material graphics: the frozen catalog ships text
  and structured metadata only, with no product images.
- `Start over` clears the session on both the UI and the agent.

`src/api/mockShoppingClient.js` and `src/data/mockCatalog.js` are kept as an
offline fallback for working on the interface without running the Python
service. They are not used by the app.

This is a browser-based UI, not a double-clickable desktop application.
