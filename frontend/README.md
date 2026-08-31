# BAD SMILE — Shopping Copilot Demo

An English, desktop-first front-end demonstration for a conversational shopping copilot. It currently uses a self-contained Mock client: no Python service, API, LLM, network request, or external product image is required.

The Mock client is intentionally separated from the UI so a future Python Agent API can replace it without redesigning the components.

## Run locally

Open Terminal, then run:

> Requires Node.js 20.19+ or 22.12+ (the current Vite version's supported range).

```bash
cd /Users/yikai/Downloads/frontend
npm install
npm run dev
```

Keep that Terminal window open. Vite will print a URL similar to:

```text
http://localhost:5173/
```

Open that URL in your browser. Stop the app with `Control + C` in Terminal.

## Production check

```bash
npm run build
```

## Demo behavior

- Handles up to ten rounds of conversation per session.
- Displays 12 catalog-style results: ranked Top 10 plus two additional considerations.
- Uses text match levels such as `Strong match` instead of precise ranking scores.
- Uses abstract category and material tiles because the source catalog contains no product images.
- `Start over` resets the in-memory Demo session completely.

This project is a browser-based UI, not a double-clickable desktop application.
