# RiskSentinel - Demo Recording Flow

Use this as the operator checklist while recording.

For the spoken track:
- `docs/demo_voiceover_tts.md`

For plain TTS paste:
- `docs/demo_voiceover_elevenlabs.txt`

## Pre-Flight

1. Open the app and keep the sidebar visible.
2. Open `⚙️ Ops` once and confirm `Show explainability panel` is ON.
3. If GPT is locked, enter the judge code in `Judge access code` and click `Unlock GPT`.
4. If the app shows `Synthetic demo dataset active`, use lower correlation thresholds around `0.30` to `0.35`.

## Recommended Recording Flow

### 1. Fast visual shock

In the sidebar:
- Under `⚡ Crisis Presets`, click `SVB Crisis`

Audio cue:
- `C1` in `docs/demo_voiceover_tts.md`
- Time: `0:10 - 0:26`
- Starts with: `It helps answer a central question...`

What this gives you:
- Auto-runs `Build` plus `Shock`
- Populates the network animation
- Populates the downstream dashboards

### 2. Show the network playback

Open `🌐 Stress Lab` and show:
- The animated graph
- The wave playback controls under the graph
- Optional: toggle `🎯 Blast radius only`

Audio cue:
- `C1`
- Keep this screen visible while the line about stress propagation is playing

### 3. Show KPI dashboard and model comparison

In the sidebar:
- Click `⚖️ Compare All 3 Models`

Then open `📊 Surveillance` and show:
- `Affected`, `Defaulted`, `Waves`, `Total Stress`, `Avg Stress`, `Risk Index`
- `Model Comparison`
- `Sector Impact`
- `Most Vulnerable`

Audio cue:
- `C2` in `docs/demo_voiceover_tts.md`
- Time: `0:26 - 0:42`
- Starts with: `This gives supervisors an immediate dashboard...`

### 4. Show the forward-looking layer

Open `🔮 Outlook`.

At the top:
- Leave the default dates unless you need a shorter run
- Click `🔮 Run Outlook`

Inside `📡 Monitor`, show:
- `Current Regime`
- `Fragility Trend`
- `Forward Stress Readiness`
- `Checkpoint Outlook`
- `Watchlist Panel`

Audio cue:
- `C3` in `docs/demo_voiceover_tts.md`
- Time: `0:42 - 1:00`
- Starts with: `The Outlook layer adds forward-looking monitoring...`

### 5. Show forward stress playback from Outlook

Still inside `🔮 Outlook`, open `💥 Stress Test`.

Fastest stable path:
- Click `Run Bank Shock`

Then show:
- The playback metrics row
- The animated network
- `Top Vulnerable Nodes`
- `Intervention Readout`
- `Why These Nodes`

Audio cue:
- `C4` in `docs/demo_voiceover_tts.md`
- Time: `1:00 - 1:20`
- Starts with: `From there, we can launch a forward stress test...`

Optional compare segment:
- Turn ON `Scenario compare`
- Set `Compare ticker` to `GS`
- Keep `Compare model` on `debtrank`
- Set `Compare shock %` to `40`
- Click `💥 Run Stress Playback` again
- Show `Scenario Compare`

### 6. Show explainability and audit

In the sidebar:
- Under `🧩 Scenario Pack`
- Choose `B) Bank comparison strategy`
- Click `▶ Run Scenario`

Then open `🔍 Audit Trail` and show:
- `Route`, `Cache hit`, `In scope`
- Planner / Executor / Critic badges
- `Decision policy`
- `Policy ↔ Executor Split`
- `Judge Dashboard`
- Export buttons such as `📥 Report (.txt)` and `📦 Submission Bundle (.zip)`

Audio cue:
- `C5` then `C6` in `docs/demo_voiceover_tts.md`
- `C5` time: `1:20 - 1:38`
- Starts with: `This is where the platform becomes fully auditable...`
- `C6` time: `1:38 - 1:52`
- Starts with: `RiskSentinel turns financial shock questions...`

### 7. Explicit GPT Agent Segment (recommended)

Use this short segment if you want to clearly demonstrate Azure GPT agents (not only local deterministic simulation).

In `⚙️ Ops`:
- Set `Run GPT on standard shock queries` to ON
- If shown, keep `Agent mode` ON
- If GPT is gated, enter `Judge access code` and click `Unlock GPT`

Then run one complex query from chat:
- `Compare JPM and GS contagion paths and propose mitigation with uncertainty and monitoring triggers.`

What to show:
- In `🌐 Stress Lab`: agent cards (`Architect`, `Quant`, `Advisor`, optional `Critic`)
- In `🔍 Audit Trail`: `Route`, planner/executor/critic badges, and `Judge Dashboard` KPIs

Operator note:
- If GPT is unavailable or rate-limited, switch back to the deterministic path (`SVB Crisis` + `Compare All 3 Models`) to keep the recording stable.

## Operator Sync Cheatsheet

- `C0` `0:00 - 0:10`: show home and tabs only
- `C1` `0:10 - 0:26`: click `SVB Crisis`, then show `Stress Lab`
- `C2` `0:26 - 0:42`: click `Compare All 3 Models`, then show `Surveillance`
- `C3` `0:42 - 1:00`: show `Outlook` monitor and `Run Outlook`
- `C4` `1:00 - 1:20`: show `Outlook` stress playback with `Run Bank Shock`
- `C5` `1:20 - 1:38`: run `B) Bank comparison strategy`, then open `Audit Trail`
- `C6` `1:38 - 1:52`: keep `Audit Trail` KPIs and export buttons on screen

## Backup Fast Flow

If you need a shorter recording:

1. Click `SVB Crisis`
2. Open `🌐 Stress Lab`
3. Click `⚖️ Compare All 3 Models`
4. Open `📊 Surveillance`
5. Open `🔮 Outlook` and click `Run Bank Shock`
6. Open `🔍 Audit Trail` only if you have time
