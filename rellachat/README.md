# Rellachat voice service

The AI voice agent that answers inbound WhatsApp calls for Rellachat organizations. It runs on Pipecat v1.17.0 with Cartesia for speech-to-text and text-to-speech and Claude as the LLM, and it can transfer a live call to a human advisor.

All our code lives in `rellachat/`. The framework under `src/` stays unmodified.

## Setup

Run these from the repository root with Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[anthropic,cartesia,webrtc,runner,silero,rnnoise]" pytest
cp rellachat/.env.example rellachat/.env   # then fill in the values
```

The `runner` extra brings in FastAPI and uvicorn. `webrtc` brings in aiortc. The `cartesia` and `silero` extras add nothing, because their dependencies (`websockets`, `onnxruntime`) are already core dependencies; we keep them in the command anyway.

## Run

```bash
python rellachat/server.py          # listens on HOST:PORT (default 0.0.0.0:7860)
pytest rellachat/tests              # unit tests, no network or provider calls
```

## Environment (`rellachat/.env`)

| Variable | Default | Purpose |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | — | Claude, for the conversation and the summaries |
| `ANTHROPIC_MODEL` | `claude-haiku-5-5` | Model for the conversation and the summaries |
| `CARTESIA_API_KEY` | — | Cartesia STT and TTS |
| `CARTESIA_VOICE_ID` | — | Default voice, used when the agent has no `voice_id` |
| `CARTESIA_STT_MODEL` | `ink-2` | Cartesia speech-to-text model (`ink-whisper` is the older one) |
| `RELLACHAT_API_URL` | — | For example `http://localhost:8000/api`. Callbacks go to `{url}/voice-service/...` |
| `VOICE_SERVICE_SECRET` | — | Shared secret, sent and checked in `X-Voice-Secret`. If it is unset, every protected endpoint returns 401 |
| `PORT` / `HOST` | `7860` / `0.0.0.0` | HTTP bind address |
| `ICE_SERVERS` | Google STUN | A JSON list of URL strings or `{urls, username, credential}` objects. Invalid input falls back to STUN |

## Endpoints

Every endpoint except `GET /health` requires the header `X-Voice-Secret`, compared in constant time. A missing or wrong secret gets 401 `{"error": "Unauthorized"}`.

| Method | Path | Body | Reply |
| --- | --- | --- | --- |
| GET | `/health` | — | `{ok, calls, test_calls}` |
| POST | `/calls/connect` | `{call_uuid, whatsapp:{token, phone_number_id, webhook}, agent, contact}` | 202 `{ok: true}` (adds `duplicate: true` when the call is already live). 422 if the webhook has no `connect` event |
| POST | `/calls/terminate` | `{call_uuid, whatsapp:{token?, phone_number_id?, webhook}}` | 200 `{ok, found}` |
| POST | `/calls/{call_uuid}/advisor-offer` | `{sdp, type?: "offer", user:{id, name}}` | 200 `{sdp, type: "answer"}`. 409 if no transfer is ringing. 500 if the advisor peer failed |
| POST | `/test/offer` | `{agent, sdp, type?: "offer"}` | 200 `{sdp, type, pc_id}` |

### How `/calls/connect` works

1. The service parses the forwarded Meta webhook into Pipecat's `WhatsAppWebhookRequest` and replies 202 straight away.
2. In the background it answers the call through Pipecat's `WhatsAppClient`, which sends `pre_accept` and then `accept` with our SDP answer on Graph v23. We keep one client per `phone_number_id`, because `update_whatsapp_phone_number_id` does not rebuild the Graph URL. The token is refreshed when it changes. No app secret is passed, because the API already received and verified the webhook.
3. The bot then starts for that `call_uuid`.

If Meta rejects the answer, the service posts `ended` with summary `"La IA no pudo contestar la llamada."` and `duration: 0`.

## The bot (`bot.py`)

The pipeline runs in this order:

```
SmallWebRTC input → CallerAudioGate → Cartesia STT → user aggregator (Silero VAD)
  → Claude → Cartesia TTS → AudioInjector → SmallWebRTC output → assistant aggregator
```

- **Language.** `agent.language` (default `es`) sets the STT and TTS language. The voice is `agent.voice_id`, or `CARTESIA_VOICE_ID` when the agent has none.
- **System prompt.** It is built from the AgentConfig: identity, company, purpose, description, website and instructions. It also carries voice style rules written in Spanish: short sentences, one question at a time, numbers read naturally, no markdown.
- **Greeting.** The agent speaks first, using `agent.greeting` or a default built from the identity and company name.
- **Tools.**
  - `end_call()`: the AI says goodbye, and then the pipeline drains and ends.
  - `transfer_to_advisor(reason)`: registered only when `transfer_enabled` is on.
- **Transcript.** Each user and assistant turn is stored as `{role, text, at}`. Advisor events are stored as `advisor` markers.
- **API callbacks.** All are best-effort: failures are logged and never end the call.
  - `answered`: when the bot starts.
  - `ended {transcript, summary, duration, usage}`: exactly once, after either side hangs up or after `end_call`. The summary is a short Spanish text generated with the Anthropic Messages API.
  - `usage` (billing, `usage.py`): `{ llm: [{model, input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens}], stt_seconds, tts_characters }`, one LLM entry per model. LLM tokens and TTS characters come from Pipecat's `LLMUsageMetricsData` / `TTSUsageMetricsData` (`enable_usage_metrics=True`, collected by `UsageObserver`); the transfer and end-of-call summaries add their own Anthropic `usage`. STT seconds are metered by `SttAudioMeter` from the caller audio that reaches Cartesia STT (Pipecat's own STT usage flush runs at cancel, after observers stop, so a hang-up would lose its tail). Usage that cannot be built is omitted; it never blocks `ended`.
- **Hang-up.** When the AI ends the call, it also terminates the WhatsApp leg through Meta. It skips that step when Meta already sent `terminate`.

## Live transfer and the bridge (`bridge.py`)

1. `transfer_to_advisor` sets the call to **TRANSFER_PENDING**.
   - The caller hears a hold message.
   - `CallerAudioGate` drops the caller's audio, so STT gets nothing and the AI stays quiet.
   - The service generates a summary of the conversation so far and posts `transfer {summary}`.
   - A timer of `transfer_ring_seconds` starts (default 25 when the value is null or 0).
2. **An advisor answers in time** (`advisor-offer`). The call becomes **BRIDGED**:
   - `AdvisorLeg` creates a second `SmallWebRTCConnection` from the advisor's offer and returns the SDP answer.
   - It runs an audio-only pipeline: `input → AdvisorAudioTap → AudioInjector → output`.
   - Audio is routed both ways:
     - caller mic → `CallerAudioGate` → advisor `AudioInjector` → advisor speaker
     - advisor mic → `AdvisorAudioTap` → caller `AudioInjector` → caller speaker
   - The caller-side injector also drops any late TTS audio, so the AI can never talk over the advisor.
   - The service posts `transferred {user_id}` in the background and adds an `advisor` transcript marker.
3. **Either side hangs up.** The other leg is closed: the advisor peer is disconnected, or the WhatsApp call is terminated through Meta. Then `ended` is posted.
4. **The timer expires.** The call goes back to **AI**:
   - The service posts `transfer-cancelled`.
   - A system message is added to the context, and the AI tells the caller that no advisor is free right now and keeps helping.
   - The same happens when the API rejects `transfer`, but `transfer-cancelled` is not sent in that case.

While a call is bridged, neither the caller nor the advisor is transcribed, so no STT cost is spent. The transcript only gets the markers `[Llamada transferida a <name>]` and `[El asesor colgó]`.

## Browser test calls (`/test/offer`)

Test calls use the same pipeline over SmallWebRTC, with no WhatsApp leg and no API callbacks: no `answered` and no `ended`. `transfer_to_advisor` is simulated, so the AI says that on a real call it would transfer now.

## Decisions and contract notes for the API and the web

- **Callbacks after we reply.** `transferred` is posted after the `advisor-offer` reply, and `answered` after the `connect` 202. This avoids deadlocks with a single-threaded `php artisan serve`. The API must accept these callbacks arriving a moment later.
- **SDP format.** `advisor-offer` and `/test/offer` expect a complete, non-trickle SDP offer, meaning ICE gathering is finished in the browser. The reply is `{sdp, type}`, plus `pc_id` for test calls.
- **Terminate after we hang up.** After we end a call ourselves, Meta sends `terminate` to the API, and the API forwards it to us. The service then returns `{ok: true, found: false}`, which is harmless.
- **Contact info.** `contact` is accepted but only used for logging.

## Known limitations

- **TURN is required in production.** The default is a public STUN server only. Set `ICE_SERVERS` with a TURN server, or WhatsApp and advisor media will fail behind NAT.
- **All state is in memory.** Live calls are lost if the process restarts, and you cannot run several replicas without sticky routing per `call_uuid`.
- **No hold music.** The caller hears silence while the transfer rings.
- **No idle timeout.** Pipeline idle timeouts are disabled, because bridged calls have no STT or TTS activity. Calls end when a peer hangs up or the AI calls `end_call`.
