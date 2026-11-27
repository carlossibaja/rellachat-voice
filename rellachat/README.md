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

To run it in Docker on a server, see [Docker / deploy](#docker--deploy).

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
| `TELNYX_API_KEY` | — | Phone calls: lets Pipecat's `TelnyxFrameSerializer` hang up at Telnyx when the agent ends the call. Without it calls still work, but `end_call` cannot hang up |
| `PUBLIC_WS_BASE_URL` | — | Phone calls: public `wss://` base of this service, for example `wss://voice.example.com`. Telnyx connects to `{base}/phone/ws` directly, so the service must be publicly reachable (tunnel in development). Unset: `/phone/calls/prepare` returns 503 |

## Docker / deploy

The image is defined in `rellachat/Dockerfile`. Its build context is the repository root, because it installs the Pipecat framework from `src/` and `pyproject.toml`. The root `.dockerignore` only lets `pyproject.toml`, `README.md`, `LICENSE`, `MANIFEST.in`, `src/` and `rellachat/` into the context, and never any `.env` file, `.git`, `.venv` or the tests.

What the image does:

- **Two stages on `python:3.12-slim`.** The builder installs `.[anthropic,cartesia,webrtc,runner,silero,rnnoise]` with uv into `/opt/venv`, as a regular (non-editable) install. The runtime stage copies only that venv and `rellachat/`. Every dependency has manylinux wheels for x86_64 and aarch64 (only pure-Python `docopt` is built from source), so no compiler is installed; the runtime only adds `libgomp1`.
- **Version.** `setuptools_scm` cannot read git tags without `.git`, so the build sets `SETUPTOOLS_SCM_PRETEND_VERSION` from the build arg `PIPECAT_VERSION` (default `1.17.0`). Bump it when the fork moves to a new Pipecat tag: `--build-arg PIPECAT_VERSION=x.y.z`.
- **Config.** Only environment variables. `config.py` loads `rellachat/.env` only when the file exists and never overrides real variables, so the variables Docker passes are used as they are.
- **Runtime.** Runs `python rellachat/server.py` as the non-root user `rellachat` (uid 10001), with `PYTHONUNBUFFERED=1`. The `HEALTHCHECK` calls `GET /health` on `127.0.0.1:$PORT`.

### Build and run

With Compose, from `rellachat/` (uses `rellachat/.env`):

```bash
cp .env.example .env               # then fill in the production values
docker compose up -d --build
docker compose logs -f voice
docker compose ps                  # shows "healthy" once /health answers
```

Without Compose, from the repository root:

```bash
docker build -f rellachat/Dockerfile -t rellachat-voice .
docker run -d --name rellachat-voice --network host --restart unless-stopped \
  --env-file rellachat/.env rellachat-voice
```

### Production environment

All variables are in the table above. These are the ones a production server must get right:

| Variable | Production value |
| --- | --- |
| `RELLACHAT_API_URL` | The URL of the Rellachat API as this server reaches it, for example `https://api.example.com/api` |
| `VOICE_SERVICE_SECRET` | The same long random secret the API uses. Unset: every protected endpoint returns 401 |
| `PUBLIC_WS_BASE_URL` | The public `wss://` base of this service behind the reverse proxy, for example `wss://voice.example.com`. Unset: phone calls are refused with 503 |
| `TELNYX_API_KEY` | Needed so `end_call` can hang up phone calls at Telnyx |
| `ICE_SERVERS` | Add a TURN server: STUN alone fails when the caller, the advisor or this server is behind a strict NAT. Single-quote the JSON in the env file: `ICE_SERVERS='[{"urls":"turn:turn.example.com:3478","username":"u","credential":"p"}]'` |
| `ANTHROPIC_API_KEY`, `CARTESIA_API_KEY`, `CARTESIA_VOICE_ID` | Provider credentials and the default voice |
| `HOST` / `PORT` | Default `0.0.0.0:7860`. With host networking this is a port on the server itself: firewall it so only the API and the reverse proxy reach it, or set `HOST=127.0.0.1` when both run on the same machine |

### Why host networking

WhatsApp calls and advisor legs are WebRTC through aiortc. aiortc gathers ICE host candidates from the network interfaces it sees and binds a random UDP port per call. On Docker's default bridge network it only sees the container's private IP (`172.17.x.x`), so the SDP advertises an address that Meta and the advisor's browser cannot reach, and media never flows. Publishing a port range does not fix it, because the ports are random and the advertised IP is still wrong. `network_mode: host` (or `--network host`) shares the server's real interfaces, so the advertised candidates are correct.

Consequences:

- Host networking works only on Linux Docker Engine. Docker Desktop on macOS or Windows does not give the container the host's interfaces.
- `ports:` mappings are ignored; the service binds `HOST:PORT` directly on the server.
- The server firewall must allow UDP on the ephemeral port range for WebRTC media, or media must go through a TURN server (`ICE_SERVERS`).

Phone calls (Telnyx) do not use WebRTC. They only use the WebSocket `/phone/ws`, which must be reachable through a public HTTPS/WSS reverse proxy.

### Reverse proxy for `/phone/ws`

Telnyx connects to `{PUBLIC_WS_BASE_URL}/phone/ws?call=...&token=...`, so the proxy must terminate TLS and forward the WebSocket upgrade. Keep the query string, and use a long read timeout, because the socket stays open for the whole call.

Caddy (TLS and WebSocket upgrades are automatic):

```caddyfile
voice.example.com {
    reverse_proxy 127.0.0.1:7860
}
```

nginx:

```nginx
server {
    listen 443 ssl;
    server_name voice.example.com;
    # ssl_certificate / ssl_certificate_key ...

    location /phone/ws {
        proxy_pass http://127.0.0.1:7860;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }

    # Optional: the HTTP endpoints, if the API reaches the service through this domain.
    # They all require X-Voice-Secret; only /phone/ws must be public.
    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_set_header Host $host;
    }
}
```

### Single replica

All call state is in memory (see Known limitations). Run one container per server, and expect live calls to drop on `docker compose up -d --build` or any restart.

## Endpoints

Every endpoint except `GET /health` requires the header `X-Voice-Secret`, compared in constant time. A missing or wrong secret gets 401 `{"error": "Unauthorized"}`.

| Method | Path | Body | Reply |
| --- | --- | --- | --- |
| GET | `/health` | — | `{ok, calls, test_calls}` |
| POST | `/calls/connect` | `{call_uuid, whatsapp:{token, phone_number_id, webhook}, agent, contact}` | 202 `{ok: true}` (adds `duplicate: true` when the call is already live). 422 if the webhook has no `connect` event |
| POST | `/calls/terminate` | `{call_uuid, whatsapp:{token?, phone_number_id?, webhook}}` | 200 `{ok, found}` |
| POST | `/calls/{call_uuid}/advisor-offer` | `{sdp, type?: "offer", user:{id, name}}` | 200 `{sdp, type: "answer"}`. 409 if no transfer is ringing. 500 if the advisor peer failed |
| POST | `/test/offer` | `{agent, sdp, type?: "offer"}` | 200 `{sdp, type, pc_id}` |
| POST | `/calls/outbound/offer` | `{call_uuid, whatsapp:{token, phone_number_id}, agent (direction forced to "outbound", optional call_instructions), contact, initiated_by:{id, name}}` | 200 `{sdp, type: "offer"}`. 409 if `call_uuid` is already live. 500 if the offer could not be created |
| POST | `/calls/{call_uuid}/answer` | `{sdp, type?: "answer", wa_call_id}` | 200 `{ok: true}`. 404 if no outbound call has that uuid. 409 if it was already answered. 422 if the SDP is invalid |
| POST | `/calls/{call_uuid}/cancel` | — | 200 `{ok: true, found}`, idempotent: an unknown or finished call is `found: false`. 409 for an inbound call. Never posts `ended` |
| POST | `/phone/calls/prepare` | `{call_uuid, direction: "inbound"\|"outbound", telnyx:{call_control_id?}, agent, contact, initiated_by?:{id, name}}` | 200 `{stream_url}`. 409 if `call_uuid` is pending or live. 503 without `PUBLIC_WS_BASE_URL` |
| POST | `/phone/calls/{call_uuid}/cancel` | — | 200 `{ok: true, found}`, idempotent. Drops a pending or live phone call. 409 for a WhatsApp call. Never posts `ended` |
| POST | `/phone/calls/{call_uuid}/hangup` | — | 200 `{ok: true, found}`. Telnyx hung up: a live call ends through the normal flow (`ended` with transcript, summary, usage) |
| WS | `/phone/ws?call=<uuid>&token=<token>` | Telnyx media stream | No secret header: the one-time token from `prepare` authorizes it |

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

## Outbound calls (`/calls/outbound/*`)

The business places the call; the API keeps all Meta signaling and the service only does WebRTC.

1. `POST /calls/outbound/offer`: `OutboundWebRTCConnection` (`outbound.py`) subclasses Pipecat's `SmallWebRTCConnection`, which can only answer offers in 1.17. It adds one `sendrecv` audio transceiver to the same aiortc peer, runs `createOffer` + `setLocalDescription` (aiortc gathers every ICE candidate first, so the SDP is complete), and keeps only `sha-256` fingerprints with CRLF endings, the same rule as Pipecat's `_filter_sdp_for_whatsapp` for answers. The call is registered and its pipeline starts, but the AI stays silent.
2. The API sends that offer to Meta's `connect`. If Meta refuses, the API calls `/cancel`: the call is torn down and no `ended` is posted.
3. `POST /calls/{uuid}/answer`: the customer's SDP answer is applied, and `wa_call_id` plus the offer's `phone_number_id`/token become the call's WhatsApp leg, so ending it (`end_call`, advisor hang-up, shutdown) terminates at Meta like an inbound call.
4. When the customer's media connects, the service posts `answered` (billing starts here) and the AI speaks first: the outbound prompt (`agent.direction == "outbound"`) tells it to introduce itself as `identity_name` from the company and say why it calls, from its purpose, instructions and what it knows of the customer. When the offer carries `agent.call_instructions` (optional, up to 2000 chars, written by the advisor in `initiated_by`), the prompt adds them as this call's goal: the AI introduces itself as usual and goes straight to that goal, in its own words, keeping its identity, company, language and rules. Empty or missing instructions leave the outbound prompt unchanged; inbound and test calls ignore the field.
5. Nobody answers within 60 s (`ANSWER_TIMEOUT_SECS`): the call is torn down and `ended` is posted with `duration: 0`, no transcript and summary `"El cliente no contestó la llamada."`. Without a `wa_call_id` the service cannot terminate the ringing call at Meta; the API must do it when it gets that `ended`.
6. Meta's `terminate` (customer hung up or rejected) arrives through the normal `/calls/terminate` forward and runs the normal end flow.

## Phone calls (`/phone/*`, Telnyx)

Real phone numbers through Telnyx Call Control with bidirectional media streaming. The API keeps all Telnyx signaling (`answer`, `dial`, webhooks); the service only handles the media stream.

1. `POST /phone/calls/prepare` registers the call (`phone.py`, `PhoneCallRegistry`) and returns `stream_url = {PUBLIC_WS_BASE_URL}/phone/ws?call=<uuid>&token=<token>`. The token is random (`secrets.token_urlsafe(32)`), compared in constant time and usable once. A prepared call Telnyx never streams expires after 120 s (`PHONE_STREAM_TTL_SECS`).
2. The API passes `stream_url` to Telnyx `answer` (inbound) or `dial` (outbound), asking for a bidirectional PCMU 8 kHz stream. Telnyx opens `/phone/ws`; a bad, unknown, expired or reused token closes it with 1008 before accepting.
3. The service reads Telnyx's `connected` and `start` events (10 s limit). `start` gives `stream_id`, `start.call_control_id` and `start.media_format.encoding` (PCMU or PCMA), which build `TelnyxFrameSerializer` with `TELNYX_API_KEY` and `auto_hang_up`. A `call_control_id` that differs from the one sent to `prepare` is rejected. A stream that fails here posts `ended` with `duration: 0` and `"La IA no pudo contestar la llamada."`.
4. `PhoneVoiceCall` (a `VoiceCall`) runs the same pipeline as WhatsApp calls over `FastAPIWebsocketTransport`: same STT/LLM/TTS, tools through the API, transcript, usage (STT meter included), summary, `answered` and `ended`. The socket lives as long as the `/phone/ws` handler, so the call runs inside it. Inbound calls greet as soon as the stream starts; outbound calls post `answered` and let the AI open then (Telnyx starts streaming once the callee answers).
5. The agent is forced to `transport: "phone"`, the prepared `direction`, `transfer_enabled: false` (no advisors on the phone in Phase 1), and loses the messaging tools (`enviar_producto`, `enviar_precios`, `crear_pedido`). The phone prompt says nothing can be sent and, with `buscar_productos`, tells the AI to say a product's name, price and availability out loud right away. WhatsApp prompts are byte-identical to before (hash test).
6. Ending: `end_call` or shutdown ends the pipeline and the serializer hangs up at Telnyx. Telnyx hanging up closes the socket, and the API also calls `/phone/calls/{uuid}/hangup`; both lead to the same single `ended`. A hang-up of a call that never streamed posts `ended` with `duration: 0` (`"El cliente no contestó la llamada."` for outbound). `/phone/calls/{uuid}/cancel` (Telnyx `dial` failed) drops the call without `ended`.

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
