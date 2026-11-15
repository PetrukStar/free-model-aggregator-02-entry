# free-model-aggregator-02-entry

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/PetrukStar/free-model-aggregator-02-entry/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/PetrukStar/free-model-aggregator-02-entry/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

A live free-model aggregator: one server that serves the OpenAI Chat Completions API and the Anthropic Messages API (JSON and server-sent events) and answers every prompt with a real completion from a free OpenRouter model, dialled through make.fun's gateway. It routes each prompt to the best-rated free model that meets its requirements, fails over to the next best when a model refuses, and reports recovered refusals on the `x-makefun-upstream-failures` header.

## Run it

```sh
node main.ts
```

Inside a trial the sandbox sets `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE`, `OPENAI_BASE_URL` and the trial's dummy keys; the server registers its port and catalog with the env, serves `GET /v1/models`, `POST /v1/chat/completions` and `POST /v1/messages`, and exits 0 when the env says the work is done. `node main.ts --list` prints the catalog document and exits.

## Design

- **Rating-ranked routing (quality):** candidates are the brief's free models that meet the prompt's requirements (tools, vision, JSON mode, context), ordered by the brief's `rating_milli`, best first — so a JSON prompt goes to the best JSON-capable model and a plain prompt to the best model overall.
- **Failover without re-trying a dead model:** a model that refuses a call is pushed behind the models that have not refused, and is not dialled again after two refusals in the trial. Every choice is a function of the brief, the requests and the gateway's answers only — no clock, no randomness — so a replay of the trial reproduces the same answers.
- **Both wires, faithfully:** Anthropic requests are translated to Chat Completions and the answer translated back, with the system prompt, tools and tool results, images, stop sequences, usage and stop reasons carried through; streamed answers are relayed as OpenAI SSE or as the Anthropic event sequence (`message_start` … `message_stop`). The `model` field on every answer and every streamed chunk is the id actually dialled.
- **Catalog:** the register body and both `GET /v1/models` answers list exactly the fourteen free models from the brief, once each.

License: MIT (see `LICENSE`).

---

<sub>Entered on [make.fun](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
