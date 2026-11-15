// Entry for make.fun's free-model-aggregator-02: a single server that fronts
// OpenRouter's free models behind both completion wires.
//
// A client points at this server and every answer it gets comes from a real
// free model dialled through the grader's gateway. The trial hands the
// process a brief naming the free models (with the house's ratings and what
// each can do) and the six prompts it will send, in order. The server picks
// a port, registers it and its catalog with the env, and answers each prompt
// with the best-rated free model that can meet it. When a model refuses, the
// prompt moves to the next best model that meets it and the refusal is
// reported back on a header, so one bad model never sinks the trial.
// Nothing here reads the clock or rolls dice: every decision follows from
// the brief, the requests and the gateway's answers, so a replay of a trial
// plays out exactly the same way.
//
// Run inside a trial with the sandbox's env vars set; `node main.ts --list`
// prints the catalog and stops.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

const WIRE_VERSION = 1;
const WORK_POLL_MS = 100;
const BODY_LIMIT = 8 * 1024 * 1024;
const CALL_BUDGET_MS = 25_000;
const REFUSAL_CAP = 2;
const FALLBACK_TOKENS = 1024;
const REFUSAL_PENALTY = 100_000;

const envBase = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const briefPath = process.env.MAKEFUN_PROMPT_FILE ?? '';
const gatewayBase = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
const gatewayToken = process.env.OPENAI_API_KEY ?? '';

type Json = { [key: string]: unknown };
const isObj = (v: unknown): v is Json => typeof v === 'object' && v !== null && !Array.isArray(v);
const asText = (v: unknown): string => (typeof v === 'string' ? v : '');
const asNum = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const asBool = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);
const numOr = (v: unknown, fallback: number): number => (typeof v === 'number' && Number.isFinite(v) ? v : fallback);

function say(msg: string): void {
  process.stderr.write(`agg: ${msg}\n`);
}

interface FreeModel {
  id: string;
  provider: string;
  label: string;
  rating: number;
  context: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

interface Need {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minCtx?: number;
}

interface PromptSpec {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  need: Need;
}

interface BriefDoc {
  models: FreeModel[];
  prompts: PromptSpec[];
  problem: string | null;
}

interface CatalogDoc {
  providers: {
    id: string;
    base_url: string;
    key_env: string;
    wire: 'openai';
    models: { id: string; free: boolean; context_length: number | null }[];
  }[];
}

const PROVIDER_HOME = 'https://openrouter.ai/api/v1';
const KEY_VAR = 'OPENROUTER_API_KEY';
const BRIEF_PREFIX = 'free-model-aggregator-02/';
const CREATED_EPOCH = 1767225600;
const CREATED_AT = '2026-01-01T00:00:00Z';

// --- the brief: two JSON documents hiding in prose, found by their schema ---

/** The offset of the `}` that closes the object opened at `from`, or -1. */
function closeBrace(text: string, from: number): number {
  let depth = 0;
  let quoted = false;
  for (let i = from; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '\\') i++;
      else if (ch === '"') quoted = false;
    } else if (ch === '"') quoted = true;
    else if (ch === '{') depth++;
    else if (ch === '}') {
      depth--;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Each top-level object in the prose that names a schema, in order. */
function schemaDocs(text: string): Json[] {
  const found: Json[] = [];
  let at = text.indexOf('{');
  while (at !== -1) {
    const end = closeBrace(text, at);
    if (end === -1) break;
    try {
      const parsed: unknown = JSON.parse(text.slice(at, end + 1));
      if (isObj(parsed) && typeof parsed.schema === 'string') found.push(parsed);
    } catch {
      // Not a standalone object; keep scanning.
    }
    at = text.indexOf('{', at + 1);
  }
  return found;
}

/** First document under our prefix holding one of `keys` as an array. */
function pickDoc(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    if (!asText(doc.schema).includes(BRIEF_PREFIX)) continue;
    for (const key of keys) if (Array.isArray(doc[key])) return { doc, key };
  }
  return null;
}

function toModel(v: unknown): FreeModel | null {
  if (!isObj(v)) return null;
  const raw = asText(v.id) || asText(v.model);
  if (raw === '') return null;
  const who = asText(v.provider) || asText(v.wire);
  const id = who === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    id,
    provider: who || 'openrouter',
    label: asText(v.display_name) || raw,
    rating: asNum(v.rating_milli) ?? 0,
    context: asNum(v.context_length),
    tools: asBool(v.tools),
    vision: asBool(v.vision),
    json: asBool(v.json),
    free: v.free !== false,
  };
}

function toPrompt(v: unknown): PromptSpec | null {
  if (!isObj(v)) return null;
  const id = asText(v.id);
  if (id === '') return null;
  const need = isObj(v.requires) ? v.requires : {};
  const out: Need = {};
  if (need.tools === true) out.tools = true;
  if (need.vision === true) out.vision = true;
  if (need.json === true) out.json = true;
  if (typeof need.min_context === 'number' && Number.isFinite(need.min_context)) out.minCtx = need.min_context;
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    need: out,
  };
}

function loadBrief(path: string): BriefDoc {
  if (path === '') return { models: [], prompts: [], problem: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], problem: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocs(text);
  const mdoc = pickDoc(docs, ['models', 'ratings']);
  if (mdoc === null) return { models: [], prompts: [], problem: `the brief at ${path} holds no models document` };
  const models = (mdoc.doc[mdoc.key] as unknown[]).flatMap((v) => {
    const m = toModel(v);
    return m === null ? [] : [m];
  });
  const pdoc = pickDoc(docs, ['prompts', 'manifest']);
  if (pdoc === null) return { models, prompts: [], problem: `the brief at ${path} holds no prompts document` };
  const prompts = (pdoc.doc[pdoc.key] as unknown[]).flatMap((v) => {
    const p = toPrompt(v);
    return p === null ? [] : [p];
  });
  return { models, prompts, problem: null };
}

const brief = loadBrief(briefPath);

/** The catalog: one provider, every free model in the brief, once each. */
function makeCatalog(doc: BriefDoc): CatalogDoc {
  const seen = new Set<string>();
  const models: CatalogDoc['providers'][number]['models'] = [];
  for (const m of doc.models) {
    if (!m.free || seen.has(m.id)) continue;
    seen.add(m.id);
    models.push({ id: m.id, free: true, context_length: m.context });
  }
  if (models.length === 0) return { providers: [] };
  return { providers: [{ id: 'openrouter', base_url: PROVIDER_HOME, key_env: KEY_VAR, wire: 'openai', models }] };
}

const catalog = makeCatalog(brief);

// --- routing: best-rated model that meets the need, refusals pushed back ---

/** Refusals per model id so far in this trial. */
const refusals = new Map<string, number>();

function canServe(need: Need, m: FreeModel): boolean {
  if (need.tools === true && m.tools !== true) return false;
  if (need.vision === true && m.vision !== true) return false;
  if (need.json === true && m.json !== true) return false;
  if (need.minCtx !== undefined && (m.context === null || m.context < need.minCtx)) return false;
  return true;
}

/**
 * Candidates for a need, best first. A model's score is its rating minus a
 * heavy penalty per refusal, so a model that has refused always trails one
 * that has not, and within each group the higher rating wins. A model that
 * refused twice is out for the rest of the trial.
 */
function ranked(need: Need): FreeModel[] {
  return brief.models
    .filter((m) => m.free && canServe(need, m) && (refusals.get(m.id) ?? 0) < REFUSAL_CAP)
    .map((m) => ({ m, score: m.rating - (refusals.get(m.id) ?? 0) * REFUSAL_PENALTY }))
    .sort((a, b) => b.score - a.score)
    .map((x) => x.m);
}

/** Which prompt is arriving, by the brief's order; the body's own needs otherwise. */
let arrived = 0;

function needFor(body: Json): Need {
  const spec = brief.prompts[arrived];
  arrived++;
  if (spec !== undefined) return spec.need;
  const msgs = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const msg of msgs) {
    if (!isObj(msg) || !Array.isArray(msg.content)) continue;
    for (const part of msg.content) {
      if (isObj(part) && (part.type === 'image' || part.type === 'image_url')) vision = true;
    }
  }
  const fmt = isObj(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: fmt === 'json_object' || fmt === 'json_schema',
  };
}

function gatewayChatUrl(): string {
  if (gatewayBase === '') return '';
  return gatewayBase.endsWith('/v1') ? `${gatewayBase}/chat/completions` : `${gatewayBase}/v1/chat/completions`;
}

interface FailRow {
  status: number;
  code: string;
  model: string | null;
}

function errCode(status: number, text: string): string {
  try {
    const v: unknown = JSON.parse(text);
    if (isObj(v) && isObj(v.error) && typeof v.error.code === 'string') return v.error.code;
    if (isObj(v) && typeof v.code === 'string') return v.code;
  } catch {
    // The body is not JSON; fall through to the status.
  }
  if (status === 429) return 'rate_limited';
  if (status === 401) return 'unauthorized';
  if (status >= 500) return 'upstream_error';
  return `http_${status}`;
}

/**
 * One dial of one model. A 2xx answer comes back; anything else is recorded
 * as a refusal and returns null, so the caller can move to the next model.
 * A prompt costs one gateway call per model it tries, never two on the same.
 */
async function tryModel(model: string, body: Json, fails: FailRow[]): Promise<Response | null> {
  const out: Json = { ...body, model };
  if (typeof out.max_tokens !== 'number' && typeof out.max_completion_tokens !== 'number') out.max_tokens = FALLBACK_TOKENS;
  const headers: Record<string, string> = { 'content-type': 'application/json', authorization: `Bearer ${gatewayToken}` };
  if (body.stream === true) headers.accept = 'text/event-stream';
  let upstream: Response;
  try {
    upstream = await fetch(gatewayChatUrl(), {
      method: 'POST',
      headers,
      body: JSON.stringify(out),
      signal: AbortSignal.timeout(CALL_BUDGET_MS),
    });
  } catch (err) {
    refusals.set(model, (refusals.get(model) ?? 0) + 1);
    fails.push({ status: 0, code: 'unreachable', model });
    say(`dial ${model} failed: ${(err as Error).message}`);
    return null;
  }
  if (!upstream.ok) {
    const text = await upstream.text().catch(() => '');
    refusals.set(model, (refusals.get(model) ?? 0) + 1);
    fails.push({ status: upstream.status, code: errCode(upstream.status, text), model });
    say(`dial ${model} refused HTTP ${upstream.status}`);
    return null;
  }
  return upstream;
}

/** Walk the ranked models until one answers; null when all of them refuse. */
async function answer(body: Json, need: Need): Promise<{ model: string; upstream: Response; fails: FailRow[] } | null> {
  const fails: FailRow[] = [];
  for (const cand of ranked(need)) {
    const upstream = await tryModel(cand.id, body, fails);
    if (upstream !== null) return { model: cand.id, upstream, fails };
  }
  return null;
}

/** The x-makefun-upstream-failures header for a prompt's trail, if there is one. */
function trailValue(fails: FailRow[]): string | null {
  if (fails.length === 0) return null;
  const text = JSON.stringify(fails.slice(0, 8).map((f) => ({ status: f.status, code: f.code, model: f.model })));
  return text.length > 8192 ? null : text;
}

// --- the two wires, and the translation between them ---

function listOpenAI(): Json {
  return {
    object: 'list',
    data: brief.models
      .filter((m) => m.free)
      .map((m) => ({ id: m.id, object: 'model', created: CREATED_EPOCH, owned_by: m.provider })),
  };
}

function listAnthropic(): Json {
  return {
    data: brief.models
      .filter((m) => m.free)
      .map((m) => ({ type: 'model', id: m.id, display_name: m.label, created_at: CREATED_AT })),
    has_more: false,
  };
}

function failOpenAI(message: string, type: string): Json {
  return { error: { message, type, code: null } };
}

function failAnthropic(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** The text of a string or a block array's text blocks. */
function plainText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isObj(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** One Anthropic block as a Chat content part, or null when it has no twin. */
function toPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isObj(block.source)) {
    const src = block.source;
    if (src.type === 'base64' && typeof src.data === 'string') {
      const media = typeof src.media_type === 'string' ? src.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${src.data}` } };
    }
    if (src.type === 'url' && typeof src.url === 'string') return { type: 'image_url', image_url: { url: src.url } };
  }
  return null;
}

/** A tool_result's content as plain text. */
function resultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const t = plainText(content);
    return t !== '' ? t : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object. */
function asObject(value: unknown): Json {
  if (isObj(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isObj(parsed)) return parsed;
    } catch {
      // Not JSON; the input is empty.
    }
  }
  return {};
}

function toStopReason(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic request as a Chat Completions body (no model yet; the caller
 * names it per attempt). System becomes a system message, images become
 * image_url parts, tool_use becomes tool_calls, tool_result becomes tool
 * messages, and caps, sampling, stop strings, tools and streaming carry over.
 */
function toChatBody(body: Json): Json {
  const messages: Json[] = [];
  const system = plainText(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isObj(raw)) continue;
    const fromAssistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: fromAssistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (fromAssistant) {
      const calls: Json[] = [];
      for (const block of content) {
        if (!isObj(block) || block.type !== 'tool_use') continue;
        calls.push({
          id: asText(block.id),
          type: 'function',
          function: { name: asText(block.name), arguments: JSON.stringify(isObj(block.input) ? block.input : {}) },
        });
      }
      const text = plainText(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(calls.length > 0 ? { tool_calls: calls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isObj(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: asText(block.tool_use_id), content: resultText(block.content) });
        continue;
      }
      const part = toPart(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isObj(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: asText(raw.description),
            parameters: isObj(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isObj(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A Chat answer as a Messages object; `model` is the id really dialled. */
function toMessageBody(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const first = isObj(choices[0]) ? choices[0] : {};
  const msg = isObj(first.message) ? first.message : {};
  const content: Json[] = [];
  const text = typeof msg.content === 'string' ? msg.content : plainText(msg.content);
  if (text !== '') content.push({ type: 'text', text });
  for (const call of Array.isArray(msg.tool_calls) ? msg.tool_calls : []) {
    if (!isObj(call) || !isObj(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: asText(call.id) || 'toolu_unknown',
      name: asText(call.function.name),
      input: asObject(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isObj(body.usage) ? body.usage : {};
  return {
    id: `msg_${asText(body.id) || 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: toStopReason(first.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numOr(usage.prompt_tokens, 0), output_tokens: Math.max(1, numOr(usage.completion_tokens, 0)) },
  };
}

/** A Chat answer stamped with the id really dialled (usage floored at one out). */
function withModel(body: Json, model: string): Json {
  const out = { ...body, model };
  if (isObj(out.usage)) out.usage = { ...out.usage, completion_tokens: Math.max(1, numOr(out.usage.completion_tokens, 0)) };
  return out;
}

/** One SSE event line pair. */
function sse(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One SSE frame with every data payload retagged to the dialled model. */
function retagFrame(frame: string, model: string): string {
  const lines = frame.split('\n').map((line) => {
    const t = line.trim();
    if (!t.startsWith('data:')) return line;
    const payload = t.slice(5).trim();
    if (payload === '' || payload === '[DONE]') return line;
    try {
      const chunk: unknown = JSON.parse(payload);
      if (isObj(chunk)) return `data: ${JSON.stringify({ ...chunk, model })}`;
    } catch {
      // Not JSON; relay as it came.
    }
    return line;
  });
  return `${lines.join('\n')}\n\n`;
}

/** Relay a Chat SSE stream, retagging each chunk to the dialled model. */
async function* relayChat(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const dec = new TextDecoder();
  let buf = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buf.indexOf('\n\n');
      while (at !== -1) {
        yield retagFrame(buf.slice(0, at), model);
        buf = buf.slice(at + 2);
        at = buf.indexOf('\n\n');
      }
    }
    buf += dec.decode();
    if (buf.trim() !== '') yield retagFrame(buf, model);
  } finally {
    reader.releaseLock();
  }
}

/**
 * A Chat SSE stream rewritten as Anthropic events: message_start, one
 * content block per text run or tool call, message_delta with the stop
 * reason and usage, then message_stop.
 */
async function* relayMessages(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const dec = new TextDecoder();
  let buf = '';
  let started = false;
  let finished = false;
  let heldStop: string | null = null;
  let textAt = -1;
  let nextAt = 0;
  const toolAt = new Map<number, number>();
  let inTokens = 0;
  let outTokens = 0;

  const begin = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sse('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const stopBlocks = (): string[] => {
    const out: string[] = [];
    if (textAt !== -1) {
      out.push(sse('content_block_stop', { type: 'content_block_stop', index: textAt }));
      textAt = -1;
    }
    for (const idx of [...toolAt.values()].sort((a, b) => a - b)) {
      out.push(sse('content_block_stop', { type: 'content_block_stop', index: idx }));
    }
    toolAt.clear();
    return out;
  };
  const finish = (reason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...stopBlocks(),
      sse('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: reason, stop_sequence: null },
        usage: { output_tokens: Math.max(1, outTokens) },
      }),
      sse('message_stop', { type: 'message_stop' }),
    ];
  };
  const eventsFor = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const t = line.trim();
      if (!t.startsWith('data:')) continue;
      const payload = t.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...begin(), ...finish(heldStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isObj(chunk)) continue;
      if (isObj(chunk.usage)) {
        inTokens = numOr(chunk.usage.prompt_tokens, 0);
        outTokens = numOr(chunk.usage.completion_tokens, 0);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const first = isObj(choices[0]) ? choices[0] : null;
      if (first === null) continue;
      const delta = isObj(first.delta) ? first.delta : {};
      out.push(...begin());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textAt === -1) {
          textAt = nextAt;
          nextAt++;
          out.push(sse('content_block_start', { type: 'content_block_start', index: textAt, content_block: { type: 'text', text: '' } }));
        }
        out.push(sse('content_block_delta', { type: 'content_block_delta', index: textAt, delta: { type: 'text_delta', text: delta.content } }));
      }
      for (const call of Array.isArray(delta.tool_calls) ? delta.tool_calls : []) {
        if (!isObj(call)) continue;
        const fromUpstream = typeof call.index === 'number' ? call.index : toolAt.size;
        let idx = toolAt.get(fromUpstream);
        if (idx === undefined) {
          if (textAt !== -1) {
            out.push(sse('content_block_stop', { type: 'content_block_stop', index: textAt }));
            textAt = -1;
          }
          idx = nextAt;
          nextAt++;
          toolAt.set(fromUpstream, idx);
          const fn = isObj(call.function) ? call.function : {};
          out.push(
            sse('content_block_start', {
              type: 'content_block_start',
              index: idx,
              content_block: { type: 'tool_use', id: asText(call.id) || `toolu_${idx}`, name: asText(fn.name), input: {} },
            }),
          );
        }
        const fn = isObj(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sse('content_block_delta', { type: 'content_block_delta', index: idx, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      if (first.finish_reason !== null && first.finish_reason !== undefined) heldStop = toStopReason(first.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buf.indexOf('\n\n');
      while (at !== -1) {
        for (const ev of eventsFor(buf.slice(0, at))) yield ev;
        buf = buf.slice(at + 2);
        at = buf.indexOf('\n\n');
      }
    }
    buf += dec.decode();
    if (buf.trim() !== '') for (const ev of eventsFor(buf)) yield ev;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const ev of begin()) yield ev;
    for (const ev of finish(heldStop ?? 'end_turn')) yield ev;
  }
}

// --- HTTP: the three routes, register, and the work poll ---

async function readBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buf.length;
    if (size > BODY_LIMIT) return null;
    chunks.push(buf);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isObj(value) ? value : null;
  } catch {
    return null;
  }
}

function replyJson(res: ServerResponse, status: number, body: unknown, extra: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extra });
  res.end(text);
}

function replyError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') replyJson(res, status, failAnthropic(type, message));
  else replyJson(res, status, failOpenAI(message, type));
}

function failureHeaders(fails: FailRow[]): Record<string, string> {
  const v = trailValue(fails);
  return v === null ? {} : { 'x-makefun-upstream-failures': v };
}

function serveModels(req: IncomingMessage, res: ServerResponse): void {
  if (typeof req.headers['x-api-key'] === 'string') replyJson(res, 200, listAnthropic());
  else replyJson(res, 200, listOpenAI());
}

async function serveChat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readBody(req);
  if (body === null) return replyError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await answer(body, needFor(body));
  if (served === null) return replyError(res, 'openai', 502, 'upstream_error', 'every free model that meets this request refused it');
  const { model, upstream, fails } = served;
  if (upstream.body === null) return replyError(res, 'openai', 502, 'upstream_error', 'the gateway answered with no body');
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...failureHeaders(fails) });
    try {
      for await (const frame of relayChat(upstream.body, model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  const found: unknown = await upstream.json();
  replyJson(res, 200, withModel(isObj(found) ? found : {}, model), failureHeaders(fails));
}

async function serveMessages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readBody(req);
  if (body === null) return replyError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await answer(toChatBody(body), needFor(body));
  if (served === null) return replyError(res, 'anthropic', 502, 'api_error', 'every free model that meets this request refused it');
  const { model, upstream, fails } = served;
  if (upstream.body === null) return replyError(res, 'anthropic', 502, 'api_error', 'the gateway answered with no body');
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...failureHeaders(fails) });
    try {
      for await (const ev of relayMessages(upstream.body, model)) res.write(ev);
    } finally {
      res.end();
    }
    return;
  }
  const found: unknown = await upstream.json();
  replyJson(res, 200, toMessageBody(isObj(found) ? found : {}, model), failureHeaders(fails));
}

async function dispatch(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    serveModels(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serveChat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serveMessages(req, res);
    return;
  }
  replyError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

/** Tell the env our port and catalog; it answers with the prompt count. */
async function announce(port: number, doc: CatalogDoc): Promise<void> {
  const res = await fetch(`${envBase}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: WIRE_VERSION, port, catalog: doc }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${envBase}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isObj(reply) && typeof reply.requests === 'number') say(`registered: the trial holds ${reply.requests} prompts`);
}

/** Ask the env for work until it says done, then stop. */
function watchWork(server: Server): void {
  let busy = false;
  const timer = setInterval(() => {
    if (busy) return;
    busy = true;
    void (async () => {
      try {
        const res = await fetch(`${envBase}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isObj(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // Not reachable yet; the next tick tries again.
      } finally {
        busy = false;
      }
    })();
  }, WORK_POLL_MS);
}

async function run(): Promise<void> {
  if (brief.problem !== null) say(brief.problem);
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (envBase === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (gatewayChatUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void dispatch(req, res).catch((err: unknown) => {
      say((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        replyError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
  const addr = server.address();
  if (addr === null || typeof addr === 'string') throw new Error('the server has no port');
  if (addr.port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  say(`listening on 0.0.0.0:${addr.port} with ${brief.models.filter((m) => m.free).length} free models`);
  await announce(addr.port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  watchWork(server);
}

try {
  await run();
} catch (err) {
  say((err as Error).stack ?? String(err));
  process.exit(1);
}
