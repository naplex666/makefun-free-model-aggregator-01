import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicEventsToOpenAIStream,
  anthropicModelList,
  anthropicResponseToOpenAI,
  anthropicToOpenAI,
  geminiEventsFromOpenAI,
  geminiToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  openAItoAnthropicRequest,
  openAItoGeminiRequest,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;
const exhaustedProviders = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Route {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

function getCandidateRoutes(requires: Requirement): Route[] {
  const candidates: Route[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (exhaustedProviders.has(probe.provider.id)) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const rating = row?.rating_milli ?? 0;
      candidates.push({ probe, model, rating });
    }
  }
  candidates.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : 1;
  });
  return candidates;
}

function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    size += buf.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buf);
  }
  try {
    const parsed: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

async function callProviderAndTranslate(
  route: Route,
  body: Json,
  isAnthropicClient: boolean
): Promise<{ upstream: Response; translatedBody: Json; wire: string } | null> {
  const base = joinBase(ENV_URL, route.probe.provider.base);
  const wire = route.probe.provider.wire;
  const auth = route.probe.provider.auth;
  const key = route.probe.provider.key;

  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(auth, key),
  };
  if (wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (body.stream === true) headers.accept = 'text/event-stream';

  let url = '';
  let upstreamBody: Json = body;

  if (isAnthropicClient) {
    // Client sent Anthropic /v1/messages request
    if (wire === 'anthropic') {
      url = `${base}/v1/messages`;
      upstreamBody = { ...body, model: route.model.id };
    } else if (wire === 'openai') {
      url = `${base}/v1/chat/completions`;
      upstreamBody = anthropicToOpenAI(body, route.model.id);
    } else if (wire === 'gemini') {
      const openAiReq = anthropicToOpenAI(body, route.model.id);
      url = `${base}/v1beta/models/${route.model.id}:${body.stream === true ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
      upstreamBody = openAItoGeminiRequest(openAiReq, route.model.id);
      if (auth === 'none' && key !== '') url += `?key=${key}`;
    }
  } else {
    // Client sent OpenAI /v1/chat/completions request
    if (wire === 'openai') {
      url = `${base}/v1/chat/completions`;
      upstreamBody = { ...body, model: route.model.id };
    } else if (wire === 'anthropic') {
      url = `${base}/v1/messages`;
      upstreamBody = openAItoAnthropicRequest(body, route.model.id);
    } else if (wire === 'gemini') {
      url = `${base}/v1beta/models/${route.model.id}:${body.stream === true ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
      upstreamBody = openAItoGeminiRequest(body, route.model.id);
      if (auth === 'none' && key !== '') url += `?key=${key}`;
    }
  }

  try {
    const upstream = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody) });
    if (upstream.status === 402 || upstream.status === 429 || upstream.status >= 500) {
      exhaustedProviders.add(route.probe.provider.id);
      return null;
    }
    return { upstream, translatedBody: upstreamBody, wire };
  } catch {
    exhaustedProviders.add(route.probe.provider.id);
    return null;
  }
}

async function handleChatOrMessages(req: IncomingMessage, res: ServerResponse, isAnthropicClient: boolean): Promise<void> {
  const body = await readJson(req);
  const wireType = isAnthropicClient ? 'anthropic' : 'openai';
  if (body === null) return sendError(res, wireType, 400, 'invalid_request_error', 'the request body is not one JSON object');

  const requires = requirementsFor(body);
  const requestedModel = typeof body.model === 'string' && body.model !== '' ? body.model : 'default';

  let successResult: { upstream: Response; wire: string; route: Route } | null = null;

  while (true) {
    const candidates = getCandidateRoutes(requires);
    if (candidates.length === 0) break;
    const route = candidates[0];
    const result = await callProviderAndTranslate(route, body, isAnthropicClient);
    if (result !== null) {
      successResult = { upstream: result.upstream, wire: result.wire, route };
      break;
    }
  }

  if (successResult === null) {
    return sendError(res, wireType, 503, isAnthropicClient ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }

  const { upstream, wire, route } = successResult;
  const isStreaming = body.stream === true;
  const routedModelId = route.model.id;

  if (isAnthropicClient) {
    if (isStreaming) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      if (wire === 'anthropic' && upstream.body !== null) {
        const reader = upstream.body.getReader();
        try {
          for (;;) {
            const { value, done } = await reader.read();
            if (done) break;
            res.write(Buffer.from(value));
          }
        } finally {
          reader.releaseLock();
        }
      } else if (wire === 'openai' && upstream.body !== null) {
        for await (const event of anthropicEventsFromOpenAI(upstream.body, requestedModel)) res.write(event);
      } else if (wire === 'gemini' && upstream.body !== null) {
        // Translate Gemini SSE to Anthropic SSE via OpenAI
        const openaiStream = (async function* () {
          for await (const chunk of geminiEventsFromOpenAI(upstream.body, routedModelId)) yield new TextEncoder().encode(chunk);
        })() as unknown as ReadableStream<Uint8Array>;
        for await (const event of anthropicEventsFromOpenAI(openaiStream, requestedModel)) res.write(event);
      }
      res.end();
      return;
    }

    // Non-streaming Anthropic client
    let jsonAns: Json = {};
    try {
      const parsed = await upstream.json();
      if (isRecord(parsed)) jsonAns = parsed;
    } catch {}

    let anthropicResponse: Json = {};
    if (wire === 'anthropic') {
      anthropicResponse = { ...jsonAns, model: requestedModel };
    } else if (wire === 'openai') {
      anthropicResponse = openAIToAnthropic(jsonAns, requestedModel);
    } else if (wire === 'gemini') {
      const openAiAns = geminiToOpenAI(jsonAns, routedModelId);
      anthropicResponse = openAIToAnthropic(openAiAns, requestedModel);
    }
    sendJson(res, 200, anthropicResponse);
    return;
  } else {
    // OpenAI client
    if (isStreaming) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      if (wire === 'openai' && upstream.body !== null) {
        const reader = upstream.body.getReader();
        try {
          for (;;) {
            const { value, done } = await reader.read();
            if (done) break;
            res.write(Buffer.from(value));
          }
        } finally {
          reader.releaseLock();
        }
      } else if (wire === 'anthropic' && upstream.body !== null) {
        for await (const event of anthropicEventsToOpenAIStream(upstream.body, routedModelId)) res.write(event);
      } else if (wire === 'gemini' && upstream.body !== null) {
        for await (const event of geminiEventsFromOpenAI(upstream.body, routedModelId)) res.write(event);
      }
      res.end();
      return;
    }

    // Non-streaming OpenAI client
    let jsonAns: Json = {};
    try {
      const parsed = await upstream.json();
      if (isRecord(parsed)) jsonAns = parsed;
    } catch {}

    let openaiResponse: Json = {};
    if (wire === 'openai') {
      openaiResponse = { ...jsonAns, model: routedModelId };
    } else if (wire === 'anthropic') {
      openaiResponse = anthropicResponseToOpenAI(jsonAns, routedModelId);
    } else if (wire === 'gemini') {
      openaiResponse = geminiToOpenAI(jsonAns, routedModelId);
    }
    sendJson(res, upstream.status, openaiResponse);
    return;
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await handleChatOrMessages(req, res, false);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await handleChatOrMessages(req, res, true);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      server.removeListener('error', reject);
      resolve();
    });
  });
}

async function register(port: number): Promise<void> {
  if (ENV_URL === '') return;
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
  });
  if (!res.ok) throw new Error(`POST /register answered HTTP ${res.status}`);
}

async function pollWork(): Promise<void> {
  if (ENV_URL === '') return;
  for (;;) {
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
    try {
      const res = await fetch(`${ENV_URL}/work`);
      if (!res.ok) continue;
      const body: unknown = await res.json();
      if (isRecord(body) && body.done === true) break;
    } catch {}
  }
}

async function main(): void {
  await load();
  if (process.argv.includes('--list')) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    process.exit(0);
  }
  if (ENV_URL === '') {
    process.stdout.write('{"providers":[]}\n');
    process.exit(0);
  }
  const server = createServer((req, res) => {
    handle(req, res).catch((err) => {
      log(`request failed: ${(err as Error).message}`);
      sendError(res, 'openai', 500, 'internal_error', (err as Error).message);
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('server failed to bind');
  try {
    await register(address.port);
  } catch (err) {
    log(`registration failed: ${(err as Error).message}`);
    process.exit(1);
  }
  await pollWork();
  server.close();
  process.exit(0);
}

main().catch((err) => {
  log(`fatal: ${(err as Error).stack ?? (err as Error).message}`);
  process.exit(1);
});
