// The client wires and the one upstream wire this baseline speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This baseline forwards both to a
// provider on the openai wire and translates back:
//
//   - the OpenAI path passes the client's body (with the routed model's id) to the provider and streams
//     the provider's JSON or SSE straight back, the way an OpenAI-compatible gateway does;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the provider's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//
// The gemini and anthropic upstream wires are not implemented in this baseline (main.ts answers 501 for a
// provider on either): a real entry needs their request and event shapes too, which is the biggest single
// gap in this starter.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and the
// same provider answer give the same bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const maybeStream = upstream as unknown as { getReader?: unknown } | null;
  if (maybeStream === null || typeof maybeStream.getReader !== 'function') throw new Error('upstream is not a readable stream');
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

/** Translate an OpenAI Chat Completions request into an Anthropic Messages request. */
export function openAItoAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  let systemText = '';
  const srcMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of srcMessages) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    const content = raw.content;
    if (role === 'system') {
      const t = textOf(content);
      if (t !== '') systemText = systemText === '' ? t : `${systemText}\n${t}`;
      continue;
    }
    if (role === 'user') {
      if (typeof content === 'string') {
        messages.push({ role: 'user', content });
      } else if (Array.isArray(content)) {
        const blocks: Json[] = [];
        for (const part of content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') {
            blocks.push({ type: 'text', text: part.text });
          } else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
            const url = part.image_url.url;
            if (url.startsWith('data:')) {
              const comma = url.indexOf(',');
              if (comma !== -1) {
                const header = url.slice(5, comma);
                const data = url.slice(comma + 1);
                const mediaType = header.split(';')[0] || 'image/png';
                blocks.push({ type: 'image', source: { type: 'base64', media_type: mediaType, data } });
              }
            } else {
              blocks.push({ type: 'image', source: { type: 'url', url } });
            }
          }
        }
        messages.push({ role: 'user', content: blocks });
      }
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      if (typeof content === 'string' && content !== '') {
        blocks.push({ type: 'text', text: content });
      } else if (Array.isArray(content)) {
        const t = textOf(content);
        if (t !== '') blocks.push({ type: 'text', text: t });
      }
      const toolCalls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of toolCalls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let inputObj: Json = {};
        try {
          const parsed = JSON.parse(typeof call.function.arguments === 'string' ? call.function.arguments : '{}');
          if (isRecord(parsed)) inputObj = parsed;
        } catch {}
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : 'toolu_0',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: inputObj,
        });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : [{ type: 'text', text: '' }] });
      continue;
    }
    if (role === 'tool') {
      const toolUseId = typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '';
      const text = typeof content === 'string' ? content : JSON.stringify(content);
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: toolUseId,
            content: text,
          },
        ],
      });
      continue;
    }
  }

  const out: Json = {
    model,
    messages: messages.length > 0 ? messages : [{ role: 'user', content: '' }],
    max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : 4096,
  };
  if (systemText !== '') out.system = systemText;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    out.tools = body.tools.map((t) => {
      if (!isRecord(t) || !isRecord(t.function)) return t;
      const fn = t.function;
      return {
        name: fn.name,
        description: fn.description,
        input_schema: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} },
      };
    });
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** Translate an OpenAI Chat Completions request into a Gemini API request. */
export function openAItoGeminiRequest(body: Json, _model: string): Json {
  const contents: Json[] = [];
  let systemInstruction: Json | undefined = undefined;
  let systemText = '';
  const srcMessages = Array.isArray(body.messages) ? body.messages : [];
  /** Tool-call id -> function name, from the assistant messages: a tool message only carries the
   * call id, and Gemini's functionResponse wants the function's name, not the id. */
  const toolNames = new Map<string, string>();
  for (const raw of srcMessages) {
    if (!isRecord(raw) || !Array.isArray(raw.tool_calls)) continue;
    for (const call of raw.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      if (typeof call.id === 'string' && typeof call.function.name === 'string') toolNames.set(call.id, call.function.name);
    }
  }

  for (const raw of srcMessages) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    const content = raw.content;
    if (role === 'system') {
      const t = textOf(content);
      if (t !== '') systemText = systemText === '' ? t : `${systemText}\n${t}`;
      continue;
    }
    if (role === 'user') {
      const parts: Json[] = [];
      if (typeof content === 'string') {
        parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const part of content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') {
            parts.push({ text: part.text });
          } else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
            const url = part.image_url.url;
            if (url.startsWith('data:')) {
              const comma = url.indexOf(',');
              if (comma !== -1) {
                const header = url.slice(5, comma);
                const data = url.slice(comma + 1);
                const mimeType = header.split(';')[0] || 'image/png';
                parts.push({ inlineData: { mimeType, data } });
              }
            } else {
              parts.push({ fileData: { fileUri: url, mimeType: 'image/png' } });
            }
          }
        }
      }
      contents.push({ role: 'user', parts: parts.length > 0 ? parts : [{ text: '' }] });
      continue;
    }
    if (role === 'assistant') {
      const parts: Json[] = [];
      if (typeof content === 'string' && content !== '') {
        parts.push({ text: content });
      } else if (Array.isArray(content)) {
        const t = textOf(content);
        if (t !== '') parts.push({ text: t });
      }
      const toolCalls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of toolCalls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let args: Json = {};
        try {
          const parsed = JSON.parse(typeof call.function.arguments === 'string' ? call.function.arguments : '{}');
          if (isRecord(parsed)) args = parsed;
        } catch {}
        parts.push({
          functionCall: {
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args,
          },
        });
      }
      contents.push({ role: 'model', parts: parts.length > 0 ? parts : [{ text: '' }] });
      continue;
    }
    if (role === 'tool') {
      const toolCallId = typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '';
      let respObj: Json = { result: typeof content === 'string' ? content : JSON.stringify(content) };
      try {
        const parsed = JSON.parse(typeof content === 'string' ? content : JSON.stringify(content));
        if (isRecord(parsed)) respObj = parsed;
      } catch {}
      contents.push({
        role: 'function',
        parts: [
          {
            functionResponse: {
              name: toolNames.get(toolCallId) ?? (toolCallId || 'function'),
              response: respObj,
            },
          },
        ],
      });
      continue;
    }
  }

  if (systemText !== '') {
    systemInstruction = { parts: [{ text: systemText }] };
  }

  const out: Json = { contents };
  if (systemInstruction) out.systemInstruction = systemInstruction;

  const generationConfig: Json = {};
  if (typeof body.temperature === 'number') generationConfig.temperature = body.temperature;
  if (typeof body.top_p === 'number') generationConfig.topP = body.top_p;
  if (typeof body.max_tokens === 'number') generationConfig.maxOutputTokens = body.max_tokens;
  if (Array.isArray(body.stop) && body.stop.length > 0) generationConfig.stopSequences = body.stop;
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    generationConfig.responseMimeType = 'application/json';
    if (isRecord(body.response_format.json_schema)) generationConfig.responseSchema = body.response_format.json_schema;
  }
  if (Object.keys(generationConfig).length > 0) out.generationConfig = generationConfig;

  if (Array.isArray(body.tools) && body.tools.length > 0) {
    const decls = body.tools.flatMap((t) => {
      if (!isRecord(t) || !isRecord(t.function)) return [];
      const fn = t.function;
      return [
        {
          name: fn.name,
          description: fn.description || '',
          parameters: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (decls.length > 0) {
      out.tools = [{ functionDeclarations: decls }];
    }
  }

  return out;
}

/** Translate a Gemini generateContent response into an OpenAI Chat Completions response. */
export function geminiToOpenAI(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];

  let textContent = '';
  const toolCalls: Json[] = [];

  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') {
      textContent = textContent === '' ? part.text : `${textContent}\n${part.text}`;
    }
    if (isRecord(part.functionCall)) {
      const fc = part.functionCall;
      const name = typeof fc.name === 'string' ? fc.name : '';
      const args = isRecord(fc.args) ? fc.args : {};
      toolCalls.push({
        id: `call_${toolCalls.length}_${name}`,
        type: 'function',
        function: {
          name,
          arguments: JSON.stringify(args),
        },
      });
    }
  }

  const finishReasonMap: Record<string, string> = {
    STOP: 'stop',
    MAX_TOKENS: 'length',
    SAFETY: 'content_filter',
    RECITATION: 'content_filter',
  };
  const rawFinish = typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP';
  const finish_reason = toolCalls.length > 0 ? 'tool_calls' : (finishReasonMap[rawFinish] ?? 'stop');

  const usageMeta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const promptTokens = numberOrZero(usageMeta.promptTokenCount);
  const completionTokens = numberOrZero(usageMeta.candidatesTokenCount);

  return {
    id: 'chatcmpl-gemini',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: textContent !== '' || toolCalls.length === 0 ? textContent : null,
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        },
        finish_reason,
      },
    ],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens,
    },
  };
}

/** Translate Gemini SSE stream chunks into OpenAI SSE stream chunks. */
export async function* geminiEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const maybeStream = upstream as unknown as { getReader?: unknown } | null;
  if (maybeStream === null || typeof maybeStream.getReader !== 'function') throw new Error('upstream is not a readable stream');
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  const sendChunk = (delta: Json, finishReason: string | null = null, usage: Json | null = null) => {
    const chunk: Json = {
      id: 'chatcmpl-gemini-stream',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finishReason }],
    };
    if (usage) chunk.usage = usage;
    return `data: ${JSON.stringify(chunk)}\n\n`;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        const dataLine = frame.split('\n').find((l) => l.startsWith('data: '));
        if (dataLine) {
          try {
            const parsed = JSON.parse(dataLine.slice(6));
            if (isRecord(parsed)) {
              const candidates = Array.isArray(parsed.candidates) ? parsed.candidates : [];
              const cand = isRecord(candidates[0]) ? candidates[0] : {};
              const cont = isRecord(cand.content) ? cand.content : {};
              const parts = Array.isArray(cont.parts) ? cont.parts : [];
              let textChunk = '';
              const tCalls: Json[] = [];
              for (const p of parts) {
                if (!isRecord(p)) continue;
                if (typeof p.text === 'string') textChunk += p.text;
                if (isRecord(p.functionCall)) {
                  const fc = p.functionCall;
                  const name = typeof fc.name === 'string' ? fc.name : '';
                  const args = isRecord(fc.args) ? fc.args : {};
                  tCalls.push({
                    index: tCalls.length,
                    id: `call_${tCalls.length}_${name}`,
                    type: 'function',
                    function: { name, arguments: JSON.stringify(args) },
                  });
                }
              }
              const delta: Json = {};
              if (textChunk !== '') delta.content = textChunk;
              if (tCalls.length > 0) delta.tool_calls = tCalls;
              const rawFinish = typeof cand.finishReason === 'string' ? cand.finishReason : null;
              const finishReason = tCalls.length > 0 ? 'tool_calls' : (rawFinish === 'STOP' ? 'stop' : rawFinish === 'MAX_TOKENS' ? 'length' : null);
              const usageMeta = isRecord(parsed.usageMetadata) ? parsed.usageMetadata : null;
              let usage: Json | null = null;
              if (usageMeta) {
                const pt = numberOrZero(usageMeta.promptTokenCount);
                const ct = numberOrZero(usageMeta.candidatesTokenCount);
                usage = { prompt_tokens: pt, completion_tokens: ct, total_tokens: pt + ct };
              }
              if (Object.keys(delta).length > 0 || finishReason !== null || usage !== null) {
                yield sendChunk(delta, finishReason, usage);
              }
            }
          } catch {}
        }
        at = buffer.indexOf('\n\n');
      }
    }
  } finally {
    reader.releaseLock();
  }
  yield 'data: [DONE]\n\n';
}

/** Translate Anthropic Messages response into an OpenAI Chat Completions response. */
export function anthropicResponseToOpenAI(body: Json, model: string): Json {
  const contentBlocks = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: Json[] = [];
  for (const block of contentBlocks) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') {
      text += block.text;
    } else if (block.type === 'tool_use') {
      const id = typeof block.id === 'string' ? block.id : 'toolu_0';
      const name = typeof block.name === 'string' ? block.name : '';
      const input = isRecord(block.input) ? block.input : {};
      toolCalls.push({
        id,
        type: 'function',
        function: {
          name,
          arguments: JSON.stringify(input),
        },
      });
    }
  }

  const stopReasonMap: Record<string, string> = {
    end_turn: 'stop',
    max_tokens: 'length',
    tool_use: 'tool_calls',
    refusal: 'content_filter',
  };
  const rawStop = typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn';
  const finish_reason = toolCalls.length > 0 ? 'tool_calls' : (stopReasonMap[rawStop] ?? 'stop');

  const usage = isRecord(body.usage) ? body.usage : {};
  const promptTokens = numberOrZero(usage.input_tokens);
  const completionTokens = numberOrZero(usage.output_tokens);

  return {
    id: typeof body.id === 'string' ? body.id : 'chatcmpl-anthropic',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: text !== '' || toolCalls.length === 0 ? text : null,
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        },
        finish_reason,
      },
    ],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens,
    },
  };
}

/** Translate Anthropic SSE stream into OpenAI SSE stream. */
export async function* anthropicEventsToOpenAIStream(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const maybeStream = upstream as unknown as { getReader?: unknown } | null;
  if (maybeStream === null || typeof maybeStream.getReader !== 'function') throw new Error('upstream is not a readable stream');
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let inputTokens = 0;
  /** Tool_use blocks seen in content_block_start, by Anthropic block index: their id, name, the OpenAI tool index they map to, and whether the first delta went out. */
  const toolBlocks = new Map<number, { id: string; name: string; openaiIndex: number; started: boolean }>();
  let nextToolIndex = 0;

  const sendChunk = (delta: Json, finishReason: string | null = null, usage: Json | null = null) => {
    const chunk: Json = {
      id: 'chatcmpl-stream',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finishReason }],
    };
    if (usage) chunk.usage = usage;
    return `data: ${JSON.stringify(chunk)}\n\n`;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        const lines = frame.split('\n');
        const dataLine = lines.find((l) => l.startsWith('data: '));
        if (dataLine) {
          try {
            const parsed = JSON.parse(dataLine.slice(6));
            if (isRecord(parsed)) {
              const type = parsed.type;
              if (type === 'message_start') {
                const msg = isRecord(parsed.message) ? parsed.message : {};
                const usageObj = isRecord(msg.usage) ? msg.usage : {};
                inputTokens = numberOrZero(usageObj.input_tokens);
              } else if (type === 'content_block_start') {
                // A tool_use block names the tool call; remember it so the deltas that follow carry
                // its id, name and a stable OpenAI tool index instead of an anonymous index 0.
                const blockIndex = typeof parsed.index === 'number' ? parsed.index : -1;
                const block = isRecord(parsed.content_block) ? parsed.content_block : {};
                if (blockIndex >= 0 && block.type === 'tool_use') {
                  toolBlocks.set(blockIndex, {
                    id: typeof block.id === 'string' && block.id !== '' ? block.id : `call_${blockIndex}`,
                    name: typeof block.name === 'string' ? block.name : '',
                    openaiIndex: nextToolIndex,
                    started: false,
                  });
                  nextToolIndex += 1;
                }
              } else if (type === 'content_block_delta') {
                const deltaObj = isRecord(parsed.delta) ? parsed.delta : {};
                if (deltaObj.type === 'text_delta' && typeof deltaObj.text === 'string') {
                  yield sendChunk({ content: deltaObj.text });
                } else if (deltaObj.type === 'input_json_delta' && typeof deltaObj.partial_json === 'string') {
                  const blockIndex = typeof parsed.index === 'number' ? parsed.index : -1;
                  let block = blockIndex >= 0 ? toolBlocks.get(blockIndex) : undefined;
                  if (block === undefined) {
                    // No content_block_start arrived for this block: synthesize one so the delta
                    // still carries an id and a name slot instead of vanishing into index 0.
                    block = { id: `call_${blockIndex}`, name: '', openaiIndex: nextToolIndex, started: false };
                    nextToolIndex += 1;
                    if (blockIndex >= 0) toolBlocks.set(blockIndex, block);
                  }
                  if (!block.started) {
                    block.started = true;
                    yield sendChunk({ tool_calls: [{ index: block.openaiIndex, id: block.id, type: 'function', function: { name: block.name, arguments: deltaObj.partial_json } }] });
                  } else {
                    yield sendChunk({ tool_calls: [{ index: block.openaiIndex, function: { arguments: deltaObj.partial_json } }] });
                  }
                }
              } else if (type === 'message_delta') {
                const deltaObj = isRecord(parsed.delta) ? parsed.delta : {};
                const stopReason = typeof deltaObj.stop_reason === 'string' ? deltaObj.stop_reason : 'end_turn';
                const mappedStop = stopReason === 'max_tokens' ? 'length' : stopReason === 'tool_use' ? 'tool_calls' : stopReason === 'refusal' ? 'content_filter' : 'stop';
                const usageObj = isRecord(parsed.usage) ? parsed.usage : {};
                const outputTokens = numberOrZero(usageObj.usage?.output_tokens ?? usageObj.output_tokens);
                yield sendChunk({}, mappedStop, { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens });
              }
            }
          } catch {}
        }
        at = buffer.indexOf('\n\n');
      }
    }
  } finally {
    reader.releaseLock();
  }
  yield 'data: [DONE]\n\n';
}
