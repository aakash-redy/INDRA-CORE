import express from 'express';
import type { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { createClient } from '@supabase/supabase-js';
import rateLimit from 'express-rate-limit';
import dotenv from 'dotenv';
import helmet from 'helmet';
import crypto from 'crypto';

dotenv.config();

// ============================================================================
// ── 1. CONFIGURATION & STARTUP CHECKS
// ============================================================================

const requiredEnvVars = [
  // 'SUPABASE_URL',        // AUTH BYPASSED — not needed for dev
  // 'SUPABASE_ANON_KEY',   // AUTH BYPASSED — not needed for dev
  'GEMINI_API_KEY',
  'GEMINI_RERANK_API_KEY',
  'NOMIC_API_KEY',
  // 'API_AUTH_TOKEN',      // AUTH BYPASSED — skipping token check
  'OPENROUTER_API_KEY_1',
  'OPENROUTER_API_KEY_2',
  'COHERE_API_KEY',
];

requiredEnvVars.forEach(v => {
  if (!process.env[v]) throw new Error(`🚨 CRITICAL: Missing ${v} in .env — server cannot start.`);
});

const app = express();
app.set('trust proxy', 1);

const PORT = process.env.PORT || 8000;
const IS_PROD = process.env.NODE_ENV === 'production';

export const VALID_DOMAINS = [
  'Chassis', 'Braking', 'Powertrain', 'Safety',
  'Aerodynamics', 'Electrical', 'General',
  'Formula Bharat 2027 Full',
] as const;
type ValidDomain = typeof VALID_DOMAINS[number];

// FIX (accuracy):
//  - MATCH_THRESHOLD: tuned from 0.4 → 0.5 for nomic-embed-text-v1.5.
//    The old 0.4 threshold was inherited from the previous Gemini
//    embedding model. Nomic vectors have a tighter similarity
//    distribution, so 0.4 was pulling in marginally related chunks and
//    diluting the model's focus. 0.5 keeps the on-topic chunks and
//    drops the noise.
//  - LEARNED_MATCH_THRESHOLD: dropped 0.75 → 0.7 so the system more
//    often uses its prior good answers (recoveries from user feedback)
//    instead of recomputing from scratch.
//  - CACHE_SIMILARITY_THRESHOLD: dropped 0.97 → 0.94. 0.97 was so
//    tight that the cache was effectively dead — two semantically
//    equivalent phrasings rarely hit 0.97. 0.94 still rejects truly
//    different queries but accepts the "front impact structure specs"
//    vs "specs for the front impact structure" style of reuse.
//  - MATCH_COUNT: bumped 5 → 7 to give the reranker more signal.
const CONFIG = {
  MATCH_THRESHOLD: 0.5,
  LEARNED_MATCH_THRESHOLD: 0.7,
  MATCH_COUNT: 7,
  LEARNED_MATCH_COUNT: 3,
  CACHE_TTL_MS: 60 * 60 * 1000,
  CACHE_SIMILARITY_THRESHOLD: 0.94,
  CACHE_MAX_ENTRIES: 500,
  MAX_MESSAGE_LENGTH: 1000,
  MIN_MESSAGE_LENGTH: 2,
  RATE_LIMIT_WINDOW_MS: 60 * 1000,
  RATE_LIMIT_MAX: IS_PROD ? 15 : 50,
  ASK_RATE_LIMIT_MAX: IS_PROD ? 10 : 30,
  BODY_SIZE_LIMIT: '10kb',
  MODEL_COOLDOWN_MS: 60 * 1000,
  GEMINI_TIMEOUT_MS: 25_000,
  RERANK_CHUNK_PREVIEW: 250,
} as const;

const OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions';
const OPENROUTER_MODEL = process.env.OPENROUTER_MODEL || 'meta-llama/llama-4-scout-17b-16e-instruct';

// ── Supabase client — still used for DB queries (not auth) ──────────────────
const supabase = createClient(
  process.env.SUPABASE_URL ?? 'http://localhost:54321',
  process.env.SUPABASE_SERVICE_KEY ?? process.env.SUPABASE_ANON_KEY ?? 'dev-key',
);

// ============================================================================
// ── 2. TYPES & INTERFACES
// ============================================================================

interface CacheEntry {
  embedding: number[];
  response: Record<string, unknown>;
  expiresAt: number;
}

interface AuthenticatedRequest extends Request {
  requestId: string;
  user: {
    id?: string;
    email?: string;
    role?: string;
  };
}

interface RuleChunk {
  rule_id: string;
  content: string;
  similarity: number;
  rerank_score?: number;
}

interface LearnedChunk {
  question: string;
  answer: string;
  source: string;
}

interface ModelRecord {
  id?: number;
  name?: string;
  category?: string;
  description?: string;
  file_url?: string;
  file_size_mb?: number;
  model_rule_tags?: { rule_id: string }[];
  [key: string]: unknown;
}

interface CadNodeMatch {
  rule_id: string;
  cad_node_name: string;
  relevance_score?: number;
}

interface CadKeywordMatch {
  model_id: number;
  highlight_meshes: string[];
  context_meshes: string[];
}

type QueryIntent = 'dimension' | 'compliance' | 'definition' | 'procedure' | 'general';

// ============================================================================
// ── 3. MODEL ROSTER INTERFACES
// ============================================================================

interface ModelSlot {
  label: string;
  modelName: string;
  client: GoogleGenerativeAI;
  coolUntil: number;
  quotaHits: number;
  successCount: number;
}

interface OpenRouterSlot {
  label: string;
  apiKey: string;
  modelName: string;
  coolUntil: number;
  quotaHits: number;
  successCount: number;
}

// ============================================================================
// ── 4. IN-MEMORY SEMANTIC CACHE & GARBAGE COLLECTOR
// ============================================================================

const semanticCache = new Map<string, CacheEntry>();

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot  += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  const mag = Math.sqrt(magA) * Math.sqrt(magB);
  return mag === 0 ? 0 : dot / mag;
}

function findCacheHit(embedding: number[], domain: string): Record<string, unknown> | null {
  const now = Date.now();
  for (const [key, entry] of semanticCache.entries()) {
    if (entry.expiresAt < now) { semanticCache.delete(key); continue; }
    if (!key.startsWith(domain + ':')) continue;
    if (cosineSimilarity(embedding, entry.embedding) >= CONFIG.CACHE_SIMILARITY_THRESHOLD) {
      return entry.response;
    }
  }
  return null;
}

// FIX (accuracy): MD5 is cryptographically broken and a poor cache key choice.
// Replaced with SHA-256, and the cache key now includes a domain prefix
// plus a short fingerprint so collisions are virtually impossible.
function writeCache(embedding: number[], domain: string, response: Record<string, unknown>): void {
  if (semanticCache.size >= CONFIG.CACHE_MAX_ENTRIES) {
    const oldestKey = semanticCache.keys().next().value;
    if (oldestKey) semanticCache.delete(oldestKey);
    logger.info(`Cache eviction: max entries (${CONFIG.CACHE_MAX_ENTRIES}) reached.`);
  }
  const hash = crypto.createHash('sha256').update(JSON.stringify(embedding)).digest('hex').slice(0, 32);
  const key = `${domain}:${hash}`;
  semanticCache.set(key, { embedding, response, expiresAt: Date.now() + CONFIG.CACHE_TTL_MS });
}

// FIX (accuracy): explicit cache invalidation hook. The old code relied on
// the 60-minute TTL and a 10-minute GC, which meant that after a rulebook
// re-ingestion the API would still serve the OLD cached answers for up to
// an hour. Now the ingestion script can call invalidateCache() via a new
// admin endpoint to flush the cache the moment the new embeddings land.
export function invalidateCache(domain?: string): number {
  if (!domain) {
    const size = semanticCache.size;
    semanticCache.clear();
    return size;
  }
  const prefix = `${domain}:`;
  let removed = 0;
  for (const key of Array.from(semanticCache.keys())) {
    if (key.startsWith(prefix)) {
      semanticCache.delete(key);
      removed++;
    }
  }
  return removed;
}

setInterval(() => {
  const now = Date.now();
  let pruned = 0;
  for (const [key, entry] of semanticCache.entries()) {
    if (entry.expiresAt < now) { semanticCache.delete(key); pruned++; }
  }
  if (pruned > 0) logger.info(`Cache GC pruned ${pruned} expired entries.`);
}, 10 * 60 * 1000);

// ============================================================================
// ── 5. LOGGING & UTILITIES
// ============================================================================

const logger = {
  info: (msg: string, meta?: Record<string, unknown>) =>
    console.log(`[INFO]  ${new Date().toISOString()} — ${msg}`, meta ?? ''),
  warn: (msg: string, meta?: Record<string, unknown>) =>
    console.warn(`[WARN]  ${new Date().toISOString()} — ${msg}`, meta ?? ''),
  error: (msg: string, error: unknown, meta?: Record<string, unknown>) =>
    console.error(`[ERROR] ${new Date().toISOString()} — ${msg}`, {
      message: error instanceof Error ? error.message : String(error),
      stack:   error instanceof Error ? error.stack   : undefined,
      ...meta,
    }),
};

async function saveLog(data: Record<string, unknown>): Promise<void> {
  try {
    const { error } = await supabase.from('sora_logs').insert([data]);
    if (error) logger.error('saveLog DB error', error);
  } catch (err: unknown) {
    logger.error('saveLog threw', err);
  }
}

async function saveLearnedPair(
  question: string,
  answer: string,
  domain: string,
  source = 'user_feedback',
): Promise<void> {
  try {
    const embedding = await embedQuery(question);
    await supabase.from('sora_learned').insert([{ question, answer, domain, embedding, source }]);
  } catch (err) {
    logger.error('saveLearnedPair threw', err);
  }
}

// ============================================================================
// ── 6. CORE AI ENGINE — TIMEOUT WRAPPER
// ============================================================================

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  const timeout = new Promise<never>((_, reject) =>
    setTimeout(() => reject(new Error(`TIMEOUT: ${label} exceeded ${ms}ms`)), ms)
  );
  return Promise.race([promise, timeout]);
}

// ============================================================================
// ── 7. GEMINI MODEL ROSTER & CIRCUIT BREAKER
// ============================================================================

function buildModelRoster(apiKey: string, keyLabel: string): ModelSlot[] {
  const client = new GoogleGenerativeAI(apiKey);
  const models = [
    'gemini-2.5-flash',
    'gemini-2.0-flash',
    'gemini-2.0-flash-lite',
  ];
  return models.map(modelName => ({
    label: `${keyLabel}/${modelName}`,
    modelName,
    client,
    coolUntil: 0,
    quotaHits: 0,
    successCount: 0,
  }));
}

const primaryRoster: ModelSlot[] = buildModelRoster(process.env.GEMINI_API_KEY!, 'PRIMARY');
const rerankRoster: ModelSlot[]  = buildModelRoster(process.env.GEMINI_RERANK_API_KEY!, 'RERANK');

function activeSlots(roster: ModelSlot[]): ModelSlot[] {
  const now = Date.now();
  const active = roster.filter(s => s.coolUntil <= now);
  return active.length > 0 ? active : roster;
}

function coolSlot(slot: ModelSlot): void {
  slot.quotaHits++;
  slot.coolUntil = Date.now() + CONFIG.MODEL_COOLDOWN_MS;
  logger.warn(`[ROSTER] ${slot.label} quota hit #${slot.quotaHits}. Cooling for ${CONFIG.MODEL_COOLDOWN_MS / 1000}s.`);
}

function isSkippableError(error: unknown): boolean {
  const msg = error instanceof Error ? error.message : String(error);
  const lower = msg.toLowerCase();
  return (
    msg.includes('429')                               ||
    lower.includes('quota')                           ||
    lower.includes('resource_exhausted')              ||
    lower.includes('rate_limit')                      ||
    msg.includes('404')                               ||
    lower.includes('not found')                       ||
    lower.includes('is not supported')                ||
    lower.includes('not supported for generatecontent')
  );
}

async function generateWithRoster(
  prompt: string,
  roster: ModelSlot[],
  requireJson = false,
  temperature = 0.7,
): Promise<string> {
  const slots = activeSlots(roster);
  for (const slot of slots) {
    try {
      const model = slot.client.getGenerativeModel({
        model: slot.modelName,
        generationConfig: {
          temperature,
          ...(requireJson ? { responseMimeType: 'application/json' } : {}),
        },
      });
      const result = await withTimeout(
        model.generateContent(prompt),
        CONFIG.GEMINI_TIMEOUT_MS,
        slot.label,
      );
      const text = result.response.text();
      slot.successCount++;
      logger.info(`[ROSTER] ${slot.label} ✓ (successes: ${slot.successCount})`);
      return text;
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : String(error);
      if (msg.startsWith('TIMEOUT:')) {
        logger.warn(`[ROSTER] ${slot.label} timed out — trying next slot.`);
        continue;
      }
      if (isSkippableError(error)) {
        coolSlot(slot);
        continue;
      }
      throw error;
    }
  }
  throw new Error('QUOTA_EXHAUSTED: All models on this roster are at quota. Try again shortly.');
}

// ============================================================================
// ── 8. OPENROUTER ROSTER
// ============================================================================

function buildOpenRouterRoster(): OpenRouterSlot[] {
  return [
    process.env.OPENROUTER_API_KEY_1,
    process.env.OPENROUTER_API_KEY_2,
  ]
  .filter((key): key is string => Boolean(key))
  .map((apiKey, i) => ({
    label: `OPENROUTER_KEY_${i + 1}`,
    apiKey,
    modelName: OPENROUTER_MODEL,
    coolUntil: 0,
    quotaHits: 0,
    successCount: 0,
  }));
}

const openRouterRoster = buildOpenRouterRoster();

async function generateWithOpenRouterRoster(
  prompt: string,
  requireJson = false,
  temperature = 0.7,
): Promise<string> {
  const now = Date.now();
  const active = openRouterRoster.filter(s => s.coolUntil <= now);
  const slots = active.length > 0 ? active : openRouterRoster;

  for (const slot of slots) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), CONFIG.GEMINI_TIMEOUT_MS);

    try {
      const response = await fetch(OPENROUTER_URL, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${slot.apiKey}`,
          'Content-Type': 'application/json',
          'HTTP-Referer': process.env.OPENROUTER_SITE_URL || 'http://localhost:8000',
          'X-Title': process.env.OPENROUTER_APP_NAME || 'INDRA RAG Backend',
        },
        body: JSON.stringify({
          model: slot.modelName,
          messages: [{ role: 'user', content: prompt }],
          temperature,
          ...(requireJson ? { response_format: { type: 'json_object' } } : {}),
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`OpenRouter HTTP ${response.status}: ${errText}`);
      }

      const data = await response.json() as {
        choices?: { message?: { content?: string } }[];
      };

      const text = data.choices?.[0]?.message?.content ?? '';
      slot.successCount++;
      logger.info(`[OPENROUTER] ${slot.label} ✓ (successes: ${slot.successCount})`);
      return text;
    } catch (err) {
      clearTimeout(timeoutId);
      const msg = err instanceof Error ? err.message : String(err);
      const lower = msg.toLowerCase();

      if (msg.includes('aborted') || lower.includes('timeout')) {
        logger.warn(`[OPENROUTER] ${slot.label} timed out — trying next slot.`);
        continue;
      }

      if (
        msg.includes('429') ||
        msg.includes('402') ||
        msg.includes('502') ||
        msg.includes('503') ||
        lower.includes('rate') ||
        lower.includes('quota') ||
        lower.includes('credits')
      ) {
        slot.quotaHits++;
        slot.coolUntil = Date.now() + CONFIG.MODEL_COOLDOWN_MS;
        logger.warn(`[OPENROUTER] ${slot.label} rate/quota limited — cooling for ${CONFIG.MODEL_COOLDOWN_MS / 1000}s.`);
        continue;
      }
      throw err;
    }
  }
  throw new Error('OPENROUTER_EXHAUSTED: All OpenRouter keys are at rate limit or timed out.');
}

// ============================================================================
// ── 9. UNIFIED GENERATE — OpenRouter → Gemini primary → Gemini rerank key
// ============================================================================

async function generate(
  prompt: string,
  requireJson = false,
  temperature = 0.7,
): Promise<string> {
  try {
    return await generateWithOpenRouterRoster(prompt, requireJson, temperature);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (!msg.startsWith('OPENROUTER_EXHAUSTED')) throw err;
    logger.warn('[ROSTER] OpenRouter exhausted — falling back to Gemini primary.');
  }

  try {
    return await generateWithRoster(prompt, primaryRoster, requireJson, temperature);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (!msg.startsWith('QUOTA_EXHAUSTED')) throw err;
    logger.warn('[ROSTER] Gemini primary exhausted — falling back to Gemini rerank key.');
  }

  return generateWithRoster(prompt, rerankRoster, requireJson, temperature);
}

// ============================================================================
// ── 10. EMBEDDINGS — Nomic Atlas
// ============================================================================

const NOMIC_EMBED_URL = 'https://api-atlas.nomic.ai/v1/embedding/text';

async function embedText(
  text: string,
  taskType: 'search_query' | 'search_document' = 'search_query',
): Promise<number[]> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), CONFIG.GEMINI_TIMEOUT_MS);

  try {
    const response = await fetch(NOMIC_EMBED_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.NOMIC_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'nomic-embed-text-v1.5',
        texts: [text],
        task_type: taskType,
        dimensionality: 768,
      }),
      signal: controller.signal,
    });

    clearTimeout(timeout);

    if (response.status === 429) {
      logger.warn('[EMBED] Nomic rate limit hit — retrying after 10s');
      await new Promise(r => setTimeout(r, 10_000));
      return embedText(text, taskType);
    }

    if (!response.ok) {
      throw new Error(`Nomic API error ${response.status}: ${await response.text()}`);
    }

    const data = await response.json() as { embeddings: number[][] };
    const embedding = data.embeddings?.[0];

    if (!embedding || embedding.length !== 768) {
      throw new Error(`EMBED_EXHAUSTED: Invalid embedding — got ${embedding?.length ?? 0} dims, expected 768`);
    }

    logger.info(`[EMBED] Nomic ✓ (dims: ${embedding.length}, task: ${taskType})`);
    return embedding;

  } catch (err: unknown) {
    clearTimeout(timeout);
    const msg = err instanceof Error ? err.message : String(err);

    if (msg.includes('aborted') || msg.includes('TIMEOUT')) {
      throw new Error('EMBED_TIMEOUT: Nomic embedding timed out. Try again.');
    }
    if (msg.startsWith('EMBED_')) throw err;

    logger.error('[EMBED] Nomic failed', err);
    throw new Error('EMBED_EXHAUSTED: Embedding failed. Try again.');
  }
}

const embedQuery    = (text: string) => embedText(text, 'search_query');
const embedDocument = (text: string) => embedText(text, 'search_document');

async function expandAndAverageEmbedding(query: string): Promise<number[]> {
  return embedQuery(query);
}

void embedDocument;

// ============================================================================
// ── 11. RERANKER — Cohere first, Gemini rerankRoster as fallback
// ============================================================================

async function rerankChunks(query: string, chunks: RuleChunk[]): Promise<RuleChunk[]> {
  if (chunks.length <= 1) return chunks;

  try {
    const response = await fetch('https://api.cohere.com/v2/rerank', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.COHERE_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'rerank-v3.5',
        query,
        documents: chunks.map(c => c.content.slice(0, CONFIG.RERANK_CHUNK_PREVIEW)),
        top_n: chunks.length,
      }),
    });

    if (!response.ok) throw new Error(`Cohere error ${response.status}: ${await response.text()}`);

    const data = await response.json() as {
      results: { index: number; relevance_score: number }[]
    };

    logger.info('[RERANKER] Cohere ✓');
    return data.results.map(r => ({
      ...chunks[r.index],
      rerank_score: r.relevance_score,
    }));

  } catch (err) {
    logger.warn('[RERANKER] Cohere failed — falling back to Gemini reranker.', { error: String(err) });
  }

  const chunkList = chunks
    .map((c, i) => `[${i}] Rule ${c.rule_id}: ${c.content.slice(0, CONFIG.RERANK_CHUNK_PREVIEW)}`)
    .join('\n\n');

  const prompt = `You are a relevance scoring engine for a motorsport regulation assistant.
Score each chunk 0.0–1.0 based on how directly it answers the query.
0.0 = unrelated, 1.0 = directly and completely answers the question.

Query: "${query}"

Chunks:
${chunkList}

Return ONLY a JSON array of ${chunks.length} floats, one score per chunk, in order.
Example for 3 chunks: [0.95, 0.3, 0.87]
No explanation. No markdown. Pure JSON array only.`;

  try {
    const raw     = await generateWithRoster(prompt, rerankRoster, true, 0.1);
    const cleaned = raw.replace(/```json|```/g, '').trim();
    const scores: number[] = JSON.parse(cleaned);

    if (!Array.isArray(scores) || scores.length !== chunks.length) {
      throw new Error(`Score array mismatch: got ${scores.length}, expected ${chunks.length}`);
    }

    logger.info('[RERANKER] Gemini fallback ✓');
    return chunks
      .map((c, i) => ({ ...c, rerank_score: typeof scores[i] === 'number' ? scores[i] : c.similarity }))
      .sort((a, b) => (b.rerank_score ?? 0) - (a.rerank_score ?? 0));

  } catch (err) {
    logger.warn('[RERANKER] Gemini fallback also failed — using cosine order.', { error: String(err) });
    return chunks
      .sort((a, b) => b.similarity - a.similarity)
      .map(c => ({ ...c, rerank_score: c.similarity }));
  }
}

// ============================================================================
// ── 12. PROMPT INJECTION GUARD
// ============================================================================

const INJECTION_PATTERNS: RegExp[] = [
  /ignore (all )?(previous|prior|above|preceding) (instructions|prompts?|directives|rules)/i,
  /you are now/i,
  /new (system )?prompt/i,
  /forget (everything|all|prior|previous)/i,
  /act as (a |an )?(?!engineer|assistant|regulations|technical)/i,
  /disregard (all |your )?(previous |prior )?instructions/i,
  /override (your )?(instructions|rules|guidelines)/i,
  /\[system\]/i,
  /<\/?system>/i,
  /<\/?(human|assistant|user|tool|function_call)\s*>/i,
  /prompt injection/i,
  /jailbreak/i,
  /do anything now/i,
  /\bdan\b.{0,40}\bmode\b/i,
  /reveal (your|the) (system|hidden|internal) (prompt|instructions|rules)/i,
  /show (your|the) (system|initial|original) prompt/i,
  /print (your|the) (system|hidden) (prompt|message)/i,
  /repeat (the )?(words|text) (above|before)/i,
  /from now on (you|ignore|disregard|answer|respond)/i,
  /translate (this|the following).{0,40}into (a |an )?(jailbreak|system prompt|instruction)/i,
  /developer mode/i,
  /sudo mode/i,
  /\bpretend (you are|to be|you're)\b/i,
  /\bno (rules|restrictions|limitations|filter)\b/i,
];

function normalizeHomoglyphs(s: string): string {
  return s
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[\u0430]/g, 'a').replace(/[\u0435]/g, 'e').replace(/[\u043E]/g, 'o')
    .replace(/[\u0440]/g, 'p').replace(/[\u0441]/g, 'c').replace(/[\u0443]/g, 'y')
    .replace(/[\u0445]/g, 'x').replace(/[\u0410]/g, 'A').replace(/[\u0415]/g, 'E')
    .replace(/[\u041E]/g, 'O').replace(/[\u0420]/g, 'P').replace(/[\u0421]/g, 'C')
    .replace(/[\u0422]/g, 'T').replace(/[\u0412]/g, 'B').replace(/[\u041A]/g, 'K')
    .replace(/[\u041C]/g, 'M').replace(/[\u041D]/g, 'H').replace(/[\u0391]/g, 'A')
    .replace(/[\u0392]/g, 'B').replace(/[\u0395]/g, 'E').replace(/[\u0396]/g, 'Z')
    .replace(/[\u0397]/g, 'H').replace(/[\u0399]/g, 'I').replace(/[\u039A]/g, 'K')
    .replace(/[\u039C]/g, 'M').replace(/[\u039D]/g, 'N').replace(/[\u039F]/g, 'O')
    .replace(/[\u03A1]/g, 'P').replace(/[\u03A4]/g, 'T').replace(/[\u03A5]/g, 'Y')
    .replace(/[\u03A7]/g, 'X');
}

function hasHiddenBlob(s: string): boolean {
  if (/\b[A-Za-z0-9+/]{60,}={0,2}\b/.test(s)) return true;
  if (/\b[0-9a-fA-F]{80,}\b/.test(s)) return true;
  return false;
}

function detectInjection(text: string): boolean {
  if (!text) return false;
  const stripped = text.replace(/[\u200B-\u200F\u2028-\u202F\uFEFF\u0000-\u001F]/g, ' ');

  if (INJECTION_PATTERNS.some(p => p.test(stripped))) return true;

  const norm = normalizeHomoglyphs(stripped);
  if (norm !== stripped && INJECTION_PATTERNS.some(p => p.test(norm))) return true;

  if (hasHiddenBlob(stripped)) return true;

  const meaningful = stripped.replace(/\s+/g, '');
  if (meaningful.length > 30) {
    const nonAlnum = (meaningful.match(/[^A-Za-z0-9]/g) ?? []).length;
    if (nonAlnum / meaningful.length > 0.35) return true;
  }

  return false;
}

// ============================================================================
// ── 13. INTENT CLASSIFICATION & PROMPT BUILDING
// ============================================================================

function classifyIntent(query: string): QueryIntent {
  const q = query.toLowerCase().trim();

  if (/\b(legal|illegal|allowed|not allowed|permitted|prohibited|banned|forbidden|pass|fail|comply|compliant|non[- ]?compliant|violation|violate|can i|may i|is it ok|is .+ allowed|is .+ legal|is .+ permitted|is .+ prohibited|will .+ pass|will .+ fail|am i allowed)\b/.test(q)) {
    return 'compliance';
  }

  if (/\b(how to|steps|step[- ]by[- ]step|procedure|process|install|mount|attach|assemble|disassemble|replace|remove|test|inspect|check|verify|calibrate|tighten|torque|weld|braze|solder|connect|wire|route|run|route .+ through)\b/.test(q)) {
    return 'procedure';
  }

  if (/^(\s*)(what is|what's|whats|define|definition of|meaning of|what does .+ mean|what do you mean by|explain what|describe what)\b/.test(q)) {
    return 'definition';
  }

  if (/\b\d+\s*(mm|cm|m|inch|in|ft)\b/.test(q)) return 'dimension';
  if (/\b(how (wide|tall|long|thick|deep|small|large|big)|dimension|dimensional|size|length|width|height|weight|distance|radius|diameter|thickness|volume|area|cross[- ]section|clearance|gap|spacing)\b/.test(q)) {
    return 'dimension';
  }

  return 'general';
}

function buildSystemPrompt(intent: QueryIntent, ruleContext: string, query: string, domain: string): string {
  const persona = `You are INDRA — the Integrated Neural Design and Regulations Assistant for Formula Student Racing Teams.

CORE DIRECTIVE: Every answer MUST cite the exact Rule ID first, then give the answer. Never give an answer without its rule reference. If the context contains the answer, you will find it — look carefully before saying you cannot answer.

CITATION FORMAT: Always inline cite as [T3.14] or [EV.5.2] immediately after the relevant statement. Lead with the rule, not a preamble.`;

  const formatRules = `
OUTPUT FORMAT (strictly follow):
- **First line**: Bold one-sentence direct answer + the primary rule ID inline. Example: **The minimum cockpit opening width is 300 mm [T2.4.1].**
- Then expand with supporting details, each sentence citing its rule ID.
- Use ## headers only if the answer genuinely spans multiple distinct sub-topics.
- Bullet points for lists of constraints or steps.
- NEVER say "Based on the provided context", "According to the regulations", or "I cannot find". Just answer from the rules given.
- If multiple rules apply, cite all of them.
- End with ⚡ INDRA NOTE only if there is a critical compliance trap or scrutineering tip.
- Zero filler. Every word earns its place.`;

  const intentInstructions: Record<QueryIntent, string> = {
    dimension: `${persona}\n${formatRules}\n\nINTENT: DIMENSION QUERY
Lead with the exact numeric value and its rule ID in bold. Then list all related dimensional constraints as bullets, each with its rule ID.`,

    compliance: `${persona}\n${formatRules}\n\nINTENT: COMPLIANCE CHECK
Lead with: ✅ COMPLIANT / ❌ NON-COMPLIANT / ⚠️ CONDITIONAL — bold, with the deciding rule ID inline.
Then state what the rule requires and how the scenario maps to it.`,

    definition: `${persona}\n${formatRules}\n\nINTENT: DEFINITION
Lead with the term, its one-sentence definition, and the rule ID that defines it — all in the first line.
Then give functional context from the rules.`,

    procedure: `${persona}\n${formatRules}\n\nINTENT: PROCEDURE
Numbered steps, each citing the relevant rule ID inline. Finish with mandatory checkpoints.`,

    general: `${persona}\n${formatRules}\n\nINTENT: GENERAL QUERY
Answer directly. Cite rule IDs inline throughout. Use ## headers only for genuinely separate sub-topics.`,
  };

  return `${intentInstructions[intent]}

DOMAIN: ${domain}

REGULATION CONTEXT (these are the exact rules — answer from them directly):
${ruleContext}

QUESTION: ${query}

Answer now. Start with the rule ID and the answer. No preamble.`;
}

function extractKeywordsFromQuery(query: string): string[] {
  const keywordsMap: Record<string, string[]> = {
    'brake': ['Braking'],        'pedal': ['Braking'],
    'roll hoop': ['Safety', 'Chassis'], 'bulkhead': ['Chassis'],
    'impact attenuator': ['Chassis'],   'aip': ['Chassis'],
    'accumulator': ['Powertrain'], 'battery': ['Powertrain'], 'motor': ['Powertrain'],
    'shutdown': ['Safety'],      'fire': ['Safety'],
    'wing': ['Aerodynamics'],    'aero': ['Aerodynamics'],
    'steering': ['Chassis'],     'suspension': ['Chassis'],
  };
  const q = query.toLowerCase();
  const matched = new Set<string>();
  for (const [kw, cats] of Object.entries(keywordsMap)) {
    if (q.includes(kw)) cats.forEach(c => matched.add(c));
  }
  return Array.from(matched);
}

function buildModelMetadata(
  model: ModelRecord,
  cadNodes: CadNodeMatch[],
  includeTags = true,
): Record<string, unknown> {
  return {
    name:        model.name        ?? 'Unknown Model',
    category:    model.category    ?? 'Uncategorized',
    tags: includeTags && Array.isArray(model.model_rule_tags)
      ? model.model_rule_tags.map(t => t.rule_id)
      : [],
    description: model.description ?? null,
    fileSize:    model.file_size_mb ? `${model.file_size_mb} MB` : null,
    cad_nodes:   cadNodes.map(n => ({
      rule_id:         n.rule_id,
      cad_node_name:   n.cad_node_name,
      relevance_score: n.relevance_score ?? null,
    })),
  };
}

async function fetchModelById(
  modelId: number,
  requestId: string,
): Promise<ModelRecord | null> {
  try {
    const { data, error } = await supabase
      .from('fb_models')
      .select('*, model_rule_tags(rule_id)')
      .eq('id', modelId)
      .single();

    if (error || !data) {
      logger.warn(`[CAD] fetchModelById: no model found for id=${modelId}`, { requestId });
      return null;
    }
    return data as ModelRecord;
  } catch (err) {
    logger.error('fetchModelById threw', err, { requestId });
    return null;
  }
}

async function fetchCadNodesForRules(
  ruleIds: string[],
  requestId: string,
): Promise<CadNodeMatch[]> {
  if (ruleIds.length === 0) return [];
  try {
    const { data, error } = await supabase.rpc('match_cad_nodes_by_prefix', { rule_ids: ruleIds });
    if (error) { logger.error('match_cad_nodes_by_prefix RPC error', error, { requestId }); return []; }
    return (data ?? []) as CadNodeMatch[];
  } catch (err) {
    logger.error('fetchCadNodesForRules threw', err, { requestId });
    return [];
  }
}

async function fetchCadByKeyword(
  query: string,
  requestId: string,
): Promise<CadKeywordMatch | null> {
  try {
    const { data, error } = await supabase
      .from('cad_keyword_map')
      .select('model_id, keyword, highlight_meshes, context_meshes');

    if (error || !data) return null;

    const lowerQuery = query.toLowerCase();

    const match = data
      .sort((a, b) => b.keyword.length - a.keyword.length)
      .find(row => lowerQuery.includes(row.keyword.toLowerCase()));

    if (!match) return null;

    return {
      model_id:         match.model_id,
      highlight_meshes: match.highlight_meshes,
      context_meshes:   match.context_meshes,
    };
  } catch (err) {
    logger.error('fetchCadByKeyword threw', err, { requestId });
    return null;
  }
}

// ============================================================================
// ── 14. MIDDLEWARE
// ============================================================================

// ── AUTH BYPASSED FOR DEV — all requests get a dev user ─────────────────────
async function requireAuth(req: Request, _res: Response, next: NextFunction): Promise<void> {
  /*
  // ── ORIGINAL AUTH (commented out for dev) ─────────────────────────────────
  // const authHeader = req.headers['authorization'];
  // const token = authHeader?.startsWith('Bearer ') ? authHeader.slice(7) : null;
  //
  // if (!token) {
  //   res.status(401).json({ error: 'Unauthorized: No token provided.', code: 'NO_TOKEN' });
  //   return;
  // }
  //
  // const validToken = process.env.API_AUTH_TOKEN!;
  // const tokenBuf   = Buffer.from(token);
  // const validBuf   = Buffer.from(validToken);
  // if (tokenBuf.length === validBuf.length && crypto.timingSafeEqual(tokenBuf, validBuf)) {
  //   (req as AuthenticatedRequest).requestId = crypto.randomUUID();
  //   (req as AuthenticatedRequest).user = { role: 'admin' };
  //   return next();
  // }
  //
  // try {
  //   const { data: { user }, error: authError } = await supabase.auth.getUser(token);
  //   if (authError || !user) {
  //     res.status(401).json({ error: 'Unauthorized: Invalid or expired session.', code: 'INVALID_TOKEN' });
  //     return;
  //   }
  //   const { data: teamMember, error: dbError } = await supabase
  //     .from('INDRA_USERS')
  //     .select('is_approved, email')
  //     .eq('id', user.id)
  //     .single();
  //   if (dbError || !teamMember || teamMember.is_approved === false) {
  //     res.status(403).json({ error: 'Your account is pending team lead approval.', code: 'ACCOUNT_PENDING' });
  //     return;
  //   }
  //   (req as AuthenticatedRequest).requestId = crypto.randomUUID();
  //   (req as AuthenticatedRequest).user = { id: user.id, email: teamMember.email };
  //   next();
  // } catch (err) {
  //   logger.error('Authentication error', err);
  //   res.status(500).json({ error: 'Authentication system error.', code: 'AUTH_ERROR' });
  // }
  // ── END ORIGINAL AUTH ─────────────────────────────────────────────────────
  */

  // DEV BYPASS — assign a default dev user, skip all token checks
  (req as AuthenticatedRequest).requestId = crypto.randomUUID();
  (req as AuthenticatedRequest).user = { id: 'dev-user', email: 'dev@local', role: 'admin' };
  logger.warn('[DEV] Auth bypassed — using dev user');
  next();
}

const askLimiter = rateLimit({
  windowMs: CONFIG.RATE_LIMIT_WINDOW_MS,
  max: CONFIG.ASK_RATE_LIMIT_MAX,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many queries. Please wait a moment.', code: 'RATE_LIMITED' },
});

const generalLimiter = rateLimit({
  windowMs: CONFIG.RATE_LIMIT_WINDOW_MS,
  max: CONFIG.RATE_LIMIT_MAX,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests. Please wait a moment.', code: 'RATE_LIMITED' },
});

app.use(helmet());
app.use(express.json({ limit: CONFIG.BODY_SIZE_LIMIT }));
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
}));

// ============================================================================
// ── 15. ROUTES
// ============================================================================

app.get('/health', (_req: Request, res: Response): void => {
  const now = Date.now();

  const geminiRosterSummary = (roster: ModelSlot[]) => ({
    total:   roster.length,
    active:  roster.filter(s => s.coolUntil <= now).length,
    cooling: roster.filter(s => s.coolUntil > now).length,
  });

  const openRouterRosterSummary = {
    total:   openRouterRoster.length,
    active:  openRouterRoster.filter(s => s.coolUntil <= now).length,
    cooling: openRouterRoster.filter(s => s.coolUntil > now).length,
  };

  res.json({
    status: 'ok',
    service: 'INDRA RAG Backend',
    version: '14.0.0-dev',
    auth_mode: 'BYPASSED (dev)',
    cad_layer: 'PARTIALLY DISCONNECTED (dev)',
    uptime_seconds: Math.floor(process.uptime()),
    cache_size: semanticCache.size,
    openrouter_roster: openRouterRosterSummary,
    primary_roster:    geminiRosterSummary(primaryRoster),
    rerank_roster:     geminiRosterSummary(rerankRoster),
  });
});

app.get('/admin/keys', generalLimiter, requireAuth, (_req: Request, res: Response): void => {
  const now = Date.now();

  const geminiRosterStatus = (roster: ModelSlot[]) =>
    roster.map(s => ({
      label:        s.label,
      model:        s.modelName,
      status:       s.coolUntil > now ? 'cooling' : 'active',
      coolUntil:    s.coolUntil > now ? new Date(s.coolUntil).toISOString() : null,
      quotaHits:    s.quotaHits,
      successCount: s.successCount,
    }));

  const openRouterRosterStatus = openRouterRoster.map(s => ({
    label:        s.label,
    model:        s.modelName,
    status:       s.coolUntil > now ? 'cooling' : 'active',
    coolUntil:    s.coolUntil > now ? new Date(s.coolUntil).toISOString() : null,
    quotaHits:    s.quotaHits,
    successCount: s.successCount,
  }));

  res.json({
    generation_chain:  'OpenRouter #1 → OpenRouter #2 → Gemini primary → Gemini rerank key',
    rerank_chain:      'Cohere rerank-v3.5 → Gemini rerank roster → cosine similarity',
    openrouter_roster: openRouterRosterStatus,
    primary_roster:    geminiRosterStatus(primaryRoster),
    rerank_roster:     geminiRosterStatus(rerankRoster),
    note: 'Models are tried in order. A cooled model is skipped until its cooldown expires.',
  });
});

app.get('/admin/cache', generalLimiter, requireAuth, (_req: Request, res: Response): void => {
  const now = Date.now();
  let active = 0, expired = 0;
  for (const entry of semanticCache.values()) {
    entry.expiresAt > now ? active++ : expired++;
  }
  res.json({ total_entries: semanticCache.size, active, expired, max_entries: CONFIG.CACHE_MAX_ENTRIES });
});

app.post('/admin/cache/invalidate', generalLimiter, requireAuth, (req: Request, res: Response): void => {
  const domain = typeof req.body?.domain === 'string' ? req.body.domain : undefined;
  const removed = invalidateCache(domain);
  logger.info('Cache invalidated by admin', { domain: domain ?? 'ALL', removed });
  res.json({ message: `Invalidated ${removed} cache entries${domain ? ` for domain "${domain}"` : ' (all domains)'}.`, removed });
});

app.post('/admin/cache/clear', generalLimiter, requireAuth, (_req: Request, res: Response): void => {
  const cleared = semanticCache.size;
  semanticCache.clear();
  logger.info('Cache cleared by admin', { cleared });
  res.json({ message: `Cleared ${cleared} cache entries.` });
});

// ── Main RAG endpoint ────────────────────────────────────────────────────────
app.post('/ask_indra', askLimiter, requireAuth, async (req: Request, res: Response): Promise<void> => {
  const authReq   = req as AuthenticatedRequest;
  const requestId = authReq.requestId;
  const { message, domain } = req.body as { message: unknown; domain: unknown };

  if (!message || typeof message !== 'string') {
    res.status(400).json({ error: 'Invalid query: message must be a non-empty string.', code: 'INVALID_INPUT' });
    return;
  }
  const trimmed = message.trim();
  if (trimmed.length < CONFIG.MIN_MESSAGE_LENGTH) {
    res.status(400).json({ error: `Query too short. Minimum ${CONFIG.MIN_MESSAGE_LENGTH} characters.`, code: 'QUERY_TOO_SHORT' });
    return;
  }
  if (trimmed.length > CONFIG.MAX_MESSAGE_LENGTH) {
    res.status(400).json({ error: `Query too long. Maximum ${CONFIG.MAX_MESSAGE_LENGTH} characters.`, code: 'QUERY_TOO_LONG' });
    return;
  }
  if (detectInjection(trimmed)) {
    logger.warn('[SECURITY] Prompt injection attempt detected', { requestId, query: trimmed.slice(0, 100) });
    res.status(400).json({ error: 'Query contains disallowed patterns.', code: 'INJECTION_DETECTED' });
    return;
  }

  const sanitizedDomain: string = (
    typeof domain === 'string' &&
    VALID_DOMAINS.includes(domain.trim() as ValidDomain)
  ) ? domain.trim() : 'General';

  if (typeof domain !== 'string' || !VALID_DOMAINS.includes(domain.trim() as ValidDomain)) {
    logger.warn('[DOMAIN] Request used missing or unknown domain; fell back to "General"', {
      requestId, received: domain ?? null,
    });
  }

  try {
    const intent            = classifyIntent(trimmed);
    const expandedEmbedding = await expandAndAverageEmbedding(trimmed);
    logger.info('Query processed', { requestId, intent, domain: sanitizedDomain });

    const cacheHit = findCacheHit(expandedEmbedding, sanitizedDomain);
    if (cacheHit) {
      logger.info('Cache hit', { requestId });
      res.json({ ...cacheHit, _cache: 'hit' });
      return;
    }

    const { data: matchedRules, error: rpcError } = await supabase.rpc('match_rulebook_chunks', {
      query_embedding: expandedEmbedding,
      match_threshold: CONFIG.MATCH_THRESHOLD,
      match_count:     CONFIG.MATCH_COUNT,
      filter_domain:   sanitizedDomain,
    });
    if (rpcError) logger.error('match_rulebook_chunks RPC error', rpcError, { requestId });

    const [learnedMatches] = await Promise.all([
      (async (): Promise<LearnedChunk[]> => {
        try {
          const { data } = await supabase.rpc('match_learned_chunks', {
            query_embedding: expandedEmbedding,
            match_threshold: CONFIG.LEARNED_MATCH_THRESHOLD,
            match_count:     CONFIG.LEARNED_MATCH_COUNT,
          });
          return (data ?? []) as LearnedChunk[];
        } catch (err) {
          logger.error('Learned chunks fetch error', err, { requestId });
          return [];
        }
      })(),
      // ── 3D CAD keyword fetch — PARTIALLY DISCONNECTED (dev) ──────────────
      // fetchCadByKeyword(trimmed, requestId),   // <-- disabled
      Promise.resolve(null),
    ]);

    const rerankedRules: RuleChunk[] = matchedRules && (matchedRules as RuleChunk[]).length > 1
      ? await rerankChunks(trimmed, matchedRules as RuleChunk[])
      : (matchedRules ?? []) as RuleChunk[];

    const hasRuleMatches    = rerankedRules.length > 0;
    const hasLearnedMatches = learnedMatches.length > 0;
    const ruleIds: string[] = rerankedRules
      .map(r => r.rule_id)
      .filter((id): id is string => Boolean(id));

    // ── 3D Model resolution — PARTIALLY DISCONNECTED (dev) ──────────────────
    const topModel: ModelRecord | null     = null;
    const highlightMeshes: string[]        = [];
    const contextMeshes: string[]          = [];
    const cadNodes: CadNodeMatch[]         = [];

    void topModel;
    void ruleIds;
    void extractKeywordsFromQuery;
    void buildModelMetadata;
    void fetchModelById;
    void fetchCadNodesForRules;
    void fetchCadByKeyword;

    logger.info('[CAD] 3D layer partially disconnected — meshes suppressed (dev mode)', { requestId });

    if (!hasRuleMatches && !hasLearnedMatches) {
      logger.info('No rulebook match found', { requestId, domain: sanitizedDomain });

      const noMatchPayload: Record<string, unknown> = {
        answer: `**No rulebook match found in the ${sanitizedDomain} domain.**\n\n⚡ INDRA NOTE: Try rephrasing with specific component names or rule keywords. Switch domain if this is a cross-domain query.`,
        citations: [],
        intent: 'no_match',
        code:   'NO_MATCH',
      };

      void saveLog({ request_id: requestId, query: trimmed, result: 'no_match', domain: sanitizedDomain, intent, created_at: new Date().toISOString() });
      res.json(noMatchPayload);
      return;
    }

    // ── Generate answer ───────────────────────────────────────────────────────
    const ruleContext = [
      ...rerankedRules.map(r => `[Rule ${r.rule_id}]\n${r.content}`),
      ...(hasLearnedMatches
        ? ['\n--- PREVIOUSLY VERIFIED ANSWERS ---',
           ...learnedMatches.map(l => `Q: ${l.question}\nA: ${l.answer}`)]
        : []),
    ].join('\n\n---\n\n');

    const systemPrompt = buildSystemPrompt(intent, ruleContext, trimmed, sanitizedDomain);
    const answer       = await generate(systemPrompt, false, 0.35);

    // ── Build response ────────────────────────────────────────────────────────
    const responsePayload: Record<string, unknown> = {
      answer,
      intent,
      citations: rerankedRules.map(r => ({
        rule_id:    r.rule_id,
        content:    r.content,
        similarity: r.similarity,
      })),
    };

    if (hasLearnedMatches) {
      responsePayload.learned_citations = learnedMatches.map(l => ({
        question: l.question,
        answer:   l.answer,
        source:   l.source,
      }));
    }

    responsePayload.model_url        = null;
    responsePayload.model_metadata   = null;
    responsePayload.highlight_meshes = [];
    responsePayload.context_meshes   = [];
    responsePayload.cad_nodes        = [];

    writeCache(expandedEmbedding, sanitizedDomain, responsePayload);
    res.json(responsePayload);

    void saveLog({
      request_id:              requestId,
      query:                   trimmed,
      result:                  'success',
      domain:                  sanitizedDomain,
      intent,
      model_found:             false,
      model_name:              null,
      citations_count:         rerankedRules.length,
      learned_citations_count: learnedMatches.length,
      cad_nodes_count:         cadNodes.length,
      highlight_meshes_count:  highlightMeshes.length,
      context_meshes_count:    contextMeshes.length,
      keyword_cad_hit:         false,
      cache_written:           true,
      created_at:              new Date().toISOString(),
    });

  } catch (error: unknown) {
    const msg              = error instanceof Error ? error.message : String(error);
    const isQuotaExhausted = msg.includes('QUOTA_EXHAUSTED') || msg.includes('OPENROUTER_EXHAUSTED');
    const isEmbedExhausted = msg.includes('EMBED_EXHAUSTED');
    const isTimeout        = msg.startsWith('TIMEOUT:') || msg.includes('EMBED_TIMEOUT');

    logger.error('Error in /ask_indra', error, { requestId });

    if (isEmbedExhausted) {
      res.status(503).json({ error: 'Service temporarily unavailable — please try again in a moment.', code: 'EMBED_EXHAUSTED' });
    } else if (isQuotaExhausted) {
      res.status(503).json({ error: 'All AI capacity is temporarily at quota limit. Please try again in a minute.', code: 'QUOTA_EXHAUSTED' });
    } else if (isTimeout) {
      res.status(504).json({ error: 'The AI took too long to respond. Please try again.', code: 'TIMEOUT' });
    } else {
      res.status(500).json({ error: 'The server encountered an error. Please try again.', code: 'INTERNAL_ERROR' });
    }
  }
});

// ── Feedback & learning ──────────────────────────────────────────────────────
const RULE_ID_RE = /\b([A-Z]{1,3}\d+(?:\.\d+){1,3})\b/;
const URL_RE     = /\bhttps?:\/\/\S+/i;
const CODE_RE    = /```|<code>|function\s*\(|class\s+\w+\s*[{:]/i;
const ESC_RE     = /\\u[0-9a-fA-F]{4}|\\x[0-9a-fA-F]{2}/;

function isValidLearnedPair(question: string, answer: string): { ok: boolean; reason?: string } {
  if (!RULE_ID_RE.test(answer)) {
    return { ok: false, reason: 'answer must reference a rule ID (e.g. [T3.1.2])' };
  }
  if (/^(i (can'?t|cannot|am unable)|as an? (ai|language model)|i don'?t have)/i.test(answer.trim())) {
    return { ok: false, reason: 'answer appears to be a refusal, not a rule citation' };
  }
  if (URL_RE.test(question) || URL_RE.test(answer)) {
    return { ok: false, reason: 'URLs are not allowed in learned pairs' };
  }
  if (CODE_RE.test(answer)) {
    return { ok: false, reason: 'code blocks are not allowed in learned pairs' };
  }
  if (ESC_RE.test(answer)) {
    return { ok: false, reason: 'escape sequences are not allowed in learned pairs' };
  }
  if (detectInjection(question) || detectInjection(answer)) {
    return { ok: false, reason: 'injection patterns detected' };
  }
  if (answer.trim().split(/\s+/).length < 8) {
    return { ok: false, reason: 'answer is too short to be a valid learned pair' };
  }
  return { ok: true };
}

app.post('/feedback', generalLimiter, requireAuth, async (req: Request, res: Response): Promise<void> => {
  const authReq   = req as AuthenticatedRequest;
  const requestId = authReq.requestId;
  const userId    = authReq.user?.id;

  const { question, answer, domain, rating } = req.body as {
    question: unknown; answer: unknown; domain: unknown; rating: unknown;
  };

  if (
    typeof question !== 'string' || question.trim().length < 2 || question.length > 1000 ||
    typeof answer   !== 'string' || answer.trim().length < 2   || answer.length > 5000   ||
    typeof domain   !== 'string' ||
    !['good', 'bad'].includes(rating as string)
  ) {
    res.status(400).json({ error: 'Invalid payload. Check field types and ensure lengths are within limits.', code: 'INVALID_INPUT' });
    return;
  }

  const sanitizedDomain = domain.trim();
  if (!VALID_DOMAINS.includes(sanitizedDomain as ValidDomain)) {
    res.status(400).json({ error: 'Invalid domain.', code: 'INVALID_DOMAIN' });
    return;
  }

  try {
    const trimmedQuestion = question.trim();
    const trimmedAnswer   = answer.trim();
    const dbTasks: Promise<void>[] = [];

    if (rating === 'good') {
      const check = isValidLearnedPair(trimmedQuestion, trimmedAnswer);
      if (!check.ok) {
        logger.warn('[FEEDBACK] Rejected learned pair', { requestId, userId, reason: check.reason });
        res.status(400).json({ error: `Cannot save learned pair: ${check.reason}.`, code: 'INVALID_LEARNED_PAIR' });
        return;
      }
      dbTasks.push(saveLearnedPair(trimmedQuestion, trimmedAnswer, sanitizedDomain, 'user_feedback'));
      logger.info('Learned pair saved from feedback', { requestId, userId, question: trimmedQuestion.slice(0, 60) });
    }

    dbTasks.push(saveLog({
      request_id: requestId,
      user_id:    userId,
      type:       'feedback',
      question:   trimmedQuestion,
      domain:     sanitizedDomain,
      rating,
      created_at: new Date().toISOString(),
    }));

    await Promise.all(dbTasks);
    res.json({ message: rating === 'good' ? 'Answer learned — thanks for the signal!' : 'Feedback noted. We will work to improve.' });
  } catch (err) {
    logger.error('Feedback save error', err, { requestId });
    res.status(500).json({ error: 'Failed to save feedback.', code: 'INTERNAL_ERROR' });
  }
});

// ── 3D Model gallery ─────────────────────────────────────────────────────────
app.get('/models', generalLimiter, requireAuth, async (req: Request, res: Response): Promise<void> => {
  const { category, limit = '20', offset = '0', search } = req.query;
  const parsedLimit  = Math.min(Math.max(Number(limit)  || 20, 1), 100);
  const parsedOffset = Math.max(Number(offset) || 0, 0);

  try {
    let query = supabase
      .from('fb_models')
      .select('id, name, category, thumbnail_url, description, file_size_mb', { count: 'exact' });

    if (category && category !== 'All') query = query.eq('category', category as string);
    if (search && typeof search === 'string' && search.trim().length > 0) {
      query = query.ilike('name', `%${search.trim()}%`);
    }

    const { data, count, error } = await query
      .order('name')
      .range(parsedOffset, parsedOffset + parsedLimit - 1);

    if (error) throw error;
    res.json({ models: data, total: count, has_more: count ? count > parsedOffset + parsedLimit : false });
  } catch (err) {
    logger.error('Error fetching models', err);
    res.status(500).json({ error: 'Failed to fetch model library.', code: 'INTERNAL_ERROR' });
  }
});

app.get('/models/:id', generalLimiter, requireAuth, async (req: Request, res: Response): Promise<void> => {
  const id: string = String(req.params['id'] ?? '');
  if (!/^[\w-]{1,64}$/.test(id)) {
    res.status(400).json({ error: 'Invalid model ID.', code: 'INVALID_INPUT' });
    return;
  }
  try {
    const { data, error } = await supabase
      .from('fb_models')
      .select('*, model_rule_tags(rule_id, relevance_score)')
      .eq('id', id)
      .single();

    if (error || !data) {
      res.status(404).json({ error: 'Model not found.', code: 'NOT_FOUND' });
      return;
    }
    res.json(data);
  } catch (err) {
    logger.error('Error fetching model by ID', err, { id });
    res.status(500).json({ error: 'Failed to fetch model details.', code: 'INTERNAL_ERROR' });
  }
});

// ── Quiz generation ──────────────────────────────────────────────────────────
app.get('/quiz', generalLimiter, requireAuth, async (req: Request, res: Response): Promise<void> => {
  const { domain = 'General', count = '3' } = req.query;
  const questionCount = Math.min(Math.max(Number(count) || 3, 1), 10);

  if (!VALID_DOMAINS.includes((domain as string).trim() as ValidDomain)) {
    res.status(400).json({ error: 'Invalid domain.', code: 'INVALID_DOMAIN' });
    return;
  }

  try {
    const { data: chunks } = await supabase
      .from('rulebook_chunks')
      .select('rule_id, content')
      .eq('domain', domain as string)
      .limit(questionCount * 3);

    if (!chunks || chunks.length === 0) {
      res.status(404).json({ error: 'No rulebook content found for this domain.', code: 'NOT_FOUND' });
      return;
    }

    const context = chunks.map(c => `[${c.rule_id}] ${c.content}`).join('\n\n');
    const prompt  = `You are a technical regulations examiner. Based on the regulation excerpts below, generate exactly ${questionCount} multiple-choice quiz questions. Each question must test specific technical knowledge from the rules.

Return ONLY a JSON array with this exact structure (no markdown, no explanation):

[
  {
    "question": "...",
    "options": ["A", "B", "C", "D"],
    "correctAnswer": 0,
    "explanation": "Rule X.Y.Z states: ...",
    "rule_id": "X.Y.Z"
  }
]

correctAnswer is the 0-based index of the correct option.

REGULATION EXCERPTS:
${context}`;

    const raw       = await generate(prompt, true, 0.4);
    const cleaned   = raw.replace(/```json|```/g, '').trim();
    let questions: unknown;
    try {
      questions = JSON.parse(cleaned);
    } catch (e) {
      logger.error('Quiz JSON parse failed', e, { preview: cleaned.slice(0, 200) });
      res.status(502).json({ error: 'Quiz generation returned invalid JSON.', code: 'INVALID_LLM_JSON' });
      return;
    }

    if (!Array.isArray(questions)) {
      res.status(502).json({ error: 'Quiz generation did not return an array.', code: 'INVALID_LLM_STRUCTURE' });
      return;
    }

    interface RawQuestion { question?: unknown; options?: unknown; correctAnswer?: unknown; explanation?: unknown; rule_id?: unknown; }
    const validQuestions: { question: string; options: string[]; explanation: string; rule_id: string }[] = [];
    for (const q of questions as RawQuestion[]) {
      if (
        typeof q.question !== 'string' || q.question.trim().length < 5 ||
        !Array.isArray(q.options) || q.options.length < 2 ||
        !q.options.every(o => typeof o === 'string' && o.trim().length > 0) ||
        typeof q.correctAnswer !== 'number' || q.correctAnswer < 0 || q.correctAnswer >= q.options.length
      ) {
        logger.warn('Quiz: dropping malformed question', { q });
        continue;
      }
      validQuestions.push({
        question:    q.question
