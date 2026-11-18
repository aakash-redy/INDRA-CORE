import express from 'express';
import type { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { createClient } from '@supabase/supabase-js';
import rateLimit from 'express-rate-limit';
import dotenv from 'dotenv';
import helmet from 'helmet';
import Groq from 'groq-sdk';
import crypto from 'crypto';

dotenv.config();

// ============================================================================
// ── 1. CONFIGURATION & STARTUP CHECKS
// ============================================================================

const requiredEnvVars = [
  'GEMINI_API_KEY',
  'NOMIC_API_KEY',
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

const CONFIG = {
  MATCH_THRESHOLD: 0.5,
  LEARNED_MATCH_THRESHOLD: 0.7,
  MATCH_COUNT: 7,
  GRAPH_RELATED_LIMIT: 6,
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

interface RuleRelation {
  target_rule_id: string;
  relation_type: string;
  context: string;
}

interface RuleChunk {
  rule_id: string;
  section?: string;
  content: string;
  relations_metadata?: RuleRelation[];
  match_role?: 'explicit_primary' | 'vector_primary' | 'graph_related';
  similarity: number;
  rerank_score?: number;
}

interface RawGraphRuleRow {
  rule_id: string;
  section: string;
  rule_text: string;
  relations_metadata: RuleRelation[] | null;
  match_role: 'explicit_primary' | 'vector_primary' | 'graph_related';
  similarity: number;
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

interface GroqSlot {
  label: string;
  client: Groq;
  coolUntil: number;
  quotaHits: number;
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

function findCacheHit(embedding: number[], domain: string, explicitRuleIds: string[]): Record<string, unknown> | null {
  const now = Date.now();
  const rulePrefix = explicitRuleIds.length > 0 ? explicitRuleIds.slice().sort().join(',') : 'none';
  const expectedPrefix = `${domain}:${rulePrefix}:`;

  for (const [key, entry] of semanticCache.entries()) {
    if (entry.expiresAt < now) { semanticCache.delete(key); continue; }
    if (!key.startsWith(expectedPrefix)) continue;
    if (cosineSimilarity(embedding, entry.embedding) >= CONFIG.CACHE_SIMILARITY_THRESHOLD) {
      return entry.response;
    }
  }
  return null;
}

function writeCache(
  embedding: number[],
  domain: string,
  explicitRuleIds: string[],
  response: Record<string, unknown>,
): void {
  if (semanticCache.size >= CONFIG.CACHE_MAX_ENTRIES) {
    const oldestKey = semanticCache.keys().next().value;
    if (oldestKey) semanticCache.delete(oldestKey);
    logger.info(`Cache eviction: max entries (${CONFIG.CACHE_MAX_ENTRIES}) reached.`);
  }
  const rulePrefix = explicitRuleIds.length > 0 ? explicitRuleIds.slice().sort().join(',') : 'none';
  const hash = crypto.createHash('sha256').update(JSON.stringify(embedding)).digest('hex').slice(0, 32);
  const key = `${domain}:${rulePrefix}:${hash}`;
  semanticCache.set(key, { embedding, response, expiresAt: Date.now() + CONFIG.CACHE_TTL_MS });
}

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

async function saveLog(entry: Record<string, unknown>) {
  try {
    const { error } = await supabase.from('query_logs').insert(entry);
    if (error) {
      logger.error('saveLog DB error', {
        message: error.message,
        details: error.details,
        code: error.code,
      });
    }
  } catch (err) {
    logger.error('saveLog DB exception', {
      message: err instanceof Error ? err.message : JSON.stringify(err),
    });
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
// ── 7. GEMINI GENERATION ENGINE (Primary Key + Backup Key)
// ============================================================================

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY || '';
const OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions';

// Primary synthesis chain (falls back server-side if any model spikes or 503s)
const OPENROUTER_SYNTHESIS_MODELS = [
  'anthropic/claude-haiku-4.5',
  'google/gemini-2.5-flash',
  'openai/gpt-4o-mini',
];

// Ultra-fast compression model
const OPENROUTER_FAST_MODELS = [
  'google/gemini-2.5-flash-lite',
  'openai/gpt-4o-mini',
];

async function callOpenRouter(
  prompt: string,
  models: string[],
  temperature: number,
  requireJson = false,
): Promise<string> {
  if (!OPENROUTER_API_KEY) {
    throw new Error('MISSING_KEY: OPENROUTER_API_KEY is not set in environment.');
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), CONFIG.GEMINI_TIMEOUT_MS);

  try {
    const res = await fetch(OPENROUTER_URL, {
      method: 'POST',
      signal: controller.signal,
      headers: {
        'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://indra-fb.vercel.app',
        'X-OpenRouter-Title': 'INDRA Formula Bharat',
      },
      body: JSON.stringify({
        model: models[0],
        models, // OpenRouter handles fallback automatically in one hop
        temperature,
        provider: { sort: 'latency' }, // Picks the fastest host
        ...(requireJson ? { response_format: { type: 'json_object' } } : {}),
        messages: [{ role: 'user', content: prompt }],
      }),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(`OpenRouter ${res.status}: ${data?.error?.message || JSON.stringify(data)}`);
    }

    const answer = data.choices?.[0]?.message?.content ?? '';
    logger.info(`[OPENROUTER] ${data.model || models[0]} via ${data.provider || 'auto'} ✓`, {
      costUsd: data.usage?.cost,
      tokens: data.usage?.total_tokens,
    });

    return answer;
  } finally {
    clearTimeout(timer);
  }
}

async function generate(
  prompt: string,
  requireJson = false,
  temperature = 0.6,
): Promise<string> {
  return callOpenRouter(prompt, OPENROUTER_SYNTHESIS_MODELS, temperature, requireJson);
}

async function generateFast(prompt: string, temperature = 0.1): Promise<string> {
  return callOpenRouter(prompt, OPENROUTER_FAST_MODELS, temperature, false);
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

    const data = await resJson(response);

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
    const raw     = await generate(prompt, true, 0.1);
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

async function resJson(response: globalThis.Response): Promise<{
  results: { index: number; relevance_score: number }[];
}> {
  return await response.json() as {
    results: { index: number; relevance_score: number }[];
  };
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
// ── 13. INTENT CLASSIFICATION, MULTI-RULE EXTRACTION & PROMPT BUILDING
// ============================================================================

function extractMentionedRuleIds(userPrompt: string): string[] {
  const regex = /\b(A|T|EV|CV|IN|S|D)\s*\.?\s*(\d+(?:\.\d+)*)\b/gi;
  const matches = [...userPrompt.matchAll(regex)];
  return [...new Set(matches.map(m => `${m[1].toUpperCase()}.${m[2]}`))];
}

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
  const intentGuidance: Record<QueryIntent, string> = {
    dimension:
      'Lead with the exact numeric limits and tolerances requested, then explain the mechanical or ergonomic reasoning behind them.',
    compliance:
      'Give a clear compliance verdict upfront, then walk through the exact engineering criteria and edge cases that decide pass/fail at tech inspection.',
    definition:
      'Explain what the component or boundary actually is in real racecar terms first, clearly separating it from easily confused rulebook terms.',
    procedure:
      'Walk through the build, mounting, or inspection steps in logical engineering order, highlighting critical scrutineering checks.',
    general:
      'Answer the exact question asked like a senior Formula Student design lead—explain the engineering logic clearly and only bring in connected rules if they directly interact with the question.',
  };

  return `You are INDRA, a senior Formula Student scrutineering and design engineer helping a team interpret the Formula Bharat rulebook.

HOW TO REASON AND WRITE:
1. **Answer ONLY what the user asked:** Do NOT summarize every rule in the context. Filter out any rule or graph neighbor that belongs to an unrelated sub-system (for example, ignore composite equivalency or IC intake rules if the user asked about a standard steel tubular Side Impact Structure).
2. **Synthesize with Engineering Intuition:** Combine exact rulebook constraints with your own mechanical/electrical engineering knowledge to explain *why* the rule exists, how load paths or failure modes behave, and what matters in CAD and tech inspection. Never just stitch rule sentences together with filler grammar.
3. **Keep Distinct Terms Separate:** Never blur separate rulebook definitions together (e.g., physical *Side Impact Structure* [T.3.14] vs. the geometric *Side Impact Structure Envelope* zone [T.1.1.17]).
4. **Natural, Unscripted Flow:** Vary your structure organically (clean prose, mixed paragraphs + bullets, or steps) based on what fits the question best.

STYLE EXAMPLE (Follow this natural engineering tone, NOT rigid rulebook paraphrasing):
- Raw Rule Input: [Rule T.3.8.1 | Section: Main Hoop] The main hoop must be constructed of a single piece of uncut, continuous, closed section steel tubing.
- Target Response Style: Your main hoop has to be bent from one continuous length of closed-section steel tubing—you cannot weld two pieces together or splice in a section anywhere along the hoop [T.3.8.1]. Keeping it uncut avoids introducing heat-affected weld zones in the primary rollover load path, which scrutineers will immediately fail during chassis inspection.

STRICT CITATION GUARDRAIL:
- Cite rule IDs inline in square brackets right after the specific requirement they support (e.g., [T.3.14.1]).
- ONLY cite rule IDs that explicitly appear as [Rule X.Y.Z ...] headers in the REGULATION CONTEXT below. Never invent or guess rule numbers.

QUERY FOCUS (${domain}):
${intentGuidance[intent]}

REGULATION CONTEXT:
${ruleContext}

TEAM QUESTION:
${query}`;
}
// ============================================================================
// ── 14. MIDDLEWARE
// ============================================================================

async function requireAuth(req: Request, _res: Response, next: NextFunction): Promise<void> {
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

// ============================================================================
// ── 15. ROUTES
// ============================================================================

app.get('/health', (_req: Request, res: Response): void => {
  res.json({
    status: 'ok',
    service: 'INDRA GraphRAG Backend',
    version: '15.1.0-graph-dev',
    auth_mode: 'BYPASSED (dev)',
    cad_layer: 'PARTIALLY DISCONNECTED (dev)',
    uptime_seconds: Math.floor(process.uptime()),
    cache_size: semanticCache.size,
    active_models: OPENROUTER_SYNTHESIS_MODELS,
    openrouter_configured: Boolean(OPENROUTER_API_KEY),
  });
});

app.get('/admin/cache', generalLimiter, requireAuth, (_req: Request, res: Response): void => {
  const now = Date.now();
  let active = 0, expired = 0;
  for (const entry of semanticCache.values()) {
    entry.expiresAt > now ? active++ : expired++;
  }
  res.json({
    total_entries: semanticCache.size,
    active,
    expired,
    max_entries: CONFIG.CACHE_MAX_ENTRIES,
  });
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

async function compressRelevantEvidence(
  query: string,
  rules: RuleChunk[],
): Promise<string> {
  if (rules.length === 0) return '';

  const rawDump = rules
    .map(
      r =>
        `[Rule ${r.rule_id}${r.section ? ` | Section: ${r.section}` : ''} | Role: ${r.match_role ?? 'primary'}]\n${r.content}`
    )
    .join('\n\n');

  const filterPrompt = `You are a strict technical relevance filter for Formula Bharat regulations.
Given a TEAM QUESTION and RETRIEVED RULES, extract ONLY the rules and exact facts that directly answer or materially constrain the question asked.

STRICT FILTERING RULES:
- Discard any rule belonging to an unrelated sub-system (e.g., drop composite monocoque or IC air intake rules if the user asked about a standard steel tubular Side Impact Structure).
- Keep distinct rulebook concepts clearly separated (e.g., physical Side Impact Structure rules vs. geometric Side Impact Structure Envelope definitions).
- Preserve exact numbers, dimensions, materials, and [Rule X.Y.Z] IDs verbatim.
- Output ONLY the filtered rules and facts—no conversational filler.

TEAM QUESTION: "${query}"

RETRIEVED RULES:
${rawDump}`;

  try {
    const compressed = await generateFast(filterPrompt, 0.1);
    return compressed.trim() || rawDump;
  } catch (err) {
    logger.warn('[COMPRESS] Evidence filter failed — falling back to raw context', {
      error: err instanceof Error ? err.message : String(err),
    });
    return rawDump;
  }
}

// ── Main GraphRAG endpoint ───────────────────────────────────────────────────
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
    const explicitRuleIds   = extractMentionedRuleIds(trimmed);
    const expandedEmbedding = await expandAndAverageEmbedding(trimmed);
    logger.info('Query processed', { requestId, intent, domain: sanitizedDomain, explicitRuleIds });

    const cacheHit = findCacheHit(expandedEmbedding, sanitizedDomain, explicitRuleIds);
    if (cacheHit) {
      logger.info('Cache hit', { requestId });
      res.json({ ...cacheHit, _cache: 'hit' });
      return;
    }

    // 1. Multi-rule hybrid vector + graph RPC
    const { data: rawGraphRules, error: rpcError } = await supabase.rpc('match_multi_rules_with_graph', {
      query_embedding:     expandedEmbedding,
      explicit_rule_ids:   explicitRuleIds,
      primary_match_count: CONFIG.MATCH_COUNT,
      related_match_limit: CONFIG.GRAPH_RELATED_LIMIT,
    });
    if (rpcError) logger.error('match_multi_rules_with_graph RPC error', rpcError, { requestId });

    const mappedRules: RuleChunk[] = ((rawGraphRules ?? []) as RawGraphRuleRow[]).map(r => ({
      rule_id:            r.rule_id,
      section:            r.section,
      content:            r.rule_text,
      relations_metadata: r.relations_metadata ?? [],
      match_role:         r.match_role,
      similarity:         r.similarity,
    }));

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
      Promise.resolve(null),
    ]);

    // 2. Keep explicitly mentioned rules locked at top; rerank semantic + graph-linked rules
    const explicitLocked     = mappedRules.filter(r => r.match_role === 'explicit_primary');
    const candidatesToRerank = mappedRules.filter(r => r.match_role !== 'explicit_primary');

    const rerankedCandidates: RuleChunk[] = candidatesToRerank.length > 1
      ? await rerankChunks(trimmed, candidatesToRerank)
      : candidatesToRerank;

    const rerankedRules: RuleChunk[] = [...explicitLocked, ...rerankedCandidates];

    const hasRuleMatches    = rerankedRules.length > 0;
    const hasLearnedMatches = learnedMatches.length > 0;
    const highlightMeshes: string[] = [];
    const contextMeshes: string[]   = [];
    const cadNodes: CadNodeMatch[]  = [];

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

    // 3. Build enriched pre-prompt context with graph relationship explanations
    const filteredRulesText =
      rerankedRules.length > 4
        ? await compressRelevantEvidence(trimmed, rerankedRules)
        : rerankedRules
            .map(
              r =>
                `[Rule ${r.rule_id}${r.section ? ` | Section: ${r.section}` : ''}]\n${r.content}`
            )
            .join('\n\n');

    const ruleContext = [
      filteredRulesText,
      ...(hasLearnedMatches
        ? [
            '\n=== VERIFIED PAST ANSWERS ===',
            ...learnedMatches.map(l => `Q: ${l.question}\nA: ${l.answer}`),
          ]
        : []),
    ].join('\n\n');

    const systemPrompt = buildSystemPrompt(intent, ruleContext, trimmed, sanitizedDomain);
    const answer       = await generate(systemPrompt, false, 0.6);
    const responsePayload: Record<string, unknown> = {
      answer,
      intent,
      citations: rerankedRules.map(r => ({
        rule_id:    r.rule_id,
        section:    r.section ?? null,
        match_role: r.match_role ?? 'vector_primary',
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

    writeCache(expandedEmbedding, sanitizedDomain, explicitRuleIds, responsePayload);
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
    const isQuotaExhausted = msg.includes('QUOTA_EXHAUSTED');
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
const RULE_ID_RE = /\b([A-Z]{1,3}\.?\d+(?:\.\d+){1,3})\b/;
const URL_RE     = /\bhttps?:\/\/\S+/i;
const CODE_RE    = /```|<code>|function\s*\(|class\s+\w+\s*[{:]/i;
const ESC_RE     = /\\u[0-9a-fA-F]{4}|\\x[0-9a-fA-F]{2}/;

function isValidLearnedPair(question: string, answer: string): { ok: boolean; reason?: string } {
  if (!RULE_ID_RE.test(answer)) {
    return { ok: false, reason: 'answer must reference a rule ID (e.g. [T.3.1.2])' };
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

// ── Quiz generation (pointed to formula_bharat_rules) ────────────────────────
app.get('/quiz', generalLimiter, requireAuth, async (req: Request, res: Response): Promise<void> => {
  const { domain = 'General', count = '3' } = req.query;
  const questionCount = Math.min(Math.max(Number(count) || 3, 1), 10);

  if (!VALID_DOMAINS.includes((domain as string).trim() as ValidDomain)) {
    res.status(400).json({ error: 'Invalid domain.', code: 'INVALID_DOMAIN' });
    return;
  }

  try {
    const { data: rawChunks } = await supabase
      .from('formula_bharat_rules')
      .select('rule_id, rule_text')
      .limit(questionCount * 3);

    const chunks = (rawChunks ?? []).map(c => ({
      rule_id: c.rule_id,
      content: c.rule_text,
    }));

    if (chunks.length === 0) {
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
        question:    q.question,
        options:     q.options as string[],
        explanation: typeof q.explanation === 'string' ? q.explanation : '',
        rule_id:     typeof q.rule_id === 'string' ? q.rule_id : '',
      });
    }

    if (validQuestions.length === 0) {
      res.status(502).json({ error: 'Quiz generation produced no valid questions.', code: 'NO_VALID_QUESTIONS' });
      return;
    }

    res.json({
      questions: validQuestions,
      domain,
      generated: true,
      _meta: { generated_count: validQuestions.length, requested_count: questionCount },
    });
  } catch (err) {
    logger.error('Quiz generation error', err);
    res.status(500).json({ error: 'Failed to generate quiz questions.', code: 'INTERNAL_ERROR' });
  }
});

// ============================================================================
// ── 16. ERROR HANDLING & STARTUP
// ============================================================================

app.use((_req: Request, res: Response): void => {
  res.status(404).json({ error: 'Route not found.', code: 'NOT_FOUND' });
});

// eslint-disable-next-line @typescript-eslint/no-unused-vars
app.use((error: Error, _req: Request, res: Response, _next: NextFunction): void => {
  logger.error('Unhandled error', error);
  res.status(500).json({ error: 'Something went wrong. Please try again.', code: 'INTERNAL_ERROR' });
});

app.listen(PORT, () => {
  console.log(`\n🚀  INDRA GraphRAG Backend v15.1.0-graph-dev`);
  console.log(`🌐  http://localhost:${PORT}`);
  console.log(`🌍  Environment  : ${IS_PROD ? 'PRODUCTION' : 'development'}`);
  console.log(`⚠️   Auth         : BYPASSED — all requests accepted as dev@local (admin)`);
  console.log(`⚠️   3D Layer     : PARTIALLY DISCONNECTED — mesh data suppressed in responses`);
  console.log(`🛡️   Security     : Helmet + Rate Limiting + Injection Guard (still active)`);
  console.log(
  `🔑  OpenRouter   : ${OPENROUTER_API_KEY ? 'configured' : 'MISSING'} (${OPENROUTER_SYNTHESIS_MODELS.join(' → ')})`
);
  console.log(`🔀  Rerank chain : Cohere rerank-v3.5 → Gemini fallback → cosine similarity`);
  console.log(`🧠  Embed model  : nomic-embed-text-v1.5 (Nomic Atlas 768d)`);
  console.log(`🕸️   Graph DB     : formula_bharat_rules (match_multi_rules_with_graph)`);
  console.log(`⚡  Cache        : Semantic in-memory | TTL: ${CONFIG.CACHE_TTL_MS / 60000}min | Max: ${CONFIG.CACHE_MAX_ENTRIES} entries\n`);
});
