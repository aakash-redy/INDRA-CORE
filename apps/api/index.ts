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
  'SUPABASE_URL',
  'SUPABASE_ANON_KEY',
  'GEMINI_API_KEY',
  'GEMINI_RERANK_API_KEY',
  'API_AUTH_TOKEN',
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
] as const;
type ValidDomain = typeof VALID_DOMAINS[number];

const CONFIG = {
  MATCH_THRESHOLD: 0.4,
  LEARNED_MATCH_THRESHOLD: 0.75,
  MATCH_COUNT: 5,
  LEARNED_MATCH_COUNT: 3,
  CACHE_TTL_MS: 60 * 60 * 1000,
  CACHE_SIMILARITY_THRESHOLD: 0.97,
  CACHE_MAX_ENTRIES: 500,
  MAX_MESSAGE_LENGTH: 1000,
  MIN_MESSAGE_LENGTH: 2,
  RATE_LIMIT_WINDOW_MS: 60 * 1000,
  RATE_LIMIT_MAX: IS_PROD ? 15 : 50,
  ASK_RATE_LIMIT_MAX: IS_PROD ? 10 : 30,
  BODY_SIZE_LIMIT: '10kb',
  // FIX: This MUST be a real embedding model — NOT a generation model.
  // 'models/gemini-2.0-flash-lite' is a generation model and will ALWAYS fail for embedContent.
  EMBEDDING_MODEL: 'models/gemini-embedding-001',
  MODEL_COOLDOWN_MS: 60 * 1000,
  GEMINI_TIMEOUT_MS: 25_000,
  RERANK_CHUNK_PREVIEW: 250,
} as const;

const supabase = createClient(process.env.SUPABASE_URL!, process.env.SUPABASE_ANON_KEY!);

// ============================================================================
// ── 2. MODEL ROSTER & PER-MODEL CIRCUIT BREAKER
// ============================================================================

interface ModelSlot {
  label: string;
  modelName: string;
  client: GoogleGenerativeAI;
  coolUntil: number;
  quotaHits: number;
  successCount: number;
}

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
  // If everything is cooling, return all slots so we still try rather than hanging
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

// ============================================================================
// ── 3. TYPES & INTERFACES
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

function writeCache(embedding: number[], domain: string, response: Record<string, unknown>): void {
  if (semanticCache.size >= CONFIG.CACHE_MAX_ENTRIES) {
    const oldestKey = semanticCache.keys().next().value;
    if (oldestKey) semanticCache.delete(oldestKey);
    logger.info(`Cache eviction: max entries (${CONFIG.CACHE_MAX_ENTRIES}) reached.`);
  }
  const key = `${domain}:${crypto.randomUUID()}`;
  semanticCache.set(key, { embedding, response, expiresAt: Date.now() + CONFIG.CACHE_TTL_MS });
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
// ── 6. CORE AI ENGINE — TIMEOUT WRAPPER + PER-MODEL ROTATION
// ============================================================================

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  const timeout = new Promise<never>((_, reject) =>
    setTimeout(() => reject(new Error(`TIMEOUT: ${label} exceeded ${ms}ms`)), ms)
  );
  return Promise.race([promise, timeout]);
}

// The roster rotates automatically: each slot is tried in order.
// If a slot hits quota (429 / resource_exhausted), it gets cooled for 60s
// and the next slot in the roster is tried immediately.
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
        coolSlot(slot); // mark this slot cooling, loop continues to next
        continue;
      }
      throw error; // hard error — don't swallow it
    }
  }
  throw new Error('QUOTA_EXHAUSTED: All models on this roster are at quota. Try again shortly.');
}

async function generate(
  prompt: string,
  requireJson = false,
  temperature = 0.7,
): Promise<string> {
  try {
    return await generateWithRoster(prompt, primaryRoster, requireJson, temperature);
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    if (msg.startsWith('QUOTA_EXHAUSTED')) {
      logger.warn('[ROSTER] Primary roster exhausted — falling back to rerank key for generation.');
      return generateWithRoster(prompt, rerankRoster, requireJson, temperature);
    }
    throw err;
  }
}

// ============================================================================
// ── 7. EMBEDDINGS
// ── FIX: Removed the double `async function embedText(async function embedText(`
// ── syntax error, fixed the wrong model name ('gemini-embedding-2-flash' does
// ── not exist), and properly wired roster-aware slot rotation with cooldowns.
// ============================================================================

async function embedText(
  text: string,
  taskType: 'RETRIEVAL_QUERY' | 'RETRIEVAL_DOCUMENT',
): Promise<number[]> {
  // Use the same roster rotation pattern as generateWithRoster so quota hits
  // on the embedding model automatically cool the slot and try the next one.
  // Note: all slots share the same underlying API key, so we use primaryRoster[0]
  // for the client but honour slot cooldowns to respect per-slot rate limiting.
  const slots = activeSlots(primaryRoster);

  for (const slot of slots) {
    try {
      // EMBEDDING_MODEL is 'models/gemini-embedding-001' — a real embedding model.
      // Never use a generation model (gemini-2.0-flash-lite etc.) here; they will
      // always throw "not supported for embedContent" which isSkippableError catches,
      // causing every slot to cool and returning a useless zero vector.
      const embeddingModel = slot.client.getGenerativeModel({
        model: CONFIG.EMBEDDING_MODEL,
      });

      const result = await withTimeout(
        embeddingModel.embedContent({
          content: { parts: [{ text }], role: 'user' },
          taskType: taskType as never,
          outputDimensionality: 768,
        } as never),
        CONFIG.GEMINI_TIMEOUT_MS,
        `${slot.label}/embed`,
      );

      let embedding = result.embedding.values;

      // Safety: trim to 768 dims to match the Supabase vector column schema
      if (embedding.length > 768) embedding = embedding.slice(0, 768);

      slot.successCount++;
      logger.info(`[EMBED] ${slot.label} ✓ (dims: ${embedding.length})`);
      return embedding;

    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);

      if (msg.startsWith('TIMEOUT:')) {
        logger.warn(`[EMBED] ${slot.label} timed out — trying next slot.`);
        continue;
      }

      if (isSkippableError(err)) {
        coolSlot(slot);
        continue;
      }

      // Hard error (auth, bad request, etc.) — surface it immediately
      logger.error(`[EMBED] Hard error on ${slot.label}`, err);
      throw err;
    }
  }

  // True last resort: all slots are quota-exhausted. Return zero vector so the
  // request can still return a graceful no-match rather than a 500 crash.
  // The zero vector will produce no cosine similarity matches, which is correct
  // behaviour — we just won't find any rule chunks this cycle.
  logger.error('[EMBED] All slots exhausted. Returning zero vector — expect no rule matches.', new Error('EMBED_EXHAUSTED'));
  return new Array(768).fill(0);
}

const embedQuery    = (text: string) => embedText(text, 'RETRIEVAL_QUERY');
const embedDocument = (text: string) => embedText(text, 'RETRIEVAL_DOCUMENT');

// Kept as a named wrapper so we can later plug in HyDE / multi-query expansion
// without changing call sites.
async function expandAndAverageEmbedding(query: string): Promise<number[]> {
  return embedQuery(query);
}

// Suppress "declared but never used" — embedDocument is exported for ingestion scripts
void embedDocument;

// ============================================================================
// ── 8. RERANKER — cross-encoder via rerankRoster (token-efficient)
// ============================================================================

async function rerankChunks(query: string, chunks: RuleChunk[]): Promise<RuleChunk[]> {
  if (chunks.length <= 1) return chunks;

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
      throw new Error(`Score array length mismatch: got ${scores.length}, expected ${chunks.length}`);
    }

    return chunks
      .map((c, i) => ({ ...c, rerank_score: typeof scores[i] === 'number' ? scores[i] : c.similarity }))
      .sort((a, b) => (b.rerank_score ?? 0) - (a.rerank_score ?? 0));
  } catch (err) {
    logger.warn('[RERANKER] Failed — falling back to cosine order.', { error: String(err) });
    return chunks
      .sort((a, b) => b.similarity - a.similarity)
      .map(c => ({ ...c, rerank_score: c.similarity }));
  }
}

// ============================================================================
// ── 9. PROMPT INJECTION GUARD
// ============================================================================

const INJECTION_PATTERNS: RegExp[] = [
  /ignore (all )?(previous|prior|above) instructions/i,
  /you are now/i,
  /new (system )?prompt/i,
  /forget (everything|all)/i,
  /act as (a |an )?(?!engineer|assistant|regulations|technical)/i,
  /disregard (all |your )?(previous |prior )?instructions/i,
  /override (your )?(instructions|rules|guidelines)/i,
  /\[system\]/i,
  /<\/?system>/i,
  /prompt injection/i,
];

function detectInjection(text: string): boolean {
  return INJECTION_PATTERNS.some(p => p.test(text));
}

// ============================================================================
// ── 10. INTENT CLASSIFICATION & PROMPT BUILDING
// ============================================================================

function classifyIntent(query: string): QueryIntent {
  const q = query.toLowerCase();
  if (/\b(how (wide|tall|long|thick|deep)|dimension|size|length|width|height|weight|distance|radius|diameter|mm|cm|kg|newton|force|thickness|volume|area)\b/.test(q)) return 'dimension';
  if (/\b(legal|illegal|allowed|permitted|prohibited|pass|fail|comply|compliant|violation|violate|can i|is it ok|is .+ allowed)\b/.test(q)) return 'compliance';
  if (/\b(what is|define|definition|what does .+ mean|explain|describe)\b/.test(q)) return 'definition';
  if (/\b(how to|steps|procedure|process|install|mount|attach|assemble|test|inspect|check)\b/.test(q)) return 'procedure';
  return 'general';
}

function buildSystemPrompt(intent: QueryIntent, ruleContext: string, query: string, domain: string): string {
  const persona = `You are INDRA — the Integrated Neural Design and Regulations Assistant for Hexawatts Racing. You are precise, authoritative, and direct. Your tone is that of a senior technical engineer: confident, no fluff, no filler. Always cite Rule IDs inline like [T3.14].`;

  const formatRules = `
RESPONSE FORMAT RULES (follow strictly):
- Open with a single bold TL;DR sentence that directly answers the question. No preamble.
- Use clean markdown: ## for section headers, **bold** for values/rule IDs, bullet points for lists.
- Never say "Based on the provided context" or "According to the regulations" — just answer.
- End with a ⚡ INDRA NOTE if there is a critical caveat or enforcement tip worth flagging.
- Keep responses tight. No padding. Every sentence must carry information.`;

  const intentInstructions: Record<QueryIntent, string> = {
    dimension: `${persona}\n${formatRules}\n\nINTENT: DIMENSION QUERY\nStructure your answer as:\n**[Value + Unit]** — state the exact spec immediately.\n## Constraints\nBullet list of all related dimensional limits with rule IDs.\n## Exceptions / Conditionals\nAny conditional rules or edge cases.`,
    compliance: `${persona}\n${formatRules}\n\nINTENT: COMPLIANCE CHECK\nStructure your answer as:\n## Verdict\n✅ COMPLIANT / ❌ NON-COMPLIANT / ⚠️ CONDITIONAL — one line, bold.\n## Determining Rules\nCite exact rule IDs and what they require.\n## Conditions / Exceptions\nWhat would change the verdict.`,
    definition: `${persona}\n${formatRules}\n\nINTENT: DEFINITION\nStructure your answer as:\n**[Term]**: One crisp sentence definition.\n## Purpose / Function\nWhy this component or rule exists in context.\n## Rule Reference\nThe defining rule ID and its exact scope.`,
    procedure: `${persona}\n${formatRules}\n\nINTENT: PROCEDURE\nStructure your answer as:\n## Steps\nNumbered list. Each step cites the relevant rule ID inline.\n## Mandatory Checkpoints\nInspection or verification points that must not be skipped.`,
    general: `${persona}\n${formatRules}\n\nINTENT: GENERAL TECHNICAL QUERY\nAnswer directly and structured. Use ## headers to separate distinct topics. Cite rule IDs inline throughout.`,
  };

  return `${intentInstructions[intent]}\n\nDOMAIN: ${domain}\n\nREGULATION CONTEXT:\n${ruleContext}\n\nQUESTION: ${query}\n\nAnswer now. No preamble.`;
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

// ── FIX: fetchModelById was called in v13 but never defined — added here ──────
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

// ── Rule-based CAD node lookup ────────────────────────────────────────────────
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

// ── Keyword-based CAD lookup (secondary shield) ───────────────────────────────
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

    // Longest-match first: "front bulkhead" before "bulkhead"
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
    // FIX: was silently swallowing errors with no log — now logged
    logger.error('fetchCadByKeyword threw', err, { requestId });
    return null;
  }
}

// ============================================================================
// ── 11. MIDDLEWARE
// ============================================================================

async function requireAuth(req: Request, res: Response, next: NextFunction): Promise<void> {
  const authHeader = req.headers['authorization'];
  const token = authHeader?.startsWith('Bearer ') ? authHeader.slice(7) : null;

  if (!token) {
    res.status(401).json({ error: 'Unauthorized: No token provided.', code: 'NO_TOKEN' });
    return;
  }

  const validToken = process.env.API_AUTH_TOKEN!;
  const tokenBuf   = Buffer.from(token);
  const validBuf   = Buffer.from(validToken);
  if (tokenBuf.length === validBuf.length && crypto.timingSafeEqual(tokenBuf, validBuf)) {
    (req as AuthenticatedRequest).requestId = crypto.randomUUID();
    (req as AuthenticatedRequest).user = { role: 'admin' };
    return next();
  }

  try {
    const { data: { user }, error: authError } = await supabase.auth.getUser(token);
    if (authError || !user) {
      res.status(401).json({ error: 'Unauthorized: Invalid or expired session.', code: 'INVALID_TOKEN' });
      return;
    }

    const { data: teamMember, error: dbError } = await supabase
      .from('hexawatts_team')
      .select('is_approved, email')
      .eq('id', user.id)
      .single();

    if (dbError || !teamMember || teamMember.is_approved === false) {
      res.status(403).json({ error: 'Your account is pending team lead approval.', code: 'ACCOUNT_PENDING' });
      return;
    }

    (req as AuthenticatedRequest).requestId = crypto.randomUUID();
    (req as AuthenticatedRequest).user = { id: user.id, email: teamMember.email };
    next();
  } catch (err) {
    logger.error('Authentication error', err);
    res.status(500).json({ error: 'Authentication system error.', code: 'AUTH_ERROR' });
  }
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
// ── 12. ROUTES
// ============================================================================

app.get('/health', (_req: Request, res: Response): void => {
  const now = Date.now();
  const rosterSummary = (roster: ModelSlot[]) => ({
    total:   roster.length,
    active:  roster.filter(s => s.coolUntil <= now).length,
    cooling: roster.filter(s => s.coolUntil > now).length,
  });
  res.json({
    status: 'ok',
    service: 'INDRA RAG Backend',
    version: '13.2.0',
    uptime_seconds: Math.floor(process.uptime()),
    cache_size: semanticCache.size,
    primary_roster: rosterSummary(primaryRoster),
    rerank_roster:  rosterSummary(rerankRoster),
  });
});

app.get('/admin/keys', generalLimiter, requireAuth, (req: Request, res: Response): void => {
  if ((req as AuthenticatedRequest).user?.role !== 'admin') {
    res.status(403).json({ error: 'Admin only.', code: 'FORBIDDEN' });
    return;
  }
  const now = Date.now();
  const rosterStatus = (roster: ModelSlot[]) =>
    roster.map(s => ({
      label:        s.label,
      model:        s.modelName,
      status:       s.coolUntil > now ? 'cooling' : 'active',
      coolUntil:    s.coolUntil > now ? new Date(s.coolUntil).toISOString() : null,
      quotaHits:    s.quotaHits,
      successCount: s.successCount,
    }));
  res.json({
    primary_roster: rosterStatus(primaryRoster),
    rerank_roster:  rosterStatus(rerankRoster),
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

app.post('/admin/cache/clear', generalLimiter, requireAuth, (req: Request, res: Response): void => {
  if ((req as AuthenticatedRequest).user?.role !== 'admin') {
    res.status(403).json({ error: 'Admin only.', code: 'FORBIDDEN' });
    return;
  }
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

    // All three run in parallel — keyword CAD lookup costs zero extra latency
    const [learnedMatches, rerankedRules, keywordCadMatch] = await Promise.all([
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
      matchedRules && (matchedRules as RuleChunk[]).length > 1
        ? rerankChunks(trimmed, matchedRules as RuleChunk[])
        : Promise.resolve((matchedRules ?? []) as RuleChunk[]),
      fetchCadByKeyword(trimmed, requestId),
    ]);

    const hasRuleMatches    = rerankedRules.length > 0;
    const hasLearnedMatches = learnedMatches.length > 0;
    const ruleIds: string[] = rerankedRules
      .map(r => r.rule_id)
      .filter((id): id is string => Boolean(id));

    // ── 3D Model resolution ──────────────────────────────────────────────────
    let topModel: ModelRecord | null = null;
    let highlightMeshes: string[]    = [];
    let contextMeshes: string[]      = [];
    let cadNodes: CadNodeMatch[]     = [];

    // 1. PRIMARY: Handcrafted keyword map — 100% precision, always wins
    if (keywordCadMatch?.model_id) {
      topModel        = await fetchModelById(keywordCadMatch.model_id, requestId);
      highlightMeshes = keywordCadMatch.highlight_meshes ?? [];
      contextMeshes   = keywordCadMatch.context_meshes   ?? [];
      logger.info('[DEMO SAVED] Used exact keyword map', { requestId, model: topModel?.name });
    }
    // 2. SECONDARY: Fuzzy rule-tag lookup
    else if (hasRuleMatches && ruleIds.length > 0) {
      const { data: tagRows, error: tagError } = await supabase
        .from('model_rule_tags')
        .select('rule_id, relevance_score, fb_models(*)')
        .in('rule_id', ruleIds)
        .order('relevance_score', { ascending: false })
        .limit(1);

      if (tagError) logger.error('model_rule_tags query error', tagError, { requestId });
      if (tagRows && tagRows.length > 0) {
        topModel = ((tagRows[0] as Record<string, unknown>).fb_models as ModelRecord) ?? null;
      }

      cadNodes        = await fetchCadNodesForRules(ruleIds, requestId);
      highlightMeshes = [...new Set(cadNodes.map(n => n.cad_node_name))];
    }

    // 3. FINAL FALLBACK: Category keyword scan
    if (!topModel) {
      const categories = extractKeywordsFromQuery(trimmed);
      if (categories.length > 0) {
        const { data: modelsByKeyword } = await supabase
          .from('fb_models')
          .select('*')
          .in('category', categories)
          .limit(1);
        if (modelsByKeyword && modelsByKeyword.length > 0) {
          topModel = modelsByKeyword[0] as ModelRecord;
        }
      }
    }

    logger.info('[CAD] Mesh resolution complete', {
      requestId,
      highlight_count: highlightMeshes.length,
      context_count:   contextMeshes.length,
      source: keywordCadMatch?.model_id ? 'keyword-first' : 'rule-fallback',
    });

    // ── No match — return clean INDRA voice response, still include CAD ──────
    if (!hasRuleMatches && !hasLearnedMatches) {
      logger.info('No rulebook match found', { requestId, domain: sanitizedDomain });

      const noMatchPayload: Record<string, unknown> = {
        answer: `**No rulebook match found in the ${sanitizedDomain} domain.**\n\n⚡ INDRA NOTE: Try rephrasing with specific component names or rule keywords. Switch domain if this is a cross-domain query.`,
        citations: [],
        intent: 'no_match',
        code:   'NO_MATCH',
      };

      if (topModel?.file_url) {
        noMatchPayload.model_url      = topModel.file_url;
        noMatchPayload.model_metadata = buildModelMetadata(topModel, cadNodes, false);
      }
      if (highlightMeshes.length > 0) noMatchPayload.highlight_meshes = highlightMeshes;
      if (contextMeshes.length > 0)   noMatchPayload.context_meshes   = contextMeshes;

      void saveLog({ request_id: requestId, query: trimmed, result: 'no_match', domain: sanitizedDomain, intent, created_at: new Date().toISOString() });
      res.json(noMatchPayload);
      return;
    }

    // ── Generate answer ───────────────────────────────────────────────────────
    const ruleContext = [
      ...rerankedRules.map(r =>
        `[Rule ${r.rule_id}${typeof r.rerank_score === 'number' ? ` | Relevance: ${r.rerank_score.toFixed(2)}` : ''}]\n${r.content}`
      ),
      ...(hasLearnedMatches
        ? ['\n--- PREVIOUSLY VERIFIED ANSWERS (high confidence) ---',
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
        rule_id:      r.rule_id,
        content:      r.content,
        similarity:   r.similarity,
        rerank_score: r.rerank_score ?? null,
      })),
    };

    if (hasLearnedMatches) {
      responsePayload.learned_citations = learnedMatches.map(l => ({
        question: l.question,
        answer:   l.answer,
        source:   l.source,
      }));
    }

    if (topModel?.file_url) {
      responsePayload.model_url      = topModel.file_url;
      responsePayload.model_metadata = buildModelMetadata(topModel, cadNodes);
    }

    // highlight_meshes → orange (directly relevant)
    // context_meshes   → translucent (spatial reference)
    // cad_nodes        → kept for backwards compatibility
    responsePayload.highlight_meshes = highlightMeshes;
    responsePayload.context_meshes   = contextMeshes;
    responsePayload.cad_nodes        = cadNodes;

    writeCache(expandedEmbedding, sanitizedDomain, responsePayload);
    res.json(responsePayload);

    void saveLog({
      request_id:              requestId,
      query:                   trimmed,
      result:                  'success',
      domain:                  sanitizedDomain,
      intent,
      model_found:             !!topModel,
      model_name:              topModel?.name ?? null,
      citations_count:         rerankedRules.length,
      learned_citations_count: learnedMatches.length,
      cad_nodes_count:         cadNodes.length,
      highlight_meshes_count:  highlightMeshes.length,
      context_meshes_count:    contextMeshes.length,
      keyword_cad_hit:         !!keywordCadMatch,
      cache_written:           true,
      created_at:              new Date().toISOString(),
    });

  } catch (error: unknown) {
    const msg              = error instanceof Error ? error.message : String(error);
    const isQuotaExhausted = msg.includes('QUOTA_EXHAUSTED');
    const isTimeout        = msg.startsWith('TIMEOUT:');

    logger.error('Error in /ask_indra', error, { requestId });

    if (isQuotaExhausted) {
      res.status(503).json({ error: 'All AI capacity is temporarily at quota limit. Please try again in a minute.', code: 'QUOTA_EXHAUSTED' });
    } else if (isTimeout) {
      res.status(504).json({ error: 'The AI took too long to respond. Please try again.', code: 'TIMEOUT' });
    } else {
      res.status(500).json({ error: 'The server encountered an error. Please try again.', code: 'INTERNAL_ERROR' });
    }
  }
});

// ── Feedback & learning ──────────────────────────────────────────────────────
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
  const { id } = req.params;
  // FIX: length-cap the regex to prevent arbitrarily long ID strings
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
    const questions = JSON.parse(cleaned);

    if (!Array.isArray(questions)) throw new Error('Invalid quiz generation response');
    res.json({ questions, domain, generated: true });
  } catch (err) {
    logger.error('Quiz generation error', err);
    res.status(500).json({ error: 'Failed to generate quiz questions.', code: 'INTERNAL_ERROR' });
  }
});

// ============================================================================
// ── 13. ERROR HANDLING & STARTUP
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
  console.log(`\n🚀  INDRA RAG Backend v13.2.0`);
  console.log(`🌐  http://localhost:${PORT}`);
  console.log(`🌍  Environment  : ${IS_PROD ? 'PRODUCTION' : 'development'}`);
  console.log(`🛡️   Security     : Helmet + Rate Limiting + Timing-Safe Auth + Injection Guard`);
  console.log(`🔑  Primary key  : ${primaryRoster.length} models (${primaryRoster.map(s => s.modelName).join(' → ')})`);
  console.log(`🔑  Rerank key   : ${rerankRoster.length} models — cross-encoder reranking active`);
  console.log(`🧠  Embed model  : ${CONFIG.EMBEDDING_MODEL}`);
  console.log(`🧠  RAG Engine   : Embed → Vector Search → Cross-Encoder Rerank → CAD Match → Keyword Shield`);
  console.log(`🔩  CAD Layer    : Keyword-first → Rule-fallback → Category-scan`);
  console.log(`⚡  Cache        : Semantic in-memory | TTL: ${CONFIG.CACHE_TTL_MS / 60000}min | Max: ${CONFIG.CACHE_MAX_ENTRIES} entries`);
  console.log(`⏱️   Timeout      : ${CONFIG.GEMINI_TIMEOUT_MS / 1000}s per Gemini call`);
  console.log(`🚫  Fallback     : No generic AI fallback — INDRA answers from rulebook only\n`);
});