import { useState, useRef, useEffect, useCallback } from "react";
import {
  Send, ShieldCheck, Trash2, Copy,Plus,
  MessageSquare, BrainCircuit, CheckCircle2, XCircle,
  ChevronRight, Menu, X, LogOut,
  Box, Maximize2, Download, ZoomIn, ZoomOut, RotateCw,
  Eye, EyeOff, Play,
  Target, Activity, Zap, Loader2, Ruler,
  Crosshair, Minimize2, Layers, RefreshCw, AlertCircle
} from "lucide-react";
import { createClient, Session } from '@supabase/supabase-js';

// ─────────────────────────────────────────────────────────────────────────────
// TYPE DECLARATIONS
// ─────────────────────────────────────────────────────────────────────────────
declare global {
  namespace JSX {
    interface IntrinsicElements {
      'model-viewer': React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement> & {
        src?: string | null;
        alt?: string;
        'auto-rotate'?: boolean | string;
        'camera-controls'?: boolean | string;
        'shadow-intensity'?: string | number;
        'environment-image'?: string;
        exposure?: string | number;
        loading?: string;
        class?: string;
        style?: React.CSSProperties;
        ref?: any;
        crossorigin?: string;
        bounds?: string;
        'touch-action'?: string;
        'camera-orbit'?: string;
        'min-camera-orbit'?: string;
        zoom?: (amount: number) => void;
        resetTurntableRotation?: () => void;
      };
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────────────────────────────────────────
const getEnvVar = (key: string, fallback = '') => import.meta.env[key] || fallback;
const API_URL = getEnvVar('VITE_API_URL', 'http://localhost:8000');
const MAX_MESSAGE_LENGTH = 1000;
const RATE_LIMIT_DELAY = 1000;
const supabase = createClient(getEnvVar('VITE_SUPABASE_URL'), getEnvVar('VITE_SUPABASE_ANON_KEY'));

// ─────────────────────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────────────────────
interface Citation { rule_id: string; content: string; }
interface CadNode { rule_id: string; cad_node_name: string; relevance_score?: number; }
interface ModelMetadata {
  name: string;
  category: string;
  tags: string[];
  description?: string;
  fileSize?: string;
  cad_nodes?: CadNode[];
}
interface Message {
  id: string;
  role: "user" | "bot" | "error";
  text: string;
  citations?: Citation[];
  model_url?: string;
  model_metadata?: ModelMetadata;
  cad_nodes?: CadNode[];
  highlight_meshes?: string[];
  context_meshes?: string[];
  timestamp: number;
}
interface AuthMessage { type: 'error' | 'success'; text: string; }
interface QuizQuestion { question: string; options: string[]; correctAnswer: number; explanation: string; }
type FocusedPanel = null | 'sidebar' | 'chat' | '3d';
type IsolationMode = 'ghost' | 'hidden';

// ─────────────────────────────────────────────────────────────────────────────
// DATA & CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────
const QUIZ_QUESTIONS: QuizQuestion[] = [
  {
    question: "What is the minimum required force that the brake pedal system must be designed to withstand without failure?",
    options: ["1000 N", "1500 N", "2000 N", "2500 N"],
    correctAnswer: 2,
    explanation: "Rule T6.1.13: The brake pedal and its mounting must be designed to withstand a force of 2000 N without yielding."
  },
  {
    question: "Which material is strictly prohibited for use in the primary structure's main roll hoop?",
    options: ["Carbon Steel", "Aluminum Alloy", "Titanium", "Chromoly"],
    correctAnswer: 1,
    explanation: "Rule T3.2.1: Aluminum alloys are not permitted for the Main Roll Hoop or Front Roll Hoop."
  },
  {
    question: "What is the required color for the Cockpit Master Switch (Shutdown Button)?",
    options: ["Red", "Blue with Red outline", "Red with a Yellow background", "Black with a Red outline"],
    correctAnswer: 2,
    explanation: "Rule EV4.3.3: All shutdown buttons must be Red, mounted on a Yellow background for high visibility."
  }
];

const QUICK_QUERIES = [
  { label: "Front impact structure specs", icon: <ShieldCheck size={13} /> },
  { label: "Roll hoop material constraints", icon: <Target size={13} /> },
  { label: "Brake pedal force limits", icon: <Activity size={13} /> },
  { label: "IA foam dimensions", icon: <Ruler size={13} /> },
];

const sanitizeInput = (s: string) => s.trim().slice(0, MAX_MESSAGE_LENGTH);

// ─────────────────────────────────────────────────────────────────────────────
// RENDER TEXT WITH CITATIONS
// ─────────────────────────────────────────────────────────────────────────────
function renderTextWithCitations(text: string, citations?: Citation[]): React.ReactNode {
  if (!citations || citations.length === 0) return <span>{text}</span>;
  const ruleIndex = new Map<string, number>();
  citations.forEach((c, i) => ruleIndex.set(c.rule_id, i + 1));
  const parts = text.split(/\[CITE:([^\]]+)\]/g);
  return (
    <>
      {parts.map((part, i) => {
        if (i % 2 === 0) return <span key={i}>{part}</span>;
        const ruleId = part;
        const num = ruleIndex.get(ruleId) ?? '?';
        const citation = citations.find(c => c.rule_id === ruleId);
        return (
          <span
            key={i}
            title={citation ? `${ruleId}: ${citation.content}` : ruleId}
            className="inline-flex items-center justify-center w-4 h-4 mx-0.5 rounded-sm bg-[#FF2800] text-white text-[8px] font-black cursor-help align-middle leading-none"
          >
            {num}
          </span>
        );
      })}
    </>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// REUSABLE COMPONENTS
// ─────────────────────────────────────────────────────────────────────────────
function FocusButton({ panel, focusedPanel, onToggle }: { panel: FocusedPanel; focusedPanel: FocusedPanel; onToggle: (p: FocusedPanel) => void; }) {
  const isActive = focusedPanel === panel;
  return (
    <button
      onClick={() => onToggle(isActive ? null : panel)}
      title={isActive ? "Exit Focus Mode" : "Focus this panel"}
      className={`relative overflow-hidden flex items-center gap-1.5 px-3 py-2 md:px-2.5 md:py-1.5 text-[8px] font-black tracking-[0.25em] uppercase border transition-all duration-300 group
        ${isActive ? 'bg-[#FF2800] border-[#FF2800] text-white shadow-[0_0_16px_rgba(255,40,0,0.45)]' : 'bg-transparent border-white/[0.08] text-slate-600 hover:border-[#FF2800]/40 hover:text-[#FF2800]/80'}`}
    >
      <span className="absolute inset-0 bg-white/10 translate-x-[-110%] group-hover:translate-x-[110%] transition-transform duration-500 skew-x-12 pointer-events-none" />
      {isActive ? <Minimize2 size={9} className="shrink-0" /> : <Crosshair size={9} className="shrink-0" />}
      <span className="relative z-10 hidden md:inline">{isActive ? 'EXIT' : 'FOCUS'}</span>
    </button>
  );
}

function PanelShell({ id, focusedPanel, children, className = '' }: { id: FocusedPanel; focusedPanel: FocusedPanel; children: React.ReactNode; className?: string; }) {
  const isFocused = focusedPanel === id;
  const isDimmed = focusedPanel !== null && !isFocused;
  return (
    <div className={`transition-all duration-500 ease-out relative ${isDimmed ? 'opacity-[0.15] blur-[2px] pointer-events-none saturate-0' : 'opacity-100 blur-0 pointer-events-auto saturate-100'} ${isFocused ? 'ring-1 ring-[#FF2800]/20 shadow-[0_0_40px_rgba(255,40,0,0.06)] z-10' : ''} ${className}`}>
      {children}
    </div>
  );
}

function TelemetryStrip() {
  const [tick, setTick] = useState(0);
  useEffect(() => { const id = setInterval(() => setTick(t => t + 1), 1200); return () => clearInterval(id); }, []);
  const rpm = (6800 + Math.sin(tick * 0.7) * 800).toFixed(0);
  const temp = (92 + Math.sin(tick * 0.3) * 4).toFixed(1);
  const voltage = (396 + Math.sin(tick * 0.5) * 3).toFixed(1);
  return (
    <div className="hidden md:flex items-center gap-6 text-[9px] font-mono tracking-widest text-slate-500 select-none">
      <span className="flex items-center gap-1.5"><span className="text-[#FF2800]">RPM</span> {rpm}</span><span className="w-px h-3 bg-white/10" />
      <span className="flex items-center gap-1.5"><span className="text-[#FF2800]">TEMP</span> {temp}°C</span><span className="w-px h-3 bg-white/10" />
      <span className="flex items-center gap-1.5"><span className="text-[#FF2800]">HV</span> {voltage}V</span><span className="w-px h-3 bg-white/10" />
      <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-[#FF2800] animate-pulse shadow-[0_0_6px_#FF2800]" /><span className="text-[#FF2800]">LIVE</span></span>
    </div>
  );
}

function Scanlines() { return <div className="pointer-events-none absolute inset-0 z-0 opacity-[0.025]" style={{ backgroundImage: 'repeating-linear-gradient(to bottom, transparent, transparent 2px, rgba(255,255,255,0.4) 2px, rgba(255,255,255,0.4) 4px)' }} />; }

// ── Toast notification ──
function Toast({ message, type, onClose }: { message: string; type: 'error' | 'success'; onClose: () => void }) {
  useEffect(() => { const t = setTimeout(onClose, 3500); return () => clearTimeout(t); }, [onClose]);
  return (
    <div className={`fixed bottom-6 right-6 z-[100] flex items-center gap-3 px-4 py-3 text-xs font-bold shadow-2xl border animate-in slide-in-from-bottom-4 duration-300
      ${type === 'error' ? 'bg-red-950 border-red-700/50 text-red-300' : 'bg-green-950 border-green-700/50 text-green-300'}`}>
      {type === 'error' ? <AlertCircle size={14} /> : <CheckCircle2 size={14} />}
      {message}
      <button onClick={onClose} className="ml-2 opacity-60 hover:opacity-100"><X size={12} /></button>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MESH MATCHING UTILITY
// ─────────────────────────────────────────────────────────────────────────────
function matchesMeshName(matName: string, targets: string[]): boolean {
  if (targets.length === 0) return false;
  const clean = (s: string) => s.toLowerCase().replace(/[_\-\s]/g, '');
  const cleanMat = clean(matName);
  return targets.some(target => {
    const cleanTarget = clean(target);
    return (
      matName === target ||
      matName.toLowerCase() === target.toLowerCase() ||
      cleanMat === cleanTarget ||
      cleanMat.includes(cleanTarget) ||
      cleanTarget.includes(cleanMat)
    );
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN APPLICATION
// ─────────────────────────────────────────────────────────────────────────────
export default function IndraWorkspace() {
  // Auth State
  const [session, setSession] = useState<Session | null>(null);
  const [isAuthLoading, setIsAuthLoading] = useState(false);
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [isSignUp, setIsSignUp] = useState(false);
  const [authMessage, setAuthMessage] = useState<AuthMessage | null>(null);

  // App State
  const [appMode, setAppMode] = useState<"ask" | "quiz">("ask");
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [focusedPanel, setFocusedPanel] = useState<FocusedPanel>(null);
  const [toast, setToast] = useState<{ message: string; type: 'error' | 'success' } | null>(null);

  const toggleFocus = useCallback((panel: FocusedPanel) => setFocusedPanel(prev => prev === panel ? null : panel), []);
  const showToast = useCallback((message: string, type: 'error' | 'success') => setToast({ message, type }), []);

  // Chat State
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isThinking, setIsThinking] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const [lastMessageTime, setLastMessageTime] = useState(0);

  // 3D State
  const [activeModelUrl, setActiveModelUrl] = useState<string | null>(null);
  const [activeModelMetadata, setActiveModelMetadata] = useState<ModelMetadata | null>(null);
  const [activeCadNodes, setActiveCadNodes] = useState<CadNode[]>([]);
  const [modelParts, setModelParts] = useState<string[]>([]);
  const [highlightMeshes, setHighlightMeshes] = useState<string[]>([]);
  const [contextMeshes, setContextMeshes] = useState<string[]>([]);
  const [isolatedParts, setIsolatedParts] = useState<string[]>([]);
  const [highlightSelected, setHighlightSelected] = useState(true);
  const [isolationMode, setIsolationMode] = useState<IsolationMode>('ghost');
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [modelLoadProgress, setModelLoadProgress] = useState(0);
  const [is3DFullscreen, setIs3DFullscreen] = useState(false);
  const [showModelInfo, setShowModelInfo] = useState(true);
  const [autoRotateEnabled, setAutoRotateEnabled] = useState(true);
  const [showDebug, setShowDebug] = useState(false);

  // Quiz State
  const [qIndex, setQIndex] = useState(0);
  const [selectedAns, setSelectedAns] = useState<number | null>(null);
  const [isAnsChecked, setIsAnsChecked] = useState(false);
  const [quizScore, setQuizScore] = useState(0);
  const [quizFinished, setQuizFinished] = useState(false);
  // BUG 5 FIX: track if score was already counted for this question
  const [scoreCounted, setScoreCounted] = useState(false);

  // Refs
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const modelViewerRef = useRef<any>(null);
  const originalMaterialsRef = useRef<Map<string, number[]>>(new Map());
  // BUG 3 FIX: use a ref for handle3DModelLoad so sendMessage can safely call it before definition
  const handle3DModelLoadRef = useRef<Function | null>(null);

  // ── Effects ──
  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => setSession(session));
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_e, s) => setSession(s));
    return () => subscription.unsubscribe();
  }, []);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, isThinking]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') setFocusedPanel(null); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  // ─────────────────────────────────────────────────────────────────────────
  // CORE 3D MATERIAL LOGIC
  // ─────────────────────────────────────────────────────────────────────────
  const applyMeshStates = useCallback(() => {
    const mv = modelViewerRef.current;
    if (!mv?.model?.materials) return;
    const materials = mv.model.materials;
    const activeTargets = isolatedParts.length > 0 ? isolatedParts : highlightMeshes;
    const siblingTargets = isolatedParts.length > 0
      ? highlightMeshes.filter(m => !isolatedParts.includes(m))
      : [];

    materials.forEach((mat: any, index: number) => {
      const matName = mat.name || `Unnamed_Material_${index}`;
      mat.name = matName;
      const original = originalMaterialsRef.current.get(matName) ?? [0.7, 0.7, 0.7, 1.0];
      const isActive  = matchesMeshName(matName, activeTargets);
      const isSibling = matchesMeshName(matName, siblingTargets);
      const isContext = matchesMeshName(matName, contextMeshes);

      if (isActive) {
        if (highlightSelected) {
          mat.pbrMetallicRoughness.setBaseColorFactor([1.0, 0.22, 0.08, 1.0]);
          mat.pbrMetallicRoughness.setMetallicFactor(0.2);
          mat.pbrMetallicRoughness.setRoughnessFactor(0.25);
        } else {
          mat.pbrMetallicRoughness.setBaseColorFactor([...original]);
          mat.pbrMetallicRoughness.setMetallicFactor(0.3);
          mat.pbrMetallicRoughness.setRoughnessFactor(0.2);
        }
        mat.setAlphaMode('OPAQUE');
      } else if (isSibling) {
        mat.pbrMetallicRoughness.setBaseColorFactor([1.0, 0.22, 0.08, 0.35]);
        mat.pbrMetallicRoughness.setMetallicFactor(0.4);
        mat.pbrMetallicRoughness.setRoughnessFactor(0.3);
        mat.setAlphaMode('BLEND');
      } else if (isContext) {
        mat.pbrMetallicRoughness.setBaseColorFactor([0.35, 0.38, 0.45, 0.12]);
        mat.pbrMetallicRoughness.setMetallicFactor(0.9);
        mat.pbrMetallicRoughness.setRoughnessFactor(0.05);
        mat.setAlphaMode('BLEND');
      } else {
        if (isolationMode === 'hidden') {
          mat.pbrMetallicRoughness.setBaseColorFactor([0, 0, 0, 0]);
          mat.setAlphaMode('BLEND');
        } else {
          mat.pbrMetallicRoughness.setBaseColorFactor([0.1, 0.12, 0.18, 0.04]);
          mat.pbrMetallicRoughness.setMetallicFactor(0.95);
          mat.pbrMetallicRoughness.setRoughnessFactor(0.02);
          mat.setAlphaMode('BLEND');
        }
      }
    });
    if (typeof mv.queueRender === 'function') mv.queueRender();
  }, [isolatedParts, highlightSelected, highlightMeshes, contextMeshes, isolationMode]);

  useEffect(() => {
    if (activeModelUrl) applyMeshStates();
  }, [isolatedParts, highlightSelected, highlightMeshes, contextMeshes, isolationMode, applyMeshStates, activeModelUrl]);

  // BUG 2 FIX: Stable effect — applyMeshStates is called directly inside the handler
  // so it doesn't need to be in the dep array. The cleanup correctly removes the exact
  // same handler function references that were added.
  useEffect(() => {
    const mv = modelViewerRef.current;
    if (!mv) return;

    const handleModelReady = () => {
      const materials = mv.model?.materials;
      if (!materials) return;
      const parts: string[] = [];
      materials.forEach((mat: any, index: number) => {
        const matName = mat.name || `Unnamed_Material_${index}`;
        mat.name = matName;
        parts.push(matName);
        if (!originalMaterialsRef.current.has(matName)) {
          const color = mat.pbrMetallicRoughness?.baseColorFactor ?? [0.7, 0.7, 0.7, 1];
          originalMaterialsRef.current.set(matName, [...color]);
        }
      });
      setModelParts(parts);
      applyMeshStates();
    };

    // BUG 4 FIX: Use model-viewer's real 'progress' event instead of fake setInterval
    const handleProgress = (e: any) => {
      const pct = Math.round((e.detail?.totalProgress ?? 0) * 100);
      setModelLoadProgress(pct);
      if (pct >= 100) setIsModelLoading(false);
    };

    mv.addEventListener('load', handleModelReady);
    mv.addEventListener('scene-graph-ready', handleModelReady);
    mv.addEventListener('progress', handleProgress);
    if (mv.model) handleModelReady();

    return () => {
      mv.removeEventListener('load', handleModelReady);
      mv.removeEventListener('scene-graph-ready', handleModelReady);
      mv.removeEventListener('progress', handleProgress);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeModelUrl]); // applyMeshStates intentionally excluded — called directly inside handler

  useEffect(() => {
    if (activeModelUrl) {
      originalMaterialsRef.current.clear();
      setModelParts([]);
      setIsolatedParts([]);
    }
  }, [activeModelUrl]);

  // ── Auth ──
  // Minor fix: wrapped in useCallback for consistency
  const handleAuthSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    setIsAuthLoading(true);
    setAuthMessage(null);
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(authEmail)) {
      setAuthMessage({ type: 'error', text: 'Invalid email address' });
      setIsAuthLoading(false);
      return;
    }
    if (authPassword.length < 6) {
      setAuthMessage({ type: 'error', text: 'Password must be ≥ 6 characters' });
      setIsAuthLoading(false);
      return;
    }
    try {
      if (isSignUp) {
        const { error } = await supabase.auth.signUp({ email: authEmail, password: authPassword });
        if (error) throw error;
        setAuthMessage({ type: 'success', text: 'Access request submitted.' });
      } else {
        const { error } = await supabase.auth.signInWithPassword({ email: authEmail, password: authPassword });
        if (error) throw error;
      }
    } catch (err: any) {
      setAuthMessage({ type: 'error', text: err.message || 'Authentication failed.' });
    } finally {
      setIsAuthLoading(false);
    }
  }, [authEmail, authPassword, isSignUp]);

  const copyToClipboard = useCallback(async (text: string, id: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(id);
      setTimeout(() => setCopied(null), 2000);
    } catch {
      showToast('Failed to copy to clipboard', 'error');
    }
  }, [showToast]);

  // BUG 3 FIX: define handle3DModelLoad BEFORE sendMessage, and also assign to ref
  const handle3DModelLoad = useCallback((
    url: string,
    meta?: ModelMetadata,
    nodes?: CadNode[],
    highlights: string[] = [],
    contexts: string[] = [],
  ) => {
    originalMaterialsRef.current.clear();
    setActiveModelUrl(url);
    setActiveModelMetadata(meta ?? null);
    setActiveCadNodes(nodes ?? []);
    setHighlightMeshes(highlights);
    setContextMeshes(contexts);
    setIsolatedParts([]);
    setModelParts([]);
    setIsModelLoading(true);
    setModelLoadProgress(0);
    // BUG 4 FIX: No fake setInterval — real progress comes from model-viewer 'progress' event in useEffect
  }, []);

  // Keep ref in sync so sendMessage can call it safely
  useEffect(() => { handle3DModelLoadRef.current = handle3DModelLoad; }, [handle3DModelLoad]);

  // ── Send Message ──
  const sendMessage = useCallback(async (text: string) => {
    const sanitized = sanitizeInput(text);
    if (!sanitized || isThinking) return;
    const now = Date.now();
    if (now - lastMessageTime < RATE_LIMIT_DELAY) return;
    setLastMessageTime(now);
    setMessages(p => [...p, { id: crypto.randomUUID(), role: "user", text: sanitized, timestamp: now }]);
    setInput("");
    setIsThinking(true);

    // BUG 1 FIX: Only reset 3D state if a model is not currently loading
    if (!isModelLoading) {
      setActiveModelUrl(null);
      setActiveModelMetadata(null);
      setActiveCadNodes([]);
      setHighlightMeshes([]);
      setContextMeshes([]);
      setIsolatedParts([]);
      setModelParts([]);
    }

    try {
      if (!session?.access_token) throw new Error('Authentication token missing.');
      const res = await fetch(`${API_URL}/ask_indra`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${session.access_token}` },
        body: JSON.stringify({ message: sanitized, domain: "Formula Bharat 2027 Full" }),
      });
      if (!res.ok) throw new Error(`Telemetry uplink failure: ${res.status}`);
      const data = await res.json();

      setMessages(p => [...p, {
        id: crypto.randomUUID(),
        role: "bot",
        text: data.answer,
        citations: data.citations,
        model_url: data.model_url,
        model_metadata: data.model_metadata,
        cad_nodes: data.cad_nodes,
        highlight_meshes: data.highlight_meshes ?? [],
        context_meshes: data.context_meshes ?? [],
        timestamp: Date.now()
      }]);

      if (data.model_url && handle3DModelLoadRef.current) {
        handle3DModelLoadRef.current(
          data.model_url,
          data.model_metadata,
          data.cad_nodes,
          data.highlight_meshes ?? [],
          data.context_meshes ?? [],
        );
      }
    } catch (err: any) {
      setMessages(p => [...p, { id: crypto.randomUUID(), role: "error", text: err.message, timestamp: Date.now() }]);
      showToast(err.message, 'error');
    } finally {
      setIsThinking(false);
      inputRef.current?.focus();
    }
  }, [isThinking, lastMessageTime, session, isModelLoading, showToast]);

  const close3DModel = useCallback(() => {
    setActiveModelUrl(null);
    setActiveModelMetadata(null);
    setActiveCadNodes([]);
    setHighlightMeshes([]);
    setContextMeshes([]);
    setIsolatedParts([]);
    setModelParts([]);
    setIs3DFullscreen(false);
    if (focusedPanel === '3d') setFocusedPanel(null);
  }, [focusedPanel]);

  const togglePartIsolation = useCallback((partName: string) => {
    setIsolatedParts(prev =>
      prev.includes(partName) ? prev.filter(p => p !== partName) : [...prev, partName]
    );
  }, []);

  const clearIsolation = useCallback(() => { setIsolatedParts([]); }, []);

  const downloadModel = useCallback(async () => {
    if (!activeModelUrl) return;
    try {
      const res = await fetch(activeModelUrl);
      if (!res.ok) throw new Error(`Download failed: ${res.status}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = activeModelMetadata?.name || 'model.glb';
      document.body.appendChild(a);
      a.click();
      URL.revokeObjectURL(url);
      document.body.removeChild(a);
      showToast('Model downloaded successfully', 'success');
    } catch (err: any) {
      showToast(`Download failed: ${err.message}`, 'error');
    }
  }, [activeModelUrl, activeModelMetadata, showToast]);

  const zoomIn  = useCallback(() => modelViewerRef.current?.zoom(-1), []);
  const zoomOut = useCallback(() => modelViewerRef.current?.zoom(1), []);
  const reset3DCamera = useCallback(() => modelViewerRef.current?.resetTurntableRotation(), []);

  const cleanPartName = (name: string): string =>
    name
      .replace(/_mat$/gi, '')
      .replace(/^[0-9]+_/, '')
      .replace(/_/g, ' ')
      .replace(/\b\w/g, l => l.toUpperCase())
      .trim() || name;

  const clearChat = useCallback(() => {
    setMessages([]);
    setActiveModelUrl(null);
    setActiveModelMetadata(null);
    setActiveCadNodes([]);
    setHighlightMeshes([]);
    setContextMeshes([]);
    setIsolatedParts([]);
    setModelParts([]);
  }, []);

  // ─────────────────────────────────────────────────────────────────────────────
  // LOGIN SCREEN
  // ─────────────────────────────────────────────────────────────────────────────
  if (!session) {
    return (
      <div className="relative flex min-h-screen items-center justify-center bg-[#060606] overflow-hidden" style={{ fontFamily: "'Rajdhani', 'DIN Next', system-ui, sans-serif" }}>
        <div className="absolute inset-0 opacity-[0.04]" style={{ backgroundImage: 'linear-gradient(rgba(255,40,0,0.6) 1px, transparent 1px), linear-gradient(90deg, rgba(255,40,0,0.6) 1px, transparent 1px)', backgroundSize: '60px 60px' }} />

        {/* Animated accent lines */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-px h-32 bg-gradient-to-b from-transparent to-[#FF2800]/40" />
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-px h-32 bg-gradient-to-t from-transparent to-[#FF2800]/40" />

        <div className="relative z-10 w-full max-w-[420px] px-4">
          <div className="text-center mb-10">
            {/* Logo mark */}
            <div className="inline-flex items-center justify-center w-14 h-14 border border-[#FF2800]/30 mb-6 relative">
              <Zap size={22} className="text-[#FF2800]" />
              <span className="absolute inset-0 border border-[#FF2800]/10 scale-110" />
            </div>
            <h1 className="text-5xl font-black text-white tracking-tighter">INDRA</h1>
            <p className="text-[#FF2800] text-sm font-bold tracking-[0.25em] mt-1">FORMULA BHARAT 2027</p>
            <p className="text-slate-600 text-xs mt-3 tracking-widest uppercase">Regulation Intelligence System</p>
          </div>

          <form onSubmit={handleAuthSubmit} className="space-y-3 bg-[#0c0c0c] border border-white/[0.06] p-8">
            <div className="relative">
              <input
                type="email"
                required
                value={authEmail}
                onChange={e => setAuthEmail(e.target.value)}
                className="w-full bg-[#111] border border-white/[0.07] px-4 py-3 text-white focus:border-[#FF2800]/40 outline-none transition-colors peer"
                placeholder=" "
                id="auth-email"
              />
              <label htmlFor="auth-email" className="absolute left-4 top-3 text-slate-600 text-sm pointer-events-none transition-all peer-focus:-top-2.5 peer-focus:text-[10px] peer-focus:text-[#FF2800]/70 peer-[:not(:placeholder-shown)]:-top-2.5 peer-[:not(:placeholder-shown)]:text-[10px] bg-[#111] px-1">Email</label>
            </div>
            <div className="relative">
              <input
                type="password"
                required
                value={authPassword}
                onChange={e => setAuthPassword(e.target.value)}
                className="w-full bg-[#111] border border-white/[0.07] px-4 py-3 text-white focus:border-[#FF2800]/40 outline-none transition-colors peer"
                placeholder=" "
                id="auth-password"
              />
              <label htmlFor="auth-password" className="absolute left-4 top-3 text-slate-600 text-sm pointer-events-none transition-all peer-focus:-top-2.5 peer-focus:text-[10px] peer-focus:text-[#FF2800]/70 peer-[:not(:placeholder-shown)]:-top-2.5 peer-[:not(:placeholder-shown)]:text-[10px] bg-[#111] px-1">Password</label>
            </div>

            {authMessage && (
              <div className={`flex items-center gap-2 text-xs font-bold p-3 ${authMessage.type === 'success' ? 'bg-green-900/30 text-green-400 border border-green-800/30' : 'bg-red-900/30 text-red-400 border border-red-800/30'}`}>
                {authMessage.type === 'error' ? <AlertCircle size={13} /> : <CheckCircle2 size={13} />}
                {authMessage.text}
              </div>
            )}

            <button
              type="submit"
              disabled={isAuthLoading}
              className="relative w-full py-4 bg-[#FF2800] text-white font-black text-[11px] tracking-[0.3em] uppercase hover:bg-[#FF4000] transition-colors overflow-hidden group disabled:opacity-60"
            >
              <span className="absolute inset-0 bg-white/10 translate-x-[-110%] group-hover:translate-x-[110%] transition-transform duration-500 skew-x-12" />
              {isAuthLoading ? (
                <span className="flex items-center justify-center gap-2"><Loader2 size={14} className="animate-spin" /> AUTHENTICATING...</span>
              ) : isSignUp ? "CREATE ACCOUNT" : "LAUNCH INDRA"}
            </button>

            <div className="text-center pt-1">
              <button type="button" onClick={() => { setIsSignUp(!isSignUp); setAuthMessage(null); }} className="text-slate-500 text-xs hover:text-slate-300 transition-colors">
                {isSignUp ? "Already have access? Sign in" : "Request new access →"}
              </button>
            </div>
          </form>

          <p className="text-center text-slate-700 text-[10px] mt-6 tracking-widest uppercase">Authorized personnel only</p>
        </div>
      </div>
    );
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // MAIN WORKSPACE
  // ─────────────────────────────────────────────────────────────────────────────
  return (
    <div className="flex h-[100dvh] w-full overflow-hidden bg-[#060606] text-slate-200 relative select-none" style={{ fontFamily: "'Rajdhani', 'DIN Next', system-ui, sans-serif" }}>
      <Scanlines />
      <div className="absolute inset-0 opacity-[0.025] pointer-events-none" style={{ backgroundImage: 'linear-gradient(rgba(255,40,0,0.5) 1px, transparent 1px), linear-gradient(90deg, rgba(255,40,0,0.5) 1px, transparent 1px)', backgroundSize: '80px 80px' }} />

      {focusedPanel !== null && <div className="pointer-events-none fixed inset-0 z-[15] bg-black/25 transition-opacity duration-500" />}
      {focusedPanel !== null && (
        <div className="fixed top-3 left-1/2 -translate-x-1/2 z-[60] pointer-events-auto animate-in fade-in slide-in-from-top-2 duration-300">
          <button onClick={() => setFocusedPanel(null)} className="flex items-center gap-2 px-4 py-2 bg-black/95 border border-[#FF2800]/25 text-[9px] font-black tracking-[0.3em] uppercase text-[#FF2800]/80 shadow-2xl">
            <Minimize2 size={9} /> EXIT FOCUS MODE
          </button>
        </div>
      )}

      {/* Toast */}
      {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}

      {isSidebarOpen && <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-[45] md:hidden" onClick={() => setIsSidebarOpen(false)} />}

      {/* ─── SIDEBAR ─────────────────────────────────────────────────────────── */}
      <PanelShell id="sidebar" focusedPanel={focusedPanel} className={`fixed inset-y-0 left-0 z-50 w-[280px] md:w-[260px] h-full md:relative md:translate-x-0 transition-transform duration-300 ease-out ${isSidebarOpen ? 'translate-x-0 shadow-2xl' : '-translate-x-full'}`}>
        <aside className="flex flex-col h-full bg-[#080808] border-r border-white/[0.05]">
          <div className="px-6 pt-6 pb-5 border-b border-white/[0.05] flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="relative">
                <Zap size={16} className="text-[#FF2800]" />
                <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 bg-[#FF2800] rounded-full animate-pulse" />
              </div>
              <div className="text-[11px] font-black text-white tracking-wider">INDRA OS</div>
            </div>
            <div className="flex items-center gap-2">
              <FocusButton panel="sidebar" focusedPanel={focusedPanel} onToggle={toggleFocus} />
              <button onClick={() => setIsSidebarOpen(false)} className="md:hidden p-2 text-slate-500"><X size={18} /></button>
            </div>
          </div>

          <div className="p-4 space-y-1">
            <button
              onClick={() => { clearChat(); setIsSidebarOpen(false); }}
              className="w-full flex items-center gap-2.5 px-4 py-3 bg-[#FF2800]/10 border border-[#FF2800]/20 text-[10px] font-bold text-[#FF2800] uppercase hover:bg-[#FF2800]/15 transition-colors group"
            >
              <Plus size={13} className="group-hover:rotate-90 transition-transform duration-200" /> New Query
            </button>
          </div>

          <div className="px-4 mb-2 flex-1">
            <p className="text-[8px] font-bold text-slate-700 uppercase tracking-[0.2em] px-3 mb-2">Navigation</p>
            <button
              onClick={() => { setAppMode("ask"); setIsSidebarOpen(false); }}
              className={`w-full flex items-center gap-3 px-3 py-3 mb-1 text-left transition-colors rounded-sm ${appMode === 'ask' ? 'text-[#FF2800] bg-[#FF2800]/5' : 'text-slate-500 hover:text-slate-300'}`}
            >
              <MessageSquare size={13} />
              <div className="text-[10px] font-bold uppercase">Regulation Query</div>
              {appMode === 'ask' && <div className="ml-auto w-1 h-1 bg-[#FF2800] rounded-full" />}
            </button>
            <button
              onClick={() => { setAppMode("quiz"); setIsSidebarOpen(false); }}
              className={`w-full flex items-center gap-3 px-3 py-3 text-left transition-colors rounded-sm ${appMode === 'quiz' ? 'text-[#FF2800] bg-[#FF2800]/5' : 'text-slate-500 hover:text-slate-300'}`}
            >
              <BrainCircuit size={13} />
              <div className="text-[10px] font-bold uppercase">Compliance Test</div>
              {appMode === 'quiz' && <div className="ml-auto w-1 h-1 bg-[#FF2800] rounded-full" />}
            </button>
          </div>

          {/* Session info */}
          <div className="px-6 py-4 border-t border-white/[0.05]">
            <p className="text-[8px] text-slate-700 uppercase tracking-widest mb-2">Session</p>
            <p className="text-[9px] text-slate-500 truncate mb-3">{session.user?.email}</p>
            <button
              onClick={() => supabase.auth.signOut()}
              className="flex items-center gap-2 text-[9px] text-slate-600 hover:text-[#FF2800] uppercase tracking-widest transition-colors"
            >
              <LogOut size={11} /> Sign Out
            </button>
          </div>
        </aside>
      </PanelShell>

      {/* ─── MAIN FRAME ──────────────────────────────────────────────────────── */}
      <main className="flex-1 flex flex-col h-full min-w-0 relative z-10">
        <header className="shrink-0 h-12 flex items-center justify-between px-4 md:px-6 border-b border-white/[0.05] bg-black/60 backdrop-blur-md relative z-30">
          <div className="flex items-center gap-3">
            <button onClick={() => setIsSidebarOpen(true)} className="md:hidden p-2 -ml-2 text-slate-500 hover:text-white transition-colors"><Menu size={20} /></button>
            <div className="flex items-center gap-2 text-[10px] md:text-[9px] font-bold tracking-[0.25em] uppercase">
              <span className="text-slate-700">INDRA</span>
              <ChevronRight size={10} className="text-slate-800" />
              <span className="text-slate-400">{appMode === 'ask' ? 'QUERY ENGINE' : 'COMPLIANCE TEST'}</span>
            </div>
          </div>
          <TelemetryStrip />
        </header>

        <div className="flex-1 flex flex-col md:flex-row overflow-hidden relative">

          {/* ── CHAT PANEL ── */}
          <PanelShell id="chat" focusedPanel={focusedPanel} className={`flex flex-col transition-all duration-500 ease-out ${activeModelUrl ? 'flex-1 md:h-full md:w-1/2 border-b md:border-b-0 md:border-r border-white/[0.05]' : 'h-full w-full'}`}>

            {appMode === 'ask' && (
              <>
                <div className="shrink-0 flex items-center justify-between px-4 md:px-6 pt-3 pb-1">
                  <span className="text-[8px] font-bold tracking-[0.3em] text-slate-700 uppercase">Query Engine</span>
                  <div className="flex items-center gap-2">
                    {messages.length > 0 && (
                      <button
                        onClick={clearChat}
                        title="Clear conversation"
                        className="flex items-center gap-1 px-2 py-1 text-[8px] font-bold text-slate-600 hover:text-[#FF2800] border border-transparent hover:border-[#FF2800]/20 transition-all uppercase tracking-wider"
                      >
                        <Trash2 size={9} /> Clear
                      </button>
                    )}
                    <FocusButton panel="chat" focusedPanel={focusedPanel} onToggle={toggleFocus} />
                  </div>
                </div>

                <div className="flex-1 overflow-y-auto px-4 md:px-6 py-4 scroll-smooth">
                  {messages.length === 0 && !isThinking ? (
                    <div className="h-full flex flex-col items-center justify-center max-w-xl mx-auto text-center">
                      <div className="relative mb-6">
                        <Target size={28} className="text-[#FF2800]/70" />
                        <div className="absolute inset-0 blur-xl bg-[#FF2800]/20 rounded-full" />
                      </div>
                      <h2 className="text-xl md:text-2xl font-black tracking-tight text-white mb-2 uppercase">Ready for Input</h2>
                      <p className="text-slate-600 text-xs mb-8 tracking-wider">Ask anything about Formula Bharat 2027 regulations</p>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full">
                        {QUICK_QUERIES.map((q, i) => (
                          <button
                            key={i}
                            onClick={() => { setInput(q.label); inputRef.current?.focus(); }}
                            className="flex items-center gap-3 px-4 py-3 md:py-4 bg-[#0c0c0c] border border-white/[0.06] hover:border-[#FF2800]/30 hover:bg-[#FF2800]/5 text-left transition-all group"
                          >
                            <span className="text-[#FF2800]/50 shrink-0 group-hover:text-[#FF2800]/80 transition-colors">{q.icon}</span>
                            <span className="text-[10px] font-bold tracking-wider uppercase text-slate-400 group-hover:text-slate-200 transition-colors">{q.label}</span>
                            <ChevronRight size={9} className="ml-auto text-slate-700 group-hover:text-[#FF2800]/50 transition-colors" />
                          </button>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="max-w-3xl mx-auto space-y-6 pb-6">
                      {messages.map(msg => (
                        <div key={msg.id} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                          <span className={`text-[8px] tracking-[0.3em] uppercase mb-1.5 font-bold ${msg.role === 'user' ? 'text-slate-600' : msg.role === 'error' ? 'text-red-500/60' : 'text-[#FF2800]/60'}`}>
                            {msg.role === 'user' ? 'YOU' : msg.role === 'error' ? 'ERROR' : 'INDRA'}
                          </span>
                          <div className={`relative p-4 md:p-5 text-sm leading-relaxed w-full transition-all
                            ${msg.role === 'user' ? 'bg-white/[0.06] border border-white/[0.08]' :
                              msg.role === 'error' ? 'bg-red-950/30 border border-red-800/30 border-l-2 border-l-red-500/60' :
                              'bg-[#0c0c0c] border border-white/[0.07] border-l-2 border-l-[#FF2800]/60'}`}>

                            {msg.role === 'error' && (
                              <div className="flex items-center gap-2 mb-2 text-red-400">
                                <AlertCircle size={13} />
                                <span className="text-[9px] font-black uppercase tracking-widest">System Error</span>
                              </div>
                            )}

                            <p className="whitespace-pre-wrap">
                              {msg.role === 'bot'
                                ? renderTextWithCitations(msg.text, msg.citations)
                                : msg.text
                              }
                            </p>

                            {msg.role === 'bot' && msg.citations && msg.citations.length > 0 && (
                              <div className="mt-3 pt-3 border-t border-white/[0.06] space-y-1.5">
                                {msg.citations.map((c, i) => (
                                  <div key={c.rule_id} className="flex items-start gap-2 text-[10px] text-slate-500">
                                    <span className="shrink-0 inline-flex items-center justify-center w-4 h-4 rounded-sm bg-[#FF2800]/20 text-[#FF2800] font-black text-[8px]">{i + 1}</span>
                                    <span><span className="text-[#FF2800]/70 font-bold">{c.rule_id}</span> — {c.content}</span>
                                  </div>
                                ))}
                              </div>
                            )}

                            {msg.role === 'bot' && (
                              <button
                                onClick={() => copyToClipboard(msg.text, msg.id)}
                                title="Copy response"
                                className="absolute top-3 right-3 p-1.5 text-slate-700 hover:text-slate-400 transition-colors"
                              >
                                {copied === msg.id
                                  ? <CheckCircle2 size={13} className="text-green-500" />
                                  : <Copy size={13} />
                                }
                              </button>
                            )}

                            {/* Timestamp */}
                            <div className="mt-2 text-[8px] text-slate-700 font-mono">
                              {new Date(msg.timestamp).toLocaleTimeString()}
                            </div>

                            {msg.model_url && msg.role === 'bot' && (
                              <div className="mt-4 pt-4 border-t border-white/[0.07]">
                                <button
                                  onClick={() => handle3DModelLoad(
                                    msg.model_url!,
                                    msg.model_metadata,
                                    msg.cad_nodes,
                                    msg.highlight_meshes ?? [],
                                    msg.context_meshes ?? [],
                                  )}
                                  className="flex items-center gap-2 px-4 py-3 md:py-2.5 text-[10px] font-black bg-[#FF2800]/10 border border-[#FF2800]/30 text-[#FF2800] uppercase w-full md:w-auto justify-center hover:bg-[#FF2800]/15 transition-colors group"
                                >
                                  <Box size={14} className="group-hover:rotate-12 transition-transform" /> Load 3D Model
                                </button>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                      {isThinking && (
                        <div className="flex items-center gap-3 text-[9px] text-[#FF2800]/60 uppercase">
                          <Loader2 size={12} className="animate-spin" />
                          <span>Scanning regulations...</span>
                          <span className="flex gap-0.5">
                            <span className="w-1 h-1 bg-[#FF2800]/40 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                            <span className="w-1 h-1 bg-[#FF2800]/40 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                            <span className="w-1 h-1 bg-[#FF2800]/40 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                          </span>
                        </div>
                      )}
                      <div ref={bottomRef} />
                    </div>
                  )}
                </div>

                <div className="shrink-0 px-4 md:px-6 py-4 pb-[env(safe-area-inset-bottom)] bg-[#060606] border-t border-white/[0.04]">
                  <div className="max-w-3xl mx-auto">
                    {/* Character count */}
                    {input.length > 800 && (
                      <div className="flex justify-end mb-1">
                        <span className={`text-[9px] font-mono ${input.length > 950 ? 'text-[#FF2800]' : 'text-slate-600'}`}>
                          {input.length}/{MAX_MESSAGE_LENGTH}
                        </span>
                      </div>
                    )}
                    <div className="relative flex flex-row items-center gap-2">
                      <input
                        ref={inputRef}
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(input); } }}
                        placeholder="Query regulations..."
                        disabled={isThinking}
                        maxLength={MAX_MESSAGE_LENGTH}
                        className="flex-1 bg-[#0c0c0c] border border-white/[0.07] focus:border-[#FF2800]/30 px-4 py-4 text-sm text-white placeholder:text-slate-600 focus:outline-none transition-colors"
                      />
                      <button
                        onClick={() => sendMessage(input)}
                        disabled={isThinking || !input.trim()}
                        className={`shrink-0 px-6 py-4 flex items-center justify-center gap-2 text-[10px] font-black uppercase transition-all ${input.trim() && !isThinking ? 'bg-[#FF2800] text-white hover:bg-[#FF4000]' : 'bg-white/5 text-slate-600 cursor-not-allowed'}`}
                      >
                        {isThinking ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
                      </button>
                    </div>
                    <p className="text-[8px] text-slate-700 mt-1.5 text-center tracking-wider">Press Enter to send · Esc to exit focus</p>
                  </div>
                </div>
              </>
            )}

            {/* ─── QUIZ MODE ───────────────────────────────────────────────── */}
            {appMode === 'quiz' && (
              <div className="flex-1 flex flex-col p-6 overflow-y-auto">
                <div className="flex-1 max-w-2xl mx-auto w-full">
                  {!quizFinished ? (
                    <>
                      {/* Progress bar */}
                      <div className="flex items-center gap-3 mb-6">
                        <div className="flex-1 h-px bg-white/[0.07]">
                          <div
                            className="h-full bg-[#FF2800] transition-all duration-500"
                            style={{ width: `${((qIndex) / QUIZ_QUESTIONS.length) * 100}%` }}
                          />
                        </div>
                        <span className="text-[#FF2800] text-xs font-black uppercase tracking-widest whitespace-nowrap">
                          {qIndex + 1} / {QUIZ_QUESTIONS.length}
                        </span>
                      </div>

                      <h2 className="text-xl font-bold text-white mb-8 leading-snug">{QUIZ_QUESTIONS[qIndex].question}</h2>

                      <div className="space-y-3">
                        {QUIZ_QUESTIONS[qIndex].options.map((opt, i) => {
                          const isCorrect = i === QUIZ_QUESTIONS[qIndex].correctAnswer;
                          const isSelected = selectedAns === i;
                          let optClass = 'border-white/10 hover:border-white/30 text-slate-300 cursor-pointer';
                          if (isAnsChecked) {
                            if (isCorrect) optClass = 'border-green-500 bg-green-500/10 text-green-400 cursor-default';
                            else if (isSelected && !isCorrect) optClass = 'border-red-500 bg-red-500/10 text-red-400 cursor-default';
                            else optClass = 'border-white/5 text-slate-600 cursor-default';
                          } else if (isSelected) {
                            optClass = 'border-[#FF2800] bg-[#FF2800]/10 text-white cursor-pointer';
                          }
                          return (
                            <button
                              key={i}
                              onClick={() => { if (!isAnsChecked) setSelectedAns(i); }}
                              disabled={isAnsChecked}
                              className={`w-full text-left p-4 border transition-all flex items-center gap-3 ${optClass}`}
                            >
                              {isAnsChecked && isCorrect && <CheckCircle2 size={15} className="shrink-0 text-green-500" />}
                              {isAnsChecked && isSelected && !isCorrect && <XCircle size={15} className="shrink-0 text-red-500" />}
                              {!(isAnsChecked && (isCorrect || (isSelected && !isCorrect))) && (
                                <span className="shrink-0 w-4 h-4 flex items-center justify-center border border-current rounded-sm text-[9px] font-black">
                                  {String.fromCharCode(65 + i)}
                                </span>
                              )}
                              <span>{opt}</span>
                            </button>
                          );
                        })}
                      </div>

                      {isAnsChecked && (
                        <div className="mt-5 p-4 bg-[#0c0c0c] border border-white/[0.08] border-l-2 border-l-[#FF2800]/60 text-sm text-slate-300 leading-relaxed">
                          <span className="text-[9px] font-black uppercase tracking-widest text-[#FF2800]/70 block mb-1">Explanation</span>
                          {QUIZ_QUESTIONS[qIndex].explanation}
                        </div>
                      )}

                      <button
                        // BUG 5 FIX: use scoreCounted flag to prevent double-increment
                        onClick={() => {
                          if (selectedAns === null) return;
                          if (!isAnsChecked) {
                            setIsAnsChecked(true);
                            if (!scoreCounted && selectedAns === QUIZ_QUESTIONS[qIndex].correctAnswer) {
                              setQuizScore(s => s + 1);
                              setScoreCounted(true);
                            }
                          } else {
                            if (qIndex + 1 < QUIZ_QUESTIONS.length) {
                              setQIndex(i => i + 1);
                              setSelectedAns(null);
                              setIsAnsChecked(false);
                              setScoreCounted(false);
                            } else {
                              setQuizFinished(true);
                            }
                          }
                        }}
                        disabled={selectedAns === null}
                        className="mt-6 w-full py-4 bg-[#FF2800] text-white font-black text-sm disabled:opacity-30 uppercase tracking-widest transition-colors hover:bg-[#FF4000] relative overflow-hidden group"
                      >
                        <span className="absolute inset-0 bg-white/10 translate-x-[-110%] group-hover:translate-x-[110%] transition-transform duration-500 skew-x-12" />
                        {!isAnsChecked ? 'CHECK ANSWER' : qIndex + 1 < QUIZ_QUESTIONS.length ? 'NEXT QUESTION →' : 'SEE RESULTS'}
                      </button>
                    </>
                  ) : (
                    <div className="text-center py-12">
                      <div className="text-6xl font-black text-[#FF2800] my-6 tabular-nums">{quizScore}/{QUIZ_QUESTIONS.length}</div>
                      <h2 className="text-3xl font-black text-white mb-2">Quiz Complete</h2>
                      <p className="text-slate-500 text-sm mb-8">
                        {quizScore === QUIZ_QUESTIONS.length ? '✓ Perfect score — full compliance.' : quizScore >= QUIZ_QUESTIONS.length / 2 ? 'Good result. Review flagged rules.' : 'Needs work. Study the rulebook carefully.'}
                      </p>
                      <div className="flex gap-3 justify-center">
                        <button
                          onClick={() => { setQIndex(0); setQuizFinished(false); setQuizScore(0); setSelectedAns(null); setIsAnsChecked(false); setScoreCounted(false); }}
                          className="px-8 py-3 bg-[#FF2800] hover:bg-[#FF4000] text-white font-bold uppercase tracking-widest transition-colors"
                        >
                          Restart Quiz
                        </button>
                        <button
                          onClick={() => setAppMode('ask')}
                          className="px-8 py-3 bg-white/10 hover:bg-white/20 text-white font-bold uppercase tracking-widest transition-colors"
                        >
                          Back to Query
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </PanelShell>

          {/* ─── 3D VIEWER PANEL ─────────────────────────────────────────────── */}
          {activeModelUrl && (
            <PanelShell id="3d" focusedPanel={focusedPanel} className={`${is3DFullscreen ? 'fixed inset-0 z-[60]' : 'flex h-[45vh] md:h-full w-full md:w-1/2'} flex-col bg-[#070707] relative`}>

              {/* Top Controls */}
              <div className="absolute top-3 left-3 right-3 z-30 flex items-center justify-between pointer-events-none">
                <div className="bg-black/80 backdrop-blur px-3 py-1.5 flex items-center gap-2 pointer-events-auto border border-[#FF2800]/20 rounded-lg">
                  <span className="w-1.5 h-1.5 bg-[#FF2800] animate-pulse rounded-full" />
                  <span className="text-[9px] font-black text-[#FF2800] uppercase tracking-wider">Live Render</span>
                  {highlightMeshes.length > 0 && (
                    <span className="text-[8px] text-slate-500 ml-1">
                      {highlightMeshes.length} parts · {contextMeshes.length} ctx
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2 pointer-events-auto">
                  <button
                    onClick={() => setShowDebug(!showDebug)}
                    className={`px-2 py-1.5 text-[9px] font-black uppercase rounded transition-colors ${showDebug ? 'bg-[#FF2800] text-white' : 'bg-black/80 text-slate-400 hover:text-white'}`}
                  >
                    Debug
                  </button>
                  <button
                    onClick={() => setShowModelInfo(v => !v)}
                    title={showModelInfo ? 'Hide info panel' : 'Show info panel'}
                    className="p-2.5 bg-black/80 hover:bg-black text-slate-400 hover:text-white rounded-lg transition-colors"
                  >
                    {showModelInfo ? <EyeOff size={14} /> : <Eye size={14} />}
                  </button>
                  <FocusButton panel="3d" focusedPanel={focusedPanel} onToggle={toggleFocus} />
                  <button onClick={() => setIs3DFullscreen(!is3DFullscreen)} className="p-2.5 bg-black/80 hover:bg-black text-white rounded-lg transition-colors">
                    <Maximize2 size={14} />
                  </button>
                  <button onClick={close3DModel} className="p-2.5 bg-[#FF2800] hover:bg-[#FF2800]/80 text-white rounded-lg transition-colors">
                    <X size={14} />
                  </button>
                </div>
              </div>

              {/* Debug Panel */}
              {showDebug && (
                <div className="absolute top-14 right-3 z-40 bg-black/95 border border-[#FF2800]/40 p-4 max-w-xs text-[9px] font-mono text-slate-300 rounded-lg shadow-2xl overflow-y-auto max-h-[40%] pointer-events-auto">
                  <h4 className="text-[#FF2800] mb-2 font-black uppercase tracking-widest border-b border-[#FF2800]/30 pb-1">CAD Inspector</h4>
                  <div className="mb-1"><strong className="text-white">Isolated:</strong> <span className="text-[#FF2800] ml-1">{isolatedParts.length > 0 ? isolatedParts.join(', ') : 'None (all highlights)'}</span></div>
                  <div className="mb-1"><strong className="text-white">Highlights:</strong> <span className="text-orange-400 ml-1">{highlightMeshes.join(', ') || 'None'}</span></div>
                  <div className="mb-2"><strong className="text-white">Context:</strong> <span className="text-slate-400 ml-1">{contextMeshes.join(', ') || 'None'}</span></div>
                  <div className="mb-1"><strong className="text-white">Progress:</strong> <span className="text-green-400 ml-1">{modelLoadProgress}%</span></div>
                  <div><strong className="text-white">GLB Materials ({modelParts.length}):</strong></div>
                  <ul className="mt-1 space-y-1 max-h-32 overflow-y-auto">
                    {modelParts.map((mat, i) => (
                      <li key={i} className={`pl-2 border-l ${matchesMeshName(mat, highlightMeshes) ? 'border-orange-500 text-orange-400' : matchesMeshName(mat, contextMeshes) ? 'border-slate-500 text-slate-400' : 'border-slate-700 text-slate-600'}`}>
                        {mat}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* 3D Canvas */}
              <div className="flex-1 w-full h-full cursor-move">
                <model-viewer
                  ref={modelViewerRef}
                  src={activeModelUrl}
                  auto-rotate={autoRotateEnabled ? "true" : "false"}
                  camera-controls="true"
                  style={{ width: '100%', height: '100%', backgroundColor: '#080808' }}
                />
              </div>

              {/* BUG 4 FIX: Progress bar driven by real model-viewer 'progress' event */}
              {isModelLoading && (
                <div className="absolute bottom-0 left-0 right-0 z-40 pointer-events-none">
                  <div className="h-[2px] bg-white/5">
                    <div
                      className="h-full bg-[#FF2800] transition-all duration-150 shadow-[0_0_8px_rgba(255,40,0,0.6)]"
                      style={{ width: `${modelLoadProgress}%` }}
                    />
                  </div>
                  <div className="px-3 py-1 bg-black/80 text-[8px] font-black uppercase tracking-widest text-[#FF2800]/70 text-right">
                    LOADING {modelLoadProgress}%
                  </div>
                </div>
              )}

              {/* Bottom Info / Part Controls */}
              {showModelInfo && activeModelMetadata && !isModelLoading && (
                <div className="absolute bottom-3 left-3 right-3 z-30 bg-black/95 backdrop-blur-md border border-white/[0.08] p-4 rounded-xl shadow-2xl max-h-[55%] overflow-y-auto">
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <h3 className="text-base font-black text-white">{activeModelMetadata.name}</h3>
                      {highlightMeshes.length > 0 && (
                        <p className="text-[9px] text-[#FF2800]/70 mt-0.5 uppercase tracking-wider">
                          {highlightMeshes.length} part{highlightMeshes.length > 1 ? 's' : ''} highlighted
                        </p>
                      )}
                    </div>
                    <button onClick={clearIsolation} className="flex items-center gap-1.5 text-xs font-bold text-slate-400 hover:text-white transition-colors">
                      <RefreshCw size={13} /> Reset
                    </button>
                  </div>

                  <div className="flex flex-wrap items-center gap-2 mb-4">
                    <button
                      onClick={() => setIsolationMode(m => m === 'ghost' ? 'hidden' : 'ghost')}
                      className={`flex items-center gap-1.5 px-3 py-1.5 text-[9px] font-black rounded-md border tracking-widest uppercase transition-all ${isolationMode === 'hidden' ? 'bg-red-500/10 border-[#FF2800] text-[#FF2800]' : 'border-white/20 text-slate-400 hover:border-white/40'}`}
                    >
                      {isolationMode === 'hidden' ? <><EyeOff size={10} /> HIDDEN</> : <><Eye size={10} /> GHOSTED</>}
                    </button>
                    <button
                      onClick={() => setHighlightSelected(h => !h)}
                      className={`flex items-center gap-1.5 px-3 py-1.5 text-[9px] font-black rounded-md border tracking-widest uppercase transition-all ${highlightSelected ? 'bg-[#FF2800]/10 border-[#FF2800] text-[#FF2800]' : 'border-white/20 text-slate-400 hover:border-white/40'}`}
                    >
                      {highlightSelected ? 'HIGHLIGHT ON' : 'HIGHLIGHT OFF'}
                    </button>
                  </div>

                  {highlightMeshes.length > 0 && (
                    <div className="mb-4">
                      <div className="flex items-center gap-2 mb-2">
                        <Layers size={13} className="text-[#FF2800]" />
                        <span className="text-[9px] font-black uppercase tracking-widest text-white">Relevant Parts</span>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        <button
                          onClick={clearIsolation}
                          className={`px-3 py-2 text-[9px] font-black uppercase tracking-wider rounded-lg transition-all border ${isolatedParts.length === 0 ? 'bg-[#FF2800] text-white border-[#FF2800] shadow-[0_0_12px_rgba(255,40,0,0.35)]' : 'bg-white/5 border-transparent hover:border-white/20 text-slate-400'}`}
                        >
                          All Parts
                        </button>
                        {highlightMeshes.map(partName => {
                          const isActive = isolatedParts.includes(partName);
                          return (
                            <button
                              key={partName}
                              onClick={() => togglePartIsolation(partName)}
                              className={`px-3 py-2 text-[9px] font-black uppercase tracking-wider rounded-lg transition-all border ${isActive
                                ? 'bg-[#FF2800]/20 border-[#FF2800] text-[#FF2800] shadow-[0_0_10px_rgba(255,40,0,0.2)]'
                                : 'bg-white/5 border-transparent hover:border-[#FF2800]/30 hover:text-[#FF2800]/70 text-slate-300'}`}
                            >
                              {cleanPartName(partName)}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {contextMeshes.length > 0 && (
                    <div className="mb-4 pt-3 border-t border-white/[0.06]">
                      <div className="flex items-center gap-2 mb-1.5">
                        <div className="w-3 h-3 rounded-sm bg-slate-500/30 border border-slate-500/50" />
                        <span className="text-[8px] font-bold uppercase tracking-widest text-slate-600">
                          Context structure ({contextMeshes.length} parts — translucent)
                        </span>
                      </div>
                    </div>
                  )}

                  <div className="flex items-center justify-between pt-3 border-t border-white/[0.06]">
                    <div className="flex gap-2">
                      <button
                        onClick={() => setAutoRotateEnabled(!autoRotateEnabled)}
                        title={autoRotateEnabled ? 'Pause rotation' : 'Start rotation'}
                        className={`p-2.5 rounded-lg transition-colors ${autoRotateEnabled ? 'bg-[#FF2800]/20 text-[#FF2800]' : 'bg-white/5 text-slate-400 hover:text-white'}`}
                      >
                        <Play size={15} />
                      </button>
                      <button onClick={reset3DCamera} title="Reset camera" className="p-2.5 bg-white/5 hover:bg-white/10 rounded-lg text-slate-400 hover:text-white transition-colors">
                        <RotateCw size={15} />
                      </button>
                    </div>
                    <div className="flex items-center gap-2">
                      <button onClick={zoomOut} title="Zoom out" className="p-2.5 bg-white/5 hover:bg-white/10 rounded-lg text-slate-400 hover:text-white transition-colors"><ZoomOut size={15} /></button>
                      <button onClick={zoomIn} title="Zoom in" className="p-2.5 bg-white/5 hover:bg-white/10 rounded-lg text-slate-400 hover:text-white transition-colors"><ZoomIn size={15} /></button>
                    </div>
                    <button
                      onClick={downloadModel}
                      className="flex items-center gap-2 px-4 py-2.5 bg-white/5 hover:bg-white/10 rounded-lg text-[9px] font-black tracking-widest text-slate-300 hover:text-white transition-colors"
                    >
                      <Download size={13} /> DOWNLOAD
                    </button>
                  </div>
                </div>
              )}
            </PanelShell>
          )}
        </div>
      </main>
    </div>
  );
}