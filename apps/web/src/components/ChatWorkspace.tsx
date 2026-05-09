import { useState, useRef, useEffect, useCallback } from "react";
import {
  Send, ShieldCheck, Trash2, Copy,Plus,
  MessageSquare, BrainCircuit, CheckCircle2, XCircle,
  ChevronRight, Menu, X, LogOut,
  Box, Maximize2, Download, ZoomIn, ZoomOut, RotateCw,
  Eye, EyeOff, Play, 
  Target, Activity, Zap, Loader2, Ruler,
  Crosshair, Minimize2, Layers, RefreshCw
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
// PATCH 1 — renderTextWithCitations
// Splits bot answer text on [CITE:rule_id] tokens injected by the backend,
// wraps each with an inline orange numbered badge, and falls back to plain
// text when no tokens are found.
// ─────────────────────────────────────────────────────────────────────────────
function renderTextWithCitations(text: string, citations?: Citation[]): React.ReactNode {
  if (!citations || citations.length === 0) return <span>{text}</span>;

  // Build a lookup: rule_id → badge index (1-based)
  const ruleIndex = new Map<string, number>();
  citations.forEach((c, i) => ruleIndex.set(c.rule_id, i + 1));

  // Split on [CITE:rule_id] tokens
  const parts = text.split(/\[CITE:([^\]]+)\]/g);

  return (
    <>
      {parts.map((part, i) => {
        if (i % 2 === 0) return <span key={i}>{part}</span>;
        // Odd indexes are the captured rule_id
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
  const toggleFocus = useCallback((panel: FocusedPanel) => setFocusedPanel(prev => prev === panel ? null : panel), []);

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
  // Multi-select: empty = show all highlights, non-empty = only those parts are orange
  const [isolatedParts, setIsolatedParts] = useState<string[]>([]);
  const [highlightSelected, setHighlightSelected] = useState(true);
  const [isolationMode, setIsolationMode] = useState<IsolationMode>('ghost');
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [modelLoadProgress, setModelLoadProgress] = useState(0);
  const [is3DFullscreen, setIs3DFullscreen] = useState(false);
  // PATCH 4 — showModelInfo now has a visible toggle button in the 3D panel header
  const [showModelInfo, setShowModelInfo] = useState(true);
  const [autoRotateEnabled, setAutoRotateEnabled] = useState(true);
  const [showDebug, setShowDebug] = useState(false);

  // Quiz State
  const [qIndex, setQIndex] = useState(0);
  const [selectedAns, setSelectedAns] = useState<number | null>(null);
  // PATCH 2 — isAnsChecked now drives visual feedback on option buttons + explanation
  const [isAnsChecked, setIsAnsChecked] = useState(false);
  const [quizScore, setQuizScore] = useState(0);
  const [quizFinished, setQuizFinished] = useState(false);

  // Refs
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const modelViewerRef = useRef<any>(null);
  const originalMaterialsRef = useRef<Map<string, number[]>>(new Map());

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
    // Multi-select: if parts are toggled, only those are orange; else all highlightMeshes.
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

    mv.addEventListener('load', handleModelReady);
    mv.addEventListener('scene-graph-ready', handleModelReady);
    if (mv.model) handleModelReady();

    return () => {
      mv.removeEventListener('load', handleModelReady);
      mv.removeEventListener('scene-graph-ready', handleModelReady);
    };
  }, [activeModelUrl, applyMeshStates]);

  useEffect(() => {
    if (activeModelUrl) {
      originalMaterialsRef.current.clear();
      setModelParts([]);
      setIsolatedParts([]);
    }
  }, [activeModelUrl]);

  // ── Auth ──
  const handleAuthSubmit = async (e: React.FormEvent) => {
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
  };

  // PATCH 3 — copyToClipboard is now wired to copy buttons on bot messages
  const copyToClipboard = useCallback(async (text: string, id: string) => {
    await navigator.clipboard.writeText(text);
    setCopied(id);
    setTimeout(() => setCopied(null), 2000);
  }, []);

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

    setActiveModelUrl(null);
    setActiveModelMetadata(null);
    setActiveCadNodes([]);
    setHighlightMeshes([]);
    setContextMeshes([]);
    setIsolatedParts([]);
    setModelParts([]);

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

      if (data.model_url) {
        handle3DModelLoad(
          data.model_url,
          data.model_metadata,
          data.cad_nodes,
          data.highlight_meshes ?? [],
          data.context_meshes ?? [],
        );
      }
    } catch (err: any) {
      setMessages(p => [...p, { id: crypto.randomUUID(), role: "error", text: err.message, timestamp: Date.now() }]);
    } finally {
      setIsThinking(false);
      inputRef.current?.focus();
    }
  }, [isThinking, lastMessageTime, session]);

  // ── 3D Model Load ──
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

    // PATCH 5 — progress bar now has a real home at the bottom of the 3D panel
    let prog = 0;
    const iv = setInterval(() => {
      prog += 15;
      setModelLoadProgress(Math.min(prog, 100));
      if (prog >= 100) { clearInterval(iv); setIsModelLoading(false); }
    }, 150);
  }, []);

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

  // PATCH 5 — download error is now surfaced via console + alert (non-intrusive)
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
    } catch (err: any) {
      console.error('Model download failed:', err);
      alert(`Download failed: ${err.message}`);
    }
  }, [activeModelUrl, activeModelMetadata]);

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

  // ─────────────────────────────────────────────────────────────────────────────
  // LOGIN SCREEN
  // ─────────────────────────────────────────────────────────────────────────────
  if (!session) {
    return (
      <div className="relative flex min-h-screen items-center justify-center bg-[#060606] overflow-hidden" style={{ fontFamily: "'Rajdhani', 'DIN Next', system-ui, sans-serif" }}>
        <div className="absolute inset-0 opacity-[0.04]" style={{ backgroundImage: 'linear-gradient(rgba(255,40,0,0.6) 1px, transparent 1px), linear-gradient(90deg, rgba(255,40,0,0.6) 1px, transparent 1px)', backgroundSize: '60px 60px' }} />
        <div className="relative z-10 w-full max-w-[420px] px-4">
          <div className="text-center mb-10">
            <h1 className="text-5xl font-black text-white tracking-tighter">INDRA</h1>
            <p className="text-[#FF2800] text-sm font-bold tracking-[0.25em] mt-1">FORMULA BHARAT 2027</p>
          </div>
          <form onSubmit={handleAuthSubmit} className="space-y-4 bg-[#0c0c0c] border border-white/[0.06] p-8">
            <input type="email" required value={authEmail} onChange={e => setAuthEmail(e.target.value)} className="w-full bg-[#111] border border-white/[0.07] px-4 py-3 text-white focus:border-[#FF2800]/40 outline-none" placeholder="Email" />
            <input type="password" required value={authPassword} onChange={e => setAuthPassword(e.target.value)} className="w-full bg-[#111] border border-white/[0.07] px-4 py-3 text-white focus:border-[#FF2800]/40 outline-none" placeholder="Password" />
            {authMessage && (
              <div className={`text-xs font-bold p-3 ${authMessage.type === 'success' ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'}`}>
                {authMessage.text}
              </div>
            )}
            <button type="submit" disabled={isAuthLoading} className="w-full py-4 bg-[#FF2800] text-white font-black text-[11px] tracking-[0.3em] uppercase hover:bg-[#FF4000] transition-colors">
              {isAuthLoading ? "AUTHENTICATING..." : isSignUp ? "CREATE ACCOUNT" : "LAUNCH INDRA"}
            </button>
            <div className="text-center">
              <button type="button" onClick={() => setIsSignUp(!isSignUp)} className="text-slate-400 text-xs hover:text-white">
                {isSignUp ? "Already have access? Sign in" : "Request new access"}
              </button>
            </div>
          </form>
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

      {isSidebarOpen && <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-[45] md:hidden" onClick={() => setIsSidebarOpen(false)} />}

      {/* ─── SIDEBAR ─────────────────────────────────────────────────────────── */}
      <PanelShell id="sidebar" focusedPanel={focusedPanel} className={`fixed inset-y-0 left-0 z-50 w-[280px] md:w-[260px] h-full md:relative md:translate-x-0 transition-transform duration-300 ease-out ${isSidebarOpen ? 'translate-x-0 shadow-2xl' : '-translate-x-full'}`}>
        <aside className="flex flex-col h-full bg-[#080808] border-r border-white/[0.05]">
          <div className="px-6 pt-6 pb-5 border-b border-white/[0.05] flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Zap size={16} className="text-[#FF2800]" />
              <div className="text-[11px] font-black text-white">INDRA OS</div>
            </div>
            <div className="flex items-center gap-2">
              <FocusButton panel="sidebar" focusedPanel={focusedPanel} onToggle={toggleFocus} />
              <button onClick={() => setIsSidebarOpen(false)} className="md:hidden p-2 text-slate-500"><X size={18} /></button>
            </div>
          </div>
          <div className="p-4">
            <button
              onClick={() => { setAppMode("ask"); setMessages([]); setActiveModelUrl(null); setHighlightMeshes([]); setContextMeshes([]); setIsolatedParts([]); setIsSidebarOpen(false); }}
              className="w-full flex items-center gap-2.5 px-4 py-3 bg-[#FF2800]/10 border border-[#FF2800]/20 text-[10px] font-bold text-[#FF2800] uppercase"
            >
              <Plus size={13} /> New Query
            </button>
          </div>
          <div className="px-4 mb-2 flex-1">
            <button onClick={() => { setAppMode("ask"); setIsSidebarOpen(false); }} className="w-full flex items-center gap-3 px-3 py-3 mb-1 text-left text-[#FF2800]">
              <MessageSquare size={13} />
              <div className="text-[10px] font-bold uppercase">Regulation Query</div>
            </button>
            <button onClick={() => { setAppMode("quiz"); setIsSidebarOpen(false); }} className="w-full flex items-center gap-3 px-3 py-3 text-left text-slate-500 hover:text-slate-300">
              <BrainCircuit size={13} />
              <div className="text-[10px] font-bold uppercase">Compliance Test</div>
            </button>
          </div>
          <div className="px-6 py-4 border-t border-white/[0.05]">
            <button onClick={() => supabase.auth.signOut()} className="flex items-center gap-2 text-[9px] text-slate-600 hover:text-slate-400 uppercase tracking-widest">
              <LogOut size={11} /> Sign Out
            </button>
          </div>
        </aside>
      </PanelShell>

      {/* ─── MAIN FRAME ──────────────────────────────────────────────────────── */}
      <main className="flex-1 flex flex-col h-full min-w-0 relative z-10">
        <header className="shrink-0 h-12 flex items-center justify-between px-4 md:px-6 border-b border-white/[0.05] bg-black/60 backdrop-blur-md relative z-30">
          <div className="flex items-center gap-3">
            <button onClick={() => setIsSidebarOpen(true)} className="md:hidden p-2 -ml-2 text-slate-500"><Menu size={20} /></button>
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
                  <FocusButton panel="chat" focusedPanel={focusedPanel} onToggle={toggleFocus} />
                </div>

                <div className="flex-1 overflow-y-auto px-4 md:px-6 py-4 scroll-smooth">
                  {messages.length === 0 && !isThinking ? (
                    <div className="h-full flex flex-col items-center justify-center max-w-xl mx-auto text-center">
                      <Target size={28} className="text-[#FF2800]/70 mb-4" />
                      <h2 className="text-xl md:text-2xl font-black tracking-tight text-white mb-8 uppercase">Ready for Input</h2>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full">
                        {QUICK_QUERIES.map((q, i) => (
                          <button key={i} onClick={() => setInput(q.label)} className="flex items-center gap-3 px-4 py-3 md:py-4 bg-[#0c0c0c] border border-white/[0.06] hover:border-[#FF2800]/30 text-left">
                            <span className="text-[#FF2800]/50 shrink-0">{q.icon}</span>
                            <span className="text-[10px] font-bold tracking-wider uppercase text-slate-400">{q.label}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="max-w-3xl mx-auto space-y-6 pb-6">
                      {messages.map(msg => (
                        <div key={msg.id} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                          <span className={`text-[8px] tracking-[0.3em] uppercase mb-1.5 font-bold ${msg.role === 'user' ? 'text-slate-600' : 'text-[#FF2800]/60'}`}>
                            {msg.role === 'user' ? 'YOU' : 'INDRA'}
                          </span>
                          <div className={`relative p-4 md:p-5 text-sm leading-relaxed w-full ${msg.role === 'user' ? 'bg-white/[0.06] border border-white/[0.08]' : 'bg-[#0c0c0c] border border-white/[0.07] border-l-2 border-l-[#FF2800]/60'}`}>

                            {/* PATCH 1 — inline citation badges rendered in text */}
                            <p className="whitespace-pre-wrap">
                              {msg.role === 'bot'
                                ? renderTextWithCitations(msg.text, msg.citations)
                                : msg.text
                              }
                            </p>

                            {/* PATCH 1 — citation legend below bot answers */}
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

                            {/* PATCH 3 — copy button on bot messages */}
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

                            {/* Load 3D model button */}
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
                                  className="flex items-center gap-2 px-4 py-3 md:py-2.5 text-[10px] font-black bg-[#FF2800]/10 border border-[#FF2800]/30 text-[#FF2800] uppercase w-full md:w-auto justify-center"
                                >
                                  <Box size={14} /> Load 3D Model
                                </button>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                      {isThinking && (
                        <div className="flex items-center gap-3 text-[9px] text-[#FF2800]/60 uppercase">
                          <Loader2 size={12} className="animate-spin" /> Scanning...
                        </div>
                      )}
                      <div ref={bottomRef} />
                    </div>
                  )}
                </div>

                <div className="shrink-0 px-4 md:px-6 py-4 pb-[env(safe-area-inset-bottom)] bg-[#060606] border-t border-white/[0.04]">
                  <div className="max-w-3xl mx-auto relative flex flex-row items-center gap-2">
                    <input
                      ref={inputRef}
                      value={input}
                      onChange={e => setInput(e.target.value)}
                      onKeyDown={e => { if (e.key === 'Enter') sendMessage(input); }}
                      placeholder="Query regulations..."
                      disabled={isThinking}
                      className="flex-1 bg-[#0c0c0c] border border-white/[0.07] focus:border-[#FF2800]/30 px-4 py-4 text-sm text-white placeholder:text-slate-600 focus:outline-none"
                    />
                    <button
                      onClick={() => sendMessage(input)}
                      disabled={isThinking || !input.trim()}
                      className={`shrink-0 px-6 py-4 flex items-center justify-center gap-2 text-[10px] font-black uppercase transition-all ${input.trim() && !isThinking ? 'bg-[#FF2800] text-white' : 'bg-white/5 text-slate-600'}`}
                    >
                      {isThinking ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
                    </button>
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
                      <div className="text-[#FF2800] text-xs font-black mb-2 uppercase tracking-widest">
                        Question {qIndex + 1} / {QUIZ_QUESTIONS.length}
                      </div>
                      <h2 className="text-xl font-bold text-white mb-8">{QUIZ_QUESTIONS[qIndex].question}</h2>

                      <div className="space-y-3">
                        {QUIZ_QUESTIONS[qIndex].options.map((opt, i) => {
                          const isCorrect = i === QUIZ_QUESTIONS[qIndex].correctAnswer;
                          const isSelected = selectedAns === i;

                          // PATCH 2 — per-option colour after CHECK ANSWER
                          let optClass = 'border-white/10 hover:border-white/30 text-slate-300';
                          if (isAnsChecked) {
                            if (isCorrect) {
                              optClass = 'border-green-500 bg-green-500/10 text-green-400';
                            } else if (isSelected && !isCorrect) {
                              optClass = 'border-red-500 bg-red-500/10 text-red-400';
                            } else {
                              optClass = 'border-white/5 text-slate-600';
                            }
                          } else if (isSelected) {
                            optClass = 'border-[#FF2800] bg-[#FF2800]/10 text-white';
                          }

                          return (
                            <button
                              key={i}
                              onClick={() => { if (!isAnsChecked) setSelectedAns(i); }}
                              disabled={isAnsChecked}
                              className={`w-full text-left p-4 border transition-all flex items-center gap-3 ${optClass}`}
                            >
                              {/* PATCH 2 — icon feedback */}
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

                      {/* PATCH 2 — explanation shown after checking */}
                      {isAnsChecked && (
                        <div className="mt-5 p-4 bg-[#0c0c0c] border border-white/[0.08] border-l-2 border-l-[#FF2800]/60 text-sm text-slate-300 leading-relaxed">
                          <span className="text-[9px] font-black uppercase tracking-widest text-[#FF2800]/70 block mb-1">Explanation</span>
                          {QUIZ_QUESTIONS[qIndex].explanation}
                        </div>
                      )}

                      <button
                        onClick={() => {
                          if (selectedAns === null) return;
                          if (!isAnsChecked) {
                            // First click: reveal feedback
                            setIsAnsChecked(true);
                            if (selectedAns === QUIZ_QUESTIONS[qIndex].correctAnswer) setQuizScore(s => s + 1);
                          } else {
                            // Second click: advance
                            if (qIndex + 1 < QUIZ_QUESTIONS.length) {
                              setQIndex(i => i + 1);
                              setSelectedAns(null);
                              setIsAnsChecked(false);
                            } else {
                              setQuizFinished(true);
                            }
                          }
                        }}
                        disabled={selectedAns === null}
                        className="mt-6 w-full py-4 bg-[#FF2800] text-white font-black text-sm disabled:opacity-30 uppercase tracking-widest transition-colors hover:bg-[#FF4000]"
                      >
                        {!isAnsChecked ? 'CHECK ANSWER' : qIndex + 1 < QUIZ_QUESTIONS.length ? 'NEXT QUESTION →' : 'SEE RESULTS'}
                      </button>
                    </>
                  ) : (
                    <div className="text-center py-12">
                      <h2 className="text-3xl font-black text-white">Quiz Complete!</h2>
                      <p className="text-6xl font-black text-[#FF2800] my-6">{quizScore}/{QUIZ_QUESTIONS.length}</p>
                      <p className="text-slate-500 text-sm mb-8">
                        {quizScore === QUIZ_QUESTIONS.length ? 'Perfect score — full compliance.' : quizScore >= QUIZ_QUESTIONS.length / 2 ? 'Good result. Review flagged rules.' : 'Needs work. Study the rulebook carefully.'}
                      </p>
                      <button
                        onClick={() => { setQIndex(0); setQuizFinished(false); setQuizScore(0); setSelectedAns(null); setIsAnsChecked(false); }}
                        className="px-8 py-3 bg-white/10 hover:bg-white/20 text-white font-bold uppercase tracking-widest transition-colors"
                      >
                        RESTART QUIZ
                      </button>
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
                      {highlightMeshes.length} parts · {contextMeshes.length} context
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2 pointer-events-auto">
                  <button onClick={() => setShowDebug(!showDebug)} className={`px-2 py-1.5 text-[9px] font-black uppercase rounded transition-colors ${showDebug ? 'bg-[#FF2800] text-white' : 'bg-black/80 text-slate-400 hover:text-white'}`}>
                    Debug
                  </button>
                  {/* PATCH 4 — toggle to show/hide the bottom info panel */}
                  <button
                    onClick={() => setShowModelInfo(v => !v)}
                    title={showModelInfo ? 'Hide panel info' : 'Show panel info'}
                    className="p-2.5 bg-black/80 hover:bg-black text-slate-400 hover:text-white rounded-lg transition-colors"
                  >
                    {showModelInfo ? <EyeOff size={14} /> : <Eye size={14} />}
                  </button>
                  <FocusButton panel="3d" focusedPanel={focusedPanel} onToggle={toggleFocus} />
                  <button onClick={() => setIs3DFullscreen(!is3DFullscreen)} className="p-2.5 bg-black/80 hover:bg-black text-white rounded-lg transition-colors"><Maximize2 size={14} /></button>
                  <button onClick={close3DModel} className="p-2.5 bg-[#FF2800] hover:bg-[#FF2800]/80 text-white rounded-lg transition-colors"><X size={14} /></button>
                </div>
              </div>

              {/* Debug Panel */}
              {showDebug && (
                <div className="absolute top-14 right-3 z-40 bg-black/95 border border-[#FF2800]/40 p-4 max-w-xs text-[9px] font-mono text-slate-300 rounded-lg shadow-2xl overflow-y-auto max-h-[40%] pointer-events-auto">
                  <h4 className="text-[#FF2800] mb-2 font-black uppercase tracking-widest border-b border-[#FF2800]/30 pb-1">CAD Node Inspector</h4>
                  <div className="mb-1"><strong className="text-white">Isolated:</strong> <span className="text-[#FF2800] ml-1">{isolatedParts.length > 0 ? isolatedParts.join(', ') : 'None (all highlights)'}</span></div>
                  <div className="mb-1"><strong className="text-white">Highlights:</strong> <span className="text-orange-400 ml-1">{highlightMeshes.join(', ') || 'None'}</span></div>
                  <div className="mb-2"><strong className="text-white">Context:</strong> <span className="text-slate-400 ml-1">{contextMeshes.join(', ') || 'None'}</span></div>
                  <div><strong className="text-white">GLB Materials:</strong></div>
                  <ul className="mt-1 space-y-1">
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

              {/* PATCH 5 — progress bar pinned to the bottom of the 3D canvas */}
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
                          {highlightMeshes.length} relevant part{highlightMeshes.length > 1 ? 's' : ''} highlighted
                        </p>
                      )}
                    </div>
                    <button onClick={clearIsolation} className="flex items-center gap-1.5 text-xs font-bold text-slate-400 hover:text-white">
                      <RefreshCw size={13} /> Reset
                    </button>
                  </div>

                  <div className="flex flex-wrap items-center gap-2 mb-4">
                    <button
                      onClick={() => setIsolationMode(m => m === 'ghost' ? 'hidden' : 'ghost')}
                      className={`flex items-center gap-1.5 px-3 py-1.5 text-[9px] font-black rounded-md border tracking-widest uppercase transition-all ${isolationMode === 'hidden' ? 'bg-red-500/10 border-[#FF2800] text-[#FF2800]' : 'border-white/20 text-slate-400'}`}
                    >
                      {isolationMode === 'hidden' ? <><EyeOff size={10} /> HIDDEN</> : <><Eye size={10} /> GHOSTED</>}
                    </button>
                    <button
                      onClick={() => setHighlightSelected(h => !h)}
                      className={`flex items-center gap-1.5 px-3 py-1.5 text-[9px] font-black rounded-md border tracking-widest uppercase transition-all ${highlightSelected ? 'bg-[#FF2800]/10 border-[#FF2800] text-[#FF2800]' : 'border-white/20 text-slate-400'}`}
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
                      <button onClick={() => setAutoRotateEnabled(!autoRotateEnabled)} className={`p-2.5 rounded-lg transition-colors ${autoRotateEnabled ? 'bg-[#FF2800]/20 text-[#FF2800]' : 'bg-white/5 text-slate-400 hover:text-white'}`}><Play size={15} /></button>
                      <button onClick={reset3DCamera} className="p-2.5 bg-white/5 hover:bg-white/10 rounded-lg text-slate-400 hover:text-white transition-colors"><RotateCw size={15} /></button>
                    </div>
                    <div className="flex items-center gap-2">
                      <button onClick={zoomOut} className="p-2.5 bg-white/5 hover:bg-white/10 rounded-lg text-slate-400 hover:text-white transition-colors"><ZoomOut size={15} /></button>
                      <button onClick={zoomIn} className="p-2.5 bg-white/5 hover:bg-white/10 rounded-lg text-slate-400 hover:text-white transition-colors"><ZoomIn size={15} /></button>
                    </div>
                    <button onClick={downloadModel} className="flex items-center gap-2 px-4 py-2.5 bg-white/5 hover:bg-white/10 rounded-lg text-[9px] font-black tracking-widest text-slate-300 hover:text-white transition-colors">
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