import { useState, useRef, useEffect, useCallback } from "react";
import {
  ArrowUp, Copy, Check, Box, Maximize2, Minimize2, ZoomIn, ZoomOut, RotateCw,
  X, Plus, AlertCircle, Loader2, ChevronDown,
} from "lucide-react";

// ─────────────────────────────────────────────────────────────────────────────
// TYPES & CONFIG (backend contract unchanged)
// ─────────────────────────────────────────────────────────────────────────────
declare global {
  namespace JSX {
    interface IntrinsicElements {
      "model-viewer": any;
    }
  }
}

const getEnvVar = (key: string, fallback = "") => (import.meta as any).env?.[key] || fallback;
const API_URL = getEnvVar("VITE_API_URL", "http://localhost:8000");
const MAX_MESSAGE_LENGTH = 1000;

interface Citation { rule_id: string; content: string; }
interface CadNode { rule_id: string; cad_node_name: string; relevance_score?: number; }
interface ModelMetadata { name: string; category: string; tags: string[]; description?: string; fileSize?: string; cad_nodes?: CadNode[]; }
interface Message {
  id: string; role: "user" | "bot" | "error"; text: string; citations?: Citation[];
  model_url?: string; model_metadata?: ModelMetadata; cad_nodes?: CadNode[];
  highlight_meshes?: string[]; context_meshes?: string[]; timestamp: number;
  retryText?: string;
}
type IsolationMode = "ghost" | "hidden";

// Ambient opening animation: scattered questions a student asks while reading a rulebook.
// x/y are % of the screen, r is tilt in degrees, o is resting opacity, m = also show on mobile.
type Floater = { t: string; x: number; y: number; r: number; s: number; o: number; kind: "key" | "code" | "q" | "mark"; m?: boolean };
const FLOATERS: Floater[] = [
  { t: "why?", x: 7, y: 16, r: -8, s: 46, o: 0.9, kind: "key", m: true },
  { t: "when?", x: 62, y: 11, r: 5, s: 42, o: 0.9, kind: "key", m: true },
  { t: "what()?", x: 3, y: 45, r: -3, s: 36, o: 0.85, kind: "code" },
  { t: "what?()", x: 74, y: 43, r: 4, s: 36, o: 0.85, kind: "code" },
  { t: "when?", x: 36, y: 74, r: -5, s: 40, o: 0.9, kind: "key", m: true },
  { t: "Why is the wall 1.5 mm?", x: 22, y: 29, r: -6, s: 15, o: 0.9, kind: "q", m: true },
  { t: "Which rule applies here?", x: 66, y: 28, r: 4, s: 15, o: 0.9, kind: "q" },
  { t: "Is this even legal?", x: 10, y: 62, r: 7, s: 15, o: 0.9, kind: "q", m: true },
  { t: "Where do I measure from?", x: 62, y: 63, r: -5, s: 15, o: 0.9, kind: "q", m: true },
  { t: "Does this need a drawing?", x: 44, y: 22, r: 3, s: 14, o: 0.8, kind: "q" },
  { t: "Who checks this at tech inspection?", x: 18, y: 82, r: -3, s: 14, o: 0.8, kind: "q" },
  { t: "Minimum or maximum?", x: 70, y: 79, r: 5, s: 14, o: 0.8, kind: "q" },
  { t: "?", x: 38, y: 14, r: 12, s: 22, o: 0.7, kind: "mark" },
  { t: "( )", x: 82, y: 71, r: -10, s: 26, o: 0.7, kind: "mark" },
  { t: "?!", x: 52, y: 56, r: -14, s: 22, o: 0.6, kind: "mark" },
  { t: "( \\", x: 6, y: 78, r: 8, s: 26, o: 0.7, kind: "mark", m: true },
];

// Deterministic per-item motion. Negative delays start each word mid-cycle, so the screen is never empty on load.
const motion = (i: number) => ({
  dur: 9 + ((i * 37) % 7),
  delay: -0.8 * i,
  dx: ((i * 53) % 40) - 20,
  dy: ((i * 29) % 40) - 20,
  dr: ((i * 11) % 15) - 7,
});
// Design tokens — one place to retune the whole look
const T = {
  bg: "#FBFBFD", ink: "#1D1D1F", mute: "#6E6E73", line: "#E8E8ED",
  fill: "#F2F2F7", accent: "#0A64D8",
};
// model-viewer expects linear 0-1 RGBA
const HIGHLIGHT: [number, number, number, number] = [0.04, 0.39, 0.85, 1.0];

export default function IndraWorkspace() {
  // ── Chat state ──
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isThinking, setIsThinking] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const [openCite, setOpenCite] = useState<{ msg: string; rule: string } | null>(null);

  // ── 3D state ──
  const [activeModelUrl, setActiveModelUrl] = useState<string | null>(null);
  const [activeModelMetadata, setActiveModelMetadata] = useState<ModelMetadata | null>(null);
  const [highlightMeshes, setHighlightMeshes] = useState<string[]>([]);
  const [contextMeshes, setContextMeshes] = useState<string[]>([]);
  const [isolatedParts, setIsolatedParts] = useState<string[]>([]);
  const [isolationMode, setIsolationMode] = useState<IsolationMode>("ghost");
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [is3DFullscreen, setIs3DFullscreen] = useState(false);
  const [showPanel, setShowPanel] = useState(true);

  // ── Refs ──
  const bottomRef = useRef<HTMLDivElement>(null);
  const modelViewerRef = useRef<any>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const originalMaterialsRef = useRef<Map<string, number[]>>(new Map());

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [messages, isThinking]);

  // Auto-grow composer
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  }, [input]);

  // Esc closes the viewer (fullscreen first, then the viewer itself)
  useEffect(() => {
    if (!activeModelUrl) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (is3DFullscreen) setIs3DFullscreen(false);
      else setActiveModelUrl(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [activeModelUrl, is3DFullscreen]);

  // ─────────────────────────────────────────────────────────────────────────
  // 3D MATERIAL LOGIC (same behaviour as before)
  // ─────────────────────────────────────────────────────────────────────────
  const matchesMeshName = (matName: string, targets: string[]) => {
    if (targets.length === 0) return false;
    const clean = (s: string) => s.toLowerCase().replace(/[_\-\s]/g, "");
    const cleanMat = clean(matName);
    return targets.some(t => cleanMat.includes(clean(t)) || clean(t).includes(cleanMat));
  };

  const applyMeshStates = useCallback(() => {
    const mv = modelViewerRef.current;
    if (!mv?.model?.materials) return;
    const activeTargets = isolatedParts.length > 0 ? isolatedParts : highlightMeshes;
    const siblingTargets = isolatedParts.length > 0 ? highlightMeshes.filter(m => !isolatedParts.includes(m)) : [];

    mv.model.materials.forEach((mat: any, i: number) => {
      const matName = mat.name || `Unnamed_Material_${i}`;
      mat.name = matName;
      if (matchesMeshName(matName, activeTargets)) {
        mat.pbrMetallicRoughness.setBaseColorFactor(HIGHLIGHT);
        mat.setAlphaMode("OPAQUE");
      } else if (matchesMeshName(matName, siblingTargets)) {
        mat.pbrMetallicRoughness.setBaseColorFactor([HIGHLIGHT[0], HIGHLIGHT[1], HIGHLIGHT[2], 0.4]);
        mat.setAlphaMode("BLEND");
      } else if (matchesMeshName(matName, contextMeshes)) {
        mat.pbrMetallicRoughness.setBaseColorFactor([0.5, 0.5, 0.5, 0.15]);
        mat.setAlphaMode("BLEND");
      } else {
        mat.pbrMetallicRoughness.setBaseColorFactor(isolationMode === "hidden" ? [0, 0, 0, 0] : [0.2, 0.2, 0.2, 0.05]);
        mat.setAlphaMode("BLEND");
      }
    });
  }, [isolatedParts, highlightMeshes, contextMeshes, isolationMode]);

  useEffect(() => { if (activeModelUrl) applyMeshStates(); }, [applyMeshStates, activeModelUrl]);

  useEffect(() => {
    const mv = modelViewerRef.current;
    if (!mv) return;
    const handleModelReady = () => {
      mv.model?.materials?.forEach((mat: any, i: number) => {
        const matName = mat.name || `Unnamed_Material_${i}`;
        mat.name = matName;
        if (!originalMaterialsRef.current.has(matName)) {
          originalMaterialsRef.current.set(matName, [...(mat.pbrMetallicRoughness?.baseColorFactor ?? [0.8, 0.8, 0.8, 1])]);
        }
      });
      setIsModelLoading(false);
      applyMeshStates();
    };
    const handleError = () => setIsModelLoading(false);
    mv.addEventListener("load", handleModelReady);
    mv.addEventListener("error", handleError);
    return () => {
      mv.removeEventListener("load", handleModelReady);
      mv.removeEventListener("error", handleError);
    };
  }, [activeModelUrl, applyMeshStates]);

  // ── Actions ──
  const handle3DModelLoad = (url: string, meta?: ModelMetadata, highlights: string[] = [], contexts: string[] = []) => {
    originalMaterialsRef.current.clear();
    setActiveModelUrl(url);
    setActiveModelMetadata(meta ?? null);
    setHighlightMeshes(highlights);
    setContextMeshes(contexts);
    setIsolatedParts([]);
    setIsModelLoading(true);
    setShowPanel(true);
  };

  const sendMessage = async (text: string, isRetry = false) => {
    const sanitized = text.trim().slice(0, MAX_MESSAGE_LENGTH);
    if (!sanitized || isThinking) return;
    if (isRetry) {
      setMessages(p => p.filter(m => m.role !== "error"));
    } else {
      setMessages(p => [...p, { id: crypto.randomUUID(), role: "user", text: sanitized, timestamp: Date.now() }]);
    }
    setInput("");
    setOpenCite(null);
    setIsThinking(true);

    try {
      const res = await fetch(`${API_URL}/ask_indra`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${getEnvVar("VITE_API_AUTH_TOKEN")}` },
        body: JSON.stringify({ message: sanitized, domain: "Formula Bharat 2027 Full" }),
      });
      if (!res.ok) throw new Error("Couldn't reach the rulebook service. Check your connection and try again.");
      const data = await res.json();

      setMessages(p => [...p, {
        id: crypto.randomUUID(), role: "bot", text: data.answer, citations: data.citations,
        model_url: data.model_url, model_metadata: data.model_metadata,
        highlight_meshes: data.highlight_meshes ?? [], context_meshes: data.context_meshes ?? [], timestamp: Date.now(),
      }]);

      if (data.model_url) handle3DModelLoad(data.model_url, data.model_metadata, data.highlight_meshes, data.context_meshes);
    } catch (err: any) {
      setMessages(p => [...p, { id: crypto.randomUUID(), role: "error", text: err.message, retryText: sanitized, timestamp: Date.now() }]);
    } finally {
      setIsThinking(false);
    }
  };

  const copyText = async (id: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text.replace(/\[CITE:[^\]]+\]/g, ""));
      setCopied(id);
      setTimeout(() => setCopied(null), 1500);
    } catch { /* clipboard unavailable */ }
  };

  const newChat = () => {
    setMessages([]); setActiveModelUrl(null); setIs3DFullscreen(false); setOpenCite(null);
    textareaRef.current?.focus();
  };

  // Citations: tap/click to reveal the rule text (works on touch, unlike title tooltips)
  const renderTextWithCitations = (msg: Message) => {
    const { text, citations } = msg;
    if (!citations?.length) return <p className="whitespace-pre-wrap">{text}</p>;
    const parts = text.split(/\[CITE:([^\]]+)\]/g);
    const active = openCite?.msg === msg.id ? citations.find(c => c.rule_id === openCite.rule) : undefined;
    return (
      <>
        <p className="whitespace-pre-wrap">
          {parts.map((part, i) => {
            if (i % 2 === 0) return <span key={i}>{part}</span>;
            const isOpen = openCite?.msg === msg.id && openCite.rule === part;
            return (
              <button
                key={i}
                onClick={() => setOpenCite(isOpen ? null : { msg: msg.id, rule: part })}
                aria-expanded={isOpen}
                aria-label={`Rule ${part}`}
                className="indra-focus mx-0.5 px-1.5 rounded-md text-[12px] font-medium align-baseline transition-colors"
                style={{ background: isOpen ? T.accent : "#E8F0FC", color: isOpen ? "#fff" : T.accent }}
              >
                {part}
              </button>
            );
          })}
        </p>
        {active && (
          <div className="indra-pop mt-3 rounded-xl px-4 py-3 text-[14px] leading-relaxed" style={{ background: T.fill, color: T.ink }}>
            <div className="font-medium mb-1" style={{ color: T.mute }}>Rule {active.rule_id}</div>
            {active.content}
          </div>
        )}
      </>
    );
  };

  const hasViewer = !!activeModelUrl;
  const splitView = hasViewer && !is3DFullscreen;
  const partLabel = (p: string) => p.replace(/_mat$/gi, "").replace(/_/g, " ").trim();

  // ─────────────────────────────────────────────────────────────────────────
  // RENDER
  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div
      className="flex h-[100dvh] w-full"
      style={{ background: T.bg, color: T.ink, fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Inter', system-ui, sans-serif", letterSpacing: "-0.011em" }}
    >
      <style>{`
        @keyframes indraRise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
        @keyframes indraPop { from { opacity: 0; transform: scale(.98); } to { opacity: 1; transform: none; } }
        @keyframes indraDot { 0%,80%,100% { opacity: .25; } 40% { opacity: 1; } }
        @keyframes indraFloat {
          0%   { opacity: 0; transform: translate(calc(var(--dx) * -2px), calc(var(--dy) * -2px)) rotate(calc(var(--r) - var(--dr))) scale(.7); }
          7%   { opacity: var(--o); transform: translate(calc(var(--dx) * -1.9px), calc(var(--dy) * -1.9px)) rotate(var(--r)) scale(1.1); }
          13%  { opacity: var(--o); transform: translate(calc(var(--dx) * -1.7px), calc(var(--dy) * -1.7px)) rotate(var(--r)) scale(1); }
          86%  { opacity: var(--o); transform: translate(calc(var(--dx) * 1.5px), calc(var(--dy) * 1.5px)) rotate(calc(var(--r) + var(--dr))) scale(1); }
          100% { opacity: 0; transform: translate(calc(var(--dx) * 2px), calc(var(--dy) * 2px)) rotate(calc(var(--r) + var(--dr))) scale(1.03); }
        }
        .indra-float { position: absolute; white-space: nowrap; pointer-events: none; user-select: none; opacity: 0;
          animation: indraFloat var(--t) linear var(--d) infinite both; }
        .indra-rise { animation: indraRise .28s ease-out both; }
        .indra-pop { animation: indraPop .18s ease-out both; }
        .indra-focus:focus-visible, .indra-btn:focus-visible { outline: 2px solid ${T.accent}; outline-offset: 2px; }
        .indra-glass { background: rgba(255,255,255,.72); backdrop-filter: saturate(180%) blur(24px); -webkit-backdrop-filter: saturate(180%) blur(24px); }
        @media (prefers-reduced-motion: reduce) { .indra-rise, .indra-pop { animation: none; } .indra-float { animation: none; opacity: var(--o); transform: rotate(var(--r)); } .indra-dot { animation: none !important; } }
      `}</style>

      {/* ── CHAT ── */}
      <main className={`relative flex flex-col h-full min-w-0 transition-[width] duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] ${splitView ? "w-full md:w-1/2" : "w-full"}`}
        style={splitView ? { borderRight: `1px solid ${T.line}` } : undefined}>

        {messages.length === 0 && (
          <div aria-hidden className="absolute inset-0 overflow-hidden z-0">
            {FLOATERS.map((f, i) => {
              const m = motion(i);
              const look =
                f.kind === "key" ? { color: T.ink, fontWeight: 600, letterSpacing: "-0.03em" }
                : f.kind === "code" ? { color: T.accent, fontWeight: 600, fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", letterSpacing: "-0.04em" }
                : f.kind === "q" ? { color: T.mute, fontWeight: 500 }
                : { color: "#A1A1A6", fontWeight: 500 };
              return (
                <span
                  key={i}
                  className={`indra-float ${f.m ? "" : "hidden sm:block"}`}
                  style={{
                    left: `${f.x}%`, top: `${f.y}%`, fontSize: f.s,
                    ["--r" as any]: `${f.r}deg`, ["--o" as any]: f.o, ["--d" as any]: `${m.delay}s`, ["--t" as any]: `${m.dur}s`,
                    ["--dx" as any]: m.dx, ["--dy" as any]: m.dy, ["--dr" as any]: `${m.dr}deg`,
                    ...look,
                  }}>
                  {f.t}
                </span>
              );
            })}
          </div>
        )}

        <header className="shrink-0 h-14 flex items-center justify-between px-5 indra-glass sticky top-0 z-10" style={{ borderBottom: `1px solid ${T.line}` }}>
          <h1 className="text-[17px] font-semibold tracking-[0.06em]">INDRA</h1>
          {messages.length > 0 && (
            <button onClick={newChat} className="indra-btn flex items-center gap-1.5 h-9 px-3 rounded-full text-[14px] font-medium transition-colors hover:bg-black/5" style={{ color: T.accent }}>
              <Plus size={16} /> New chat
            </button>
          )}
        </header>

        <div className="relative z-[1] flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-[720px] px-5 py-8 min-h-full flex flex-col">
            {messages.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center text-center py-10 indra-rise">
                <h2 className="text-[64px] md:text-[104px] leading-none font-semibold tracking-[-0.02em]">
                  INDRA
                </h2>
                <p className="mt-5 text-[17px] md:text-[19px] leading-relaxed max-w-[420px]" style={{ color: T.mute }}>
                  Every rulebook question, answered with the rule and the part.
                </p>
              </div>
            ) : (
              <div className="space-y-8">
                {messages.map(msg => (
                  <div key={msg.id} className={`indra-rise flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                    {msg.role === "user" && (
                      <div className="max-w-[85%] px-4 py-2.5 rounded-[22px] text-[16px] leading-[1.45] whitespace-pre-wrap" style={{ background: T.fill }}>
                        {msg.text}
                      </div>
                    )}

                    {msg.role === "error" && (
                      <div className="flex items-start gap-3 max-w-[85%] text-[15px] leading-relaxed">
                        <AlertCircle size={18} className="mt-0.5 shrink-0" style={{ color: "#D70015" }} />
                        <div>
                          <p>{msg.text}</p>
                          {msg.retryText && (
                            <button onClick={() => sendMessage(msg.retryText!, true)} className="indra-btn mt-1 font-medium rounded" style={{ color: T.accent }}>
                              Try again
                            </button>
                          )}
                        </div>
                      </div>
                    )}

                    {msg.role === "bot" && (
                      <div className="w-full text-[16px] leading-[1.6]">
                        {renderTextWithCitations(msg)}

                        {msg.model_url && (
                          <button
                            onClick={() => handle3DModelLoad(msg.model_url!, msg.model_metadata, msg.highlight_meshes, msg.context_meshes)}
                            className="indra-btn mt-4 w-full sm:w-auto flex items-center gap-3 pl-3 pr-5 py-2.5 rounded-2xl text-left transition-colors hover:bg-[#E8E8ED]"
                            style={{ background: T.fill }}>
                            <span className="w-9 h-9 rounded-xl flex items-center justify-center bg-white" style={{ color: T.accent }}>
                              <Box size={18} />
                            </span>
                            <span>
                              <span className="block text-[15px] font-medium leading-tight">{msg.model_metadata?.name ?? "3D model"}</span>
                              <span className="block text-[13px]" style={{ color: T.mute }}>View in 3D</span>
                            </span>
                          </button>
                        )}

                        <div className="mt-3 -ml-2">
                          <button onClick={() => copyText(msg.id, msg.text)} aria-label="Copy answer"
                            className="indra-btn p-2 rounded-lg transition-colors hover:bg-black/5" style={{ color: T.mute }}>
                            {copied === msg.id ? <Check size={16} /> : <Copy size={16} />}
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                ))}

                {isThinking && (
                  <div className="flex items-center gap-1.5 h-6" role="status" aria-label="Searching the rulebook">
                    {[0, 1, 2].map(i => (
                      <span key={i} className="indra-dot w-2 h-2 rounded-full" style={{ background: T.mute, animation: `indraDot 1.2s ${i * 0.15}s infinite` }} />
                    ))}
                    <span className="ml-2 text-[14px]" style={{ color: T.mute }}>Searching the rulebook</span>
                  </div>
                )}
                <div ref={bottomRef} />
              </div>
            )}
          </div>
        </div>

        {/* Composer */}
        <div className="relative z-[1] shrink-0 px-5 pb-5 pt-2" style={{ paddingBottom: "max(1.25rem, env(safe-area-inset-bottom))" }}>
          <div className="mx-auto max-w-[720px]">
            <div className="flex items-end gap-2 pl-5 pr-2 py-2 rounded-[26px] bg-[#F2F2F7] transition-shadow focus-within:shadow-[0_0_0_3px_rgba(10,100,216,0.18)]"
              style={{ border: `1px solid ${T.line}` }}>
              <textarea
                ref={textareaRef}
                rows={1}
                value={input}
                onChange={e => setInput(e.target.value.slice(0, MAX_MESSAGE_LENGTH))}
                onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); sendMessage(input); } }}
                placeholder="Ask about a rule"
                aria-label="Message"
                className="flex-1 resize-none bg-transparent py-2 text-[16px] leading-6 focus:outline-none placeholder:text-[#6E6E73]"
              />
              <button
                onClick={() => sendMessage(input)}
                disabled={isThinking || !input.trim()}
                aria-label="Send"
                className="indra-btn shrink-0 w-9 h-9 rounded-full flex items-center justify-center text-white transition-all disabled:cursor-not-allowed"
                style={{ background: input.trim() && !isThinking ? T.accent : "#D1D1D6" }}>
                {isThinking ? <Loader2 size={16} className="animate-spin" /> : <ArrowUp size={18} strokeWidth={2.5} />}
              </button>
            </div>
            {input.length > MAX_MESSAGE_LENGTH * 0.8 && (
              <p className="mt-1.5 text-right text-[12px]" style={{ color: T.mute }}>{input.length} / {MAX_MESSAGE_LENGTH}</p>
            )}
          </div>
        </div>
      </main>

      {/* ── 3D VIEWER ──
          Desktop: right half. Mobile: full-screen sheet (previously hidden, so the button did nothing). */}
      {hasViewer && (
        <section
          aria-label="3D viewer"
          className={`${is3DFullscreen ? "fixed inset-0 z-50" : "fixed inset-0 z-40 md:static md:z-auto md:w-1/2"} flex flex-col relative overflow-hidden indra-pop`}
          style={{ background: "linear-gradient(180deg,#F5F5F7 0%,#E9E9EE 100%)" }}>

          <div className="absolute top-0 inset-x-0 z-10 flex items-center justify-between p-4" style={{ paddingTop: "max(1rem, env(safe-area-inset-top))" }}>
            <button onClick={() => setActiveModelUrl(null)} aria-label="Close 3D viewer"
              className="indra-btn indra-glass w-10 h-10 rounded-full flex items-center justify-center shadow-sm transition-colors hover:bg-white">
              <X size={18} />
            </button>
            <button onClick={() => setIs3DFullscreen(f => !f)} aria-label={is3DFullscreen ? "Exit full screen" : "Full screen"}
              className="indra-btn indra-glass hidden md:flex w-10 h-10 rounded-full items-center justify-center shadow-sm transition-colors hover:bg-white">
              {is3DFullscreen ? <Minimize2 size={17} /> : <Maximize2 size={17} />}
            </button>
          </div>

          <div className="flex-1 w-full h-full">
            <model-viewer
              ref={modelViewerRef}
              src={activeModelUrl}
              alt={activeModelMetadata?.name ?? "3D model"}
              auto-rotate="true"
              auto-rotate-delay="3000"
              camera-controls="true"
              touch-action="pan-y"
              shadow-intensity="1"
              style={{ width: "100%", height: "100%", backgroundColor: "transparent", outline: "none" }}
            />
          </div>

          {isModelLoading && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none" role="status">
              <div className="indra-glass flex items-center gap-2 px-4 py-2 rounded-full text-[14px] shadow-sm">
                <Loader2 size={16} className="animate-spin" /> Loading model
              </div>
            </div>
          )}

          {activeModelMetadata && !isModelLoading && (
            <div className="absolute inset-x-0 bottom-0 z-10 p-4" style={{ paddingBottom: "max(1rem, env(safe-area-inset-bottom))" }}>
              <div className="indra-glass mx-auto max-w-[560px] rounded-[26px] shadow-[0_8px_32px_rgba(0,0,0,0.10)]" style={{ border: "1px solid rgba(255,255,255,.7)" }}>
                <button onClick={() => setShowPanel(s => !s)} aria-expanded={showPanel}
                  className="indra-btn w-full flex items-center justify-between px-5 py-3.5 rounded-[26px] text-left">
                  <span className="text-[16px] font-semibold tracking-tight truncate">{activeModelMetadata.name}</span>
                  <ChevronDown size={18} className={`shrink-0 transition-transform ${showPanel ? "" : "-rotate-180"}`} style={{ color: T.mute }} />
                </button>

                {showPanel && (
                  <div className="px-5 pb-4">
                    {highlightMeshes.length > 0 && (
                      <div className="flex flex-wrap gap-2 max-h-28 overflow-y-auto">
                        <button onClick={() => setIsolatedParts([])} aria-pressed={isolatedParts.length === 0}
                          className="indra-btn px-3.5 h-8 rounded-full text-[13px] font-medium transition-colors"
                          style={isolatedParts.length === 0 ? { background: T.ink, color: "#fff" } : { background: "rgba(0,0,0,.06)" }}>
                          All parts
                        </button>
                        {highlightMeshes.map(part => {
                          const on = isolatedParts.includes(part);
                          return (
                            <button key={part} aria-pressed={on}
                              onClick={() => setIsolatedParts(p => p.includes(part) ? p.filter(x => x !== part) : [...p, part])}
                              className="indra-btn px-3.5 h-8 rounded-full text-[13px] font-medium transition-colors"
                              style={on ? { background: T.accent, color: "#fff" } : { background: "rgba(0,0,0,.06)" }}>
                              {partLabel(part)}
                            </button>
                          );
                        })}
                      </div>
                    )}

                    <div className="mt-4 flex items-center justify-between gap-3">
                      {/* Segmented control: how to treat everything that isn't selected */}
                      <div role="group" aria-label="Surrounding parts" className="flex p-0.5 rounded-full" style={{ background: "rgba(0,0,0,.06)" }}>
                        {(["ghost", "hidden"] as IsolationMode[]).map(m => (
                          <button key={m} onClick={() => setIsolationMode(m)} aria-pressed={isolationMode === m}
                            className="indra-btn px-3 h-7 rounded-full text-[13px] font-medium transition-all"
                            style={isolationMode === m ? { background: "#fff", boxShadow: "0 1px 3px rgba(0,0,0,.12)" } : { color: T.mute }}>
                            {m === "ghost" ? "Faded" : "Hidden"}
                          </button>
                        ))}
                      </div>
                      <div className="flex gap-1" style={{ color: T.mute }}>
                        <button aria-label="Zoom out" onClick={() => modelViewerRef.current?.zoom(-1)} className="indra-btn p-2 rounded-full transition-colors hover:bg-black/5"><ZoomOut size={17} /></button>
                        <button aria-label="Zoom in" onClick={() => modelViewerRef.current?.zoom(1)} className="indra-btn p-2 rounded-full transition-colors hover:bg-black/5"><ZoomIn size={17} /></button>
                        <button aria-label="Reset view" onClick={() => modelViewerRef.current?.resetTurntableRotation()} className="indra-btn p-2 rounded-full transition-colors hover:bg-black/5"><RotateCw size={17} /></button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
