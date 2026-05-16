"use client";

import { useState, useRef, useEffect } from "react";

// --- Types ---
type StatusEvent = {
  step: string;
  message: string;
  progress: number;
};

type ResultData = {
  filename: string;
  caption: string;
  translation: string;
  insight: string;
};

// --- Pipeline step config ---
const PIPELINE_STEPS = [
  { id: "upload", label: "Image Input", icon: "📷" },
  { id: "vit", label: "ViT Encoder", icon: "🧠" },
  { id: "gpt2", label: "GPT-2 Decoder", icon: "✍️" },
  { id: "translate", label: "বাংলা Translation", icon: "🌐" },
  { id: "done", label: "Complete", icon: "✅" },
];

const API_BASE = "https://bangla-image-captioning.onrender.com";
// For local dev, uncomment the line below:
// const API_BASE = "http://localhost:8000";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<ResultData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Pipeline status
  const [logs, setLogs] = useState<StatusEvent[]>([]);
  const [currentStep, setCurrentStep] = useState<string | null>(null);
  const [completedSteps, setCompletedSteps] = useState<string[]>([]);
  const [progress, setProgress] = useState(0);

  const logEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll the log
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // Generate preview when file selected
  useEffect(() => {
    if (!selectedFile) {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(selectedFile);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [selectedFile]);

  function resetState() {
    setResult(null);
    setError(null);
    setLogs([]);
    setCurrentStep(null);
    setCompletedSteps([]);
    setProgress(0);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!selectedFile) return;

    resetState();
    setLoading(true);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      // Use fetch with streaming to read SSE from POST
      const response = await fetch(`${API_BASE}/analyze-image-stream`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) throw new Error("No response stream");

      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Parse SSE events from buffer
        const events = buffer.split("\n\n");
        buffer = events.pop() || ""; // Keep incomplete event in buffer

        for (const eventBlock of events) {
          const lines = eventBlock.trim().split("\n");
          let eventType = "";
          let eventData = "";

          for (const line of lines) {
            if (line.startsWith("event: ")) {
              eventType = line.slice(7);
            } else if (line.startsWith("data: ")) {
              eventData = line.slice(6);
            }
          }

          if (!eventType || !eventData) continue;

          try {
            const parsed = JSON.parse(eventData);

            if (eventType === "status") {
              const status = parsed as StatusEvent;
              setLogs((prev) => [...prev, status]);
              setCurrentStep(status.step);
              setProgress(status.progress);

              // Mark previous steps as completed when we move to a new one
              const stepIndex = PIPELINE_STEPS.findIndex((s) => s.id === status.step);
              if (stepIndex > 0) {
                const completed = PIPELINE_STEPS.slice(0, stepIndex).map((s) => s.id);
                setCompletedSteps(completed);
              }
              if (status.step === "done") {
                setCompletedSteps(PIPELINE_STEPS.map((s) => s.id));
              }
            } else if (eventType === "result") {
              setResult(parsed as ResultData);
            } else if (eventType === "error") {
              setError(parsed.message);
            }
          } catch {
            // Skip malformed JSON
          }
        }
      }
    } catch (err) {
      console.error(err);
      setError(
        err instanceof Error ? err.message : "Something went wrong."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-zinc-50 px-6 py-12 text-zinc-900">
      <div className="mx-auto max-w-3xl">
        <p className="text-sm font-medium uppercase tracking-[0.2em] text-zinc-500">
          Multimodal AI Studio
        </p>

        <h1 className="mt-4 text-4xl font-semibold tracking-tight sm:text-5xl">
          Image captioning and translation for accessible AI experiences
        </h1>

        <p className="mt-6 text-lg leading-8 text-zinc-600">
          Upload an image to generate an English caption, a translated version,
          and a short machine-generated insight.
        </p>

        {/* Upload form */}
        <form
          onSubmit={handleSubmit}
          className="mt-10 rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm"
        >
          <label className="block text-sm font-medium text-zinc-700">
            Upload an image
          </label>

          <input
            type="file"
            accept="image/*"
            onChange={(e) => {
              setSelectedFile(e.target.files?.[0] ?? null);
              resetState();
            }}
            className="mt-3 block w-full rounded-lg border border-zinc-200 p-3 text-sm"
          />

          {/* Image preview */}
          {preview && (
            <div className="mt-4 overflow-hidden rounded-xl border border-zinc-100">
              <img
                src={preview}
                alt="Preview"
                className="h-auto max-h-64 w-full object-contain bg-zinc-100"
              />
            </div>
          )}

          <button
            type="submit"
            disabled={!selectedFile || loading}
            className="mt-5 rounded-full bg-zinc-900 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-zinc-700 disabled:opacity-50"
          >
            {loading ? "Running Pipeline..." : "Analyze Image"}
          </button>
        </form>

        {/* Pipeline status — shows during and after processing */}
        {(loading || logs.length > 0) && (
          <div className="mt-8 rounded-2xl border border-zinc-200 bg-white shadow-sm overflow-hidden">
            {/* Pipeline step indicators */}
            <div className="border-b border-zinc-100 px-5 py-4">
              <div className="flex items-center justify-between gap-1">
                {PIPELINE_STEPS.map((step, i) => {
                  const isActive = step.id === currentStep;
                  const isDone = completedSteps.includes(step.id);

                  return (
                    <div key={step.id} className="flex items-center gap-1">
                      <div
                        className={`
                          flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium transition-all duration-500
                          ${isActive
                            ? "bg-zinc-900 text-white scale-105"
                            : isDone
                              ? "bg-emerald-50 text-emerald-700"
                              : "bg-zinc-50 text-zinc-300"
                          }
                        `}
                      >
                        <span className="text-sm">{isDone && !isActive ? "✓" : step.icon}</span>
                        <span className="hidden sm:inline">{step.label}</span>
                      </div>
                      {i < PIPELINE_STEPS.length - 1 && (
                        <div
                          className={`h-px w-4 transition-colors duration-500 ${
                            isDone ? "bg-emerald-300" : "bg-zinc-100"
                          }`}
                        />
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Progress bar */}
            <div className="h-1 bg-zinc-50">
              <div
                className="h-full bg-zinc-900 transition-all duration-700 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>

            {/* Live log */}
            <div className="max-h-48 overflow-y-auto px-5 py-3 font-mono text-xs">
              {logs.map((log, i) => (
                <div
                  key={i}
                  className="flex gap-3 py-1 animate-in fade-in"
                  style={{ animationDelay: `${i * 50}ms` }}
                >
                  <span className="shrink-0 text-zinc-300 tabular-nums">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <span
                    className={`${
                      log.step === "done"
                        ? "text-emerald-600 font-semibold"
                        : "text-zinc-600"
                    }`}
                  >
                    {log.message}
                  </span>
                </div>
              ))}
              {loading && (
                <div className="flex gap-3 py-1">
                  <span className="shrink-0 text-zinc-300 tabular-nums">
                    {String(logs.length + 1).padStart(2, "0")}
                  </span>
                  <span className="text-zinc-400 animate-pulse">
                    Processing...
                  </span>
                </div>
              )}
              <div ref={logEndRef} />
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mt-6 rounded-2xl border border-red-200 bg-red-50 p-5">
            <h2 className="text-sm font-semibold text-red-600">Error</h2>
            <p className="mt-2 text-sm text-red-700">{error}</p>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="mt-8 space-y-4">
            <div className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm">
              <h2 className="text-sm font-semibold text-zinc-500">File</h2>
              <p className="mt-2 text-sm text-zinc-800">{result.filename}</p>
            </div>

            <div className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm">
              <h2 className="text-sm font-semibold text-zinc-500">
                English Caption
              </h2>
              <p className="mt-2 text-base text-zinc-800">{result.caption}</p>
            </div>

            <div className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm">
              <h2 className="text-sm font-semibold text-zinc-500">
                বাংলা Translation
              </h2>
              <p className="mt-2 text-base text-zinc-800">
                {result.translation}
              </p>
            </div>

            <div className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm">
              <h2 className="text-sm font-semibold text-zinc-500">Insight</h2>
              <p className="mt-2 text-base text-zinc-800">{result.insight}</p>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}