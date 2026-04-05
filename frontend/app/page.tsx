"use client";

import { useState } from "react";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<null | {
    filename: string;
    caption: string;
    translation: string;
    insight: string;
  }>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("file", selectedFile);

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch("https://bangla-image-captioning.onrender.com/analyze-image", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      console.log("API Response:", data);
      setResult(data);
    } catch (error) {
      console.error(error);
      alert("Something went wrong while uploading the image.");
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
            onChange={(e) => setSelectedFile(e.target.files?.[0] ?? null)}
            className="mt-3 block w-full rounded-lg border border-zinc-200 p-3 text-sm"
          />

          <button
            type="submit"
            disabled={!selectedFile || loading}
            className="mt-5 rounded-full bg-zinc-900 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-zinc-700 disabled:opacity-50"
          >
            {loading ? "Analyzing..." : "Analyze Image"}
          </button>
        </form>

        {result && (
          <div className="mt-8 space-y-4">
            <div className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm">
              <h2 className="text-sm font-semibold text-zinc-500">File</h2>
              <p className="mt-2 text-sm text-zinc-800">{result.filename}</p>
            </div>

            <div className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm">
              <h2 className="text-sm font-semibold text-zinc-500">English Caption</h2>
              <p className="mt-2 text-base text-zinc-800">{result.caption}</p>
            </div>

            <div className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm">
              <h2 className="text-sm font-semibold text-zinc-500">Translation</h2>
              <p className="mt-2 text-base text-zinc-800">{result.translation}</p>
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