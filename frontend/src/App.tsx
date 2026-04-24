import { useState, useCallback, useEffect } from 'react';
import { RefreshCcw, Upload, Sparkles, BrainCircuit } from 'lucide-react';
import FileUpload from './components/FileUpload';
import ResultsView from './components/ResultsView';
import { ProcessingStatus } from './types';
import type { AnalyzedDocument, DocumentCluster } from './types';
import { getState, resetState, extractAndCluster } from './services/api';

function readFileText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsText(file);
  });
}

export default function App() {
  const [status, setStatus] = useState<ProcessingStatus>(ProcessingStatus.IDLE);
  const [documents, setDocuments] = useState<AnalyzedDocument[]>([]);
  const [clusters, setClusters] = useState<DocumentCluster[]>([]);
  const [progress, setProgress] = useState({ processed: 0, total: 0 });
  const [error, setError] = useState<string | null>(null);

  const isLoading = status === ProcessingStatus.EXTRACTING || status === ProcessingStatus.CLUSTERING;

  // Sync state on mount
  useEffect(() => {
    getState()
      .then((s: any) => {
        setDocuments(s.documents || []);
        setClusters(s.clusters || []);
        if (s.documents?.length > 0) setStatus(ProcessingStatus.COMPLETED);
      })
      .catch(() => { });
  }, []);

  const handleFilesAdded = useCallback(async (files: File[]) => {
    setError(null);
    const total = files.length;
    setProgress({ processed: 0, total });
    setStatus(ProcessingStatus.EXTRACTING);

    try {
      // Phase 1: Đọc tất cả files (local)
      const texts: string[] = [];
      const fileNames: string[] = [];
      for (let i = 0; i < total; i++) {
        texts.push(await readFileText(files[i]));
        fileNames.push(files[i].name);
        setProgress({ processed: i + 1, total });  // 1/n, 2/n — đọc file
      }

      // Phase 2: Gọi backend (TextRank + KeyBERT + LLM clustering)
      setStatus(ProcessingStatus.CLUSTERING);
      const result: any = await extractAndCluster(texts, fileNames);

      setClusters(result.final_clusters || []);
      setDocuments(result.all_documents || []);
      setStatus(ProcessingStatus.COMPLETED);
    } catch (e: unknown) {
      console.error('Lỗi:', e);
      setError(e instanceof Error ? e.message : String(e));
      setStatus(ProcessingStatus.ERROR);
    }
  }, []);

  const handleReset = useCallback(async () => {
    try {
      await resetState();
      setDocuments([]);
      setClusters([]);
      setStatus(ProcessingStatus.IDLE);
      setError(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  // ── Hero / Idle view ──
  const renderHero = () => (
    <div className="flex flex-col items-center justify-center py-20 space-y-8">
      {/* Badge */}
      <div className="flex items-center gap-2 px-4 py-1.5 bg-indigo-50 rounded-full border border-indigo-100">
        <Sparkles size={14} className="text-indigo-600" />
        <span className="text-xs font-medium text-indigo-600">
          TextRank + KeyBERT • Kilo AI Clustering
        </span>
      </div>

      {/* Heading */}
      <div className="text-center space-y-3">
        <h1 className="text-4xl font-bold text-slate-900">
          Phân Cụm Tài Liệu Theo Chủ Đề
        </h1>
        <p className="text-slate-500 max-w-lg mx-auto text-sm leading-relaxed">
          Tải lên tài liệu, hệ thống sẽ tự động trích xuất từ khóa, tóm tắt nội dung
          và phân cụm theo chủ đề thông minh.
        </p>
      </div>

      {/* Upload */}
      <div className="w-full max-w-lg">
        <FileUpload onFilesAdded={handleFilesAdded} disabled={isLoading} />
      </div>

      {/* Supported formats */}
      <div className="flex items-center gap-4 text-xs text-slate-400">
        <span>Định dạng: PDF, TXT</span>
        <span>•</span>
        <span>Nhiều file cùng lúc</span>
      </div>
    </div>
  );

  // ── Loading overlay ──
  const renderLoading = () => {
    const isExtracting = status === ProcessingStatus.EXTRACTING;
    const pct = progress.total > 0
      ? isExtracting
        ? Math.round((progress.processed / progress.total) * 50)   // Phase 1: 0→50%
        : 50 + Math.round((progress.processed / progress.total) * 50)  // never shown
      : 0;

    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/10 backdrop-blur-sm">
        <div className="bg-white rounded-2xl shadow-xl p-8 flex flex-col items-center space-y-5 min-w-80">
          {/* Spinner + icon */}
          <div className="relative">
            <div className="w-16 h-16 border-4 border-slate-200 border-t-indigo-600 rounded-full animate-spin" />
            <BrainCircuit
              size={24}
              className="absolute inset-0 m-auto text-indigo-600 animate-pulse"
            />
          </div>

          {/* Phase label */}
          <div className="flex gap-2">
            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${isExtracting
                ? 'bg-indigo-100 text-indigo-700'
                : 'bg-slate-100 text-slate-400 line-through'
              }`}>
              1 — Đọc file
            </span>
            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${!isExtracting
                ? 'bg-amber-100 text-amber-700'
                : 'bg-slate-100 text-slate-400'
              }`}>
              2 — Phân nhóm
            </span>
          </div>

          {/* Status text */}
          <div className="text-center space-y-1">
            {isExtracting ? (
              <>
                <p className="text-sm font-semibold text-slate-700">Đang đọc tài liệu…</p>
                {progress.total > 0 && (
                  <p className="text-xs text-slate-500">
                    Tài liệu&nbsp;
                    <span className="font-bold text-indigo-600">{progress.processed}</span>
                    &nbsp;/&nbsp;
                    <span className="font-bold">{progress.total}</span>&nbsp;đã đọc
                  </p>
                )}
              </>
            ) : (
              <>
                <p className="text-sm font-semibold text-slate-700">Đang phân nhóm theo chủ đề…</p>
                <p className="text-xs text-slate-500">
                  TextRank + KeyBERT + Kilo AI · Multi-label
                </p>
              </>
            )}
          </div>

          {/* Progress bar */}
          <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
            <div
              className="h-1.5 rounded-full transition-all duration-500 ease-out bg-indigo-600"
              style={{ width: isExtracting ? `${pct}%` : '100%' }}
            />
          </div>
        </div>
      </div>
    );
  };

  // ── Results view ──
  const renderResults = () => (
    <div>
      <ResultsView clusters={clusters} allDocuments={documents} />
    </div>
  );

  return (
    <div className="min-h-screen bg-[#fcfdfe] text-slate-900 font-sans">
      {/* ── Header ── */}
      <header className="sticky top-0 z-40 bg-white/80 backdrop-blur-md border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center">
                <Upload size={20} className="text-white" />
              </div>
              <div>
                <h1 className="text-base font-bold text-slate-900 leading-none">TaxonomyAI</h1>
                <p className="text-[11px] text-slate-500 leading-none mt-0.5">
                  Phân cụm tài liệu thông minh
                </p>
              </div>
            </div>

            {/* Actions */}
            <div className="flex items-center gap-3">
              {/* Compact upload */}
              <FileUpload
                onFilesAdded={handleFilesAdded}
                disabled={isLoading}
                compact
              />
              {/* Reset */}
              {documents.length > 0 && (
                <button
                  onClick={handleReset}
                  className="flex items-center gap-2 px-3 py-2 text-sm text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-all"
                >
                  <RefreshCcw size={14} />
                  Reset
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* ── Main ── */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        {/* Error banner */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-100 rounded-xl p-4 flex items-center justify-between">
            <p className="text-sm text-red-600">{error}</p>
            <button
              onClick={() => setError(null)}
              className="text-red-400 hover:text-red-600 text-lg leading-none"
            >
              ×
            </button>
          </div>
        )}

        {/* Content */}
        {isLoading ? renderLoading() : clusters.length > 0 ? renderResults() : renderHero()}
      </main>
    </div>
  );
}
