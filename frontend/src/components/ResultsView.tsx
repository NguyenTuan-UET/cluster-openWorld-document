import { Search, Users, FileText } from 'lucide-react';
import { useState, useMemo } from 'react';
import type { DocumentCluster, AnalyzedDocument } from '../types';

interface ResultsViewProps {
  clusters: DocumentCluster[];
  allDocuments: AnalyzedDocument[];
}

export default function ResultsView({ clusters, allDocuments }: ResultsViewProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCluster, setSelectedCluster] = useState<string | null>(null);

  // Docs to show based on cluster filter
  const displayedDocs = useMemo(() => {
    if (!selectedCluster) return allDocuments;
    const cluster = clusters.find((c) => c.label === selectedCluster);
    return cluster?.documents ?? [];
  }, [selectedCluster, clusters, allDocuments]);

  // Filter docs by search term
  const filteredDocs = useMemo(() => {
    if (!searchTerm.trim()) return displayedDocs;
    const q = searchTerm.toLowerCase();
    return displayedDocs.filter(
      (doc) =>
        doc.fileName.toLowerCase().includes(q) ||
        doc.keyphrases.some((k) => k.toLowerCase().includes(q))
    );
  }, [displayedDocs, searchTerm]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
      {/* ── Sidebar ── */}
      <aside className="lg:col-span-1 space-y-4">
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-100 flex items-center gap-2">
            <Users size={16} className="text-slate-500" />
            <span className="text-sm font-semibold text-slate-700">
              Cụm tài liệu ({clusters.length})
            </span>
          </div>

          <div className="p-2 space-y-1">
            {/* All docs button */}
            <button
              onClick={() => setSelectedCluster(null)}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition-all flex items-center justify-between
                ${selectedCluster === null
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-600 hover:bg-slate-100'
                }`}
            >
              <span>Tất cả tài liệu</span>
              <span className={`text-xs px-2 py-0.5 rounded-full
                ${selectedCluster === null ? 'bg-indigo-500 text-white' : 'bg-slate-200 text-slate-500'}`}>
                {allDocuments.length}
              </span>
            </button>

            {/* Cluster buttons */}
            {[...clusters]
              .sort((a, b) => a.label.localeCompare(b.label))
              .map((cluster) => (
                <button
                  key={cluster.label}
                  onClick={() =>
                    setSelectedCluster(selectedCluster === cluster.label ? null : cluster.label)
                  }
                  className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition-all flex items-center justify-between group
                    ${selectedCluster === cluster.label
                      ? 'bg-indigo-600 text-white shadow-sm'
                      : 'text-slate-600 hover:bg-slate-100'
                    }`}
                >
                  <span className="truncate flex-1 mr-2">{cluster.label}</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full shrink-0
                    ${selectedCluster === cluster.label ? 'bg-indigo-500 text-white' : 'bg-slate-200 text-slate-500'}`}>
                    {cluster.documents.length}
                  </span>
                </button>
              ))}
          </div>
        </div>

        {/* Cluster info card */}
        {selectedCluster && (() => {
          const cluster = clusters.find((c) => c.label === selectedCluster);
          if (!cluster) return null;
          const topKeywords = [...new Set(cluster.documents.flatMap((d) => d.keyphrases.slice(0, 3)))].slice(0, 6);
          return (
            <div className="bg-indigo-50 rounded-xl p-4 space-y-2">
              <p className="text-sm font-semibold text-indigo-900">
                Về cụm "{cluster.label}"
              </p>
              <p className="text-xs text-indigo-700">
                {cluster.documents.length} tài liệu
              </p>
              {topKeywords.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-2">
                  {topKeywords.map((kw, i) => (
                    <span key={i} className="text-xs px-2 py-0.5 bg-indigo-100 text-indigo-700 rounded-full">
                      {kw}
                    </span>
                  ))}
                </div>
              )}
            </div>
          );
        })()}
      </aside>

      {/* ── Main content ── */}
      <main className="lg:col-span-3 space-y-4">
        {/* Search bar */}
        <div className="flex items-center gap-3">
          <div className="flex-1 relative">
            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              placeholder="Tìm theo tên file hoặc từ khóa..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-slate-200 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
          </div>
          {searchTerm && (
            <span className="text-xs text-slate-500">
              {filteredDocs.length} kết quả
            </span>
          )}
        </div>

        {/* Document grid */}
        {filteredDocs.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 text-slate-400">
            <Search size={48} className="mb-4 opacity-30" />
            <p className="text-sm">Không tìm thấy tài liệu phù hợp</p>
          </div>
        ) : (
          <div className="space-y-4">
            {filteredDocs.map((doc) => (
              <DocCard key={doc.id} doc={doc} />
            ))}
          </div>
        )}
      </main>
    </div>
  );
}

function DocCard({ doc }: { doc: AnalyzedDocument }) {
  const isPdf = doc.fileName?.toLowerCase().endsWith('.pdf');

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4 hover:shadow-md transition-all group">
      <div className="flex items-start gap-3">
        {/* File type icon */}
        <div className={`shrink-0 w-10 h-10 rounded-lg flex items-center justify-center
          ${isPdf ? 'bg-rose-100' : 'bg-sky-100'}`}>
          {isPdf ? (
            <span className={`text-xs font-bold ${isPdf ? 'text-rose-600' : 'text-sky-600'}`}>PDF</span>
          ) : (
            <FileText size={18} className="text-sky-600" />
          )}
        </div>

        <div className="flex-1 min-w-0">
          {/* File name */}
          <h3 className="text-sm font-semibold text-slate-900 group-hover:text-indigo-600 transition-colors truncate">
            {doc.fileName || 'Untitled'}
          </h3>

          {/* Summary */}
          {doc.summary && (
            <p className="text-xs text-slate-500 italic mt-1 line-clamp-1">
              {doc.summary}
            </p>
          )}

          {/* Keyphrases */}
          {doc.keyphrases.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {doc.keyphrases.slice(0, 5).map((kw, i) => (
                <span
                  key={i}
                  className="text-[10px] uppercase tracking-wide px-2 py-0.5 bg-slate-100 text-slate-600 rounded font-medium"
                >
                  {kw}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
