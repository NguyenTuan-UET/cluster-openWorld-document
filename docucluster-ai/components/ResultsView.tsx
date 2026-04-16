
import React, { useState } from 'react';
import { AnalyzedDocument, DocumentCluster } from '../types';
import { FileText, Tag, Search, Filter, Info, FileCode, Users } from 'lucide-react';

interface ResultsViewProps {
  clusters: DocumentCluster[];
  allDocuments: AnalyzedDocument[];
}

export const ResultsView: React.FC<ResultsViewProps> = ({ clusters, allDocuments }) => {
  const [selectedClusterLabel, setSelectedClusterLabel] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');

  const sortedClusters = [...clusters].sort((a,b) => a.label.localeCompare(b.label));

  const activeCluster = clusters.find(c => c.label === selectedClusterLabel);

  const displayedDocs = activeCluster 
    ? activeCluster.documents 
    : allDocuments;

  const filteredDocs = displayedDocs.filter(doc => {
    return doc.fileName.toLowerCase().includes(searchTerm.toLowerCase()) || 
           doc.keyphrases.some(k => k.toLowerCase().includes(searchTerm.toLowerCase()));
  });

  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
      
      <aside className="lg:col-span-1 space-y-6">
        <div>
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
            <Users className="w-4 h-4" />
            Cụm Tài liệu ({clusters.length})
          </h3>
          <div className="flex flex-col gap-1">
            <button
              onClick={() => setSelectedClusterLabel(null)}
              className={`text-left px-3 py-2 rounded-lg text-sm font-medium transition-colors flex justify-between items-center ${!selectedClusterLabel ? 'bg-indigo-600 text-white shadow-md' : 'text-slate-600 hover:bg-slate-100'}`}
            >
              Tất cả tài liệu
              <span className="text-xs opacity-70">{allDocuments.length}</span>
            </button>
            
            {sortedClusters.map(cluster => (
              <button
                key={cluster.label}
                onClick={() => setSelectedClusterLabel(cluster.label)}
                className={`text-left px-3 py-2 rounded-lg text-sm font-medium transition-colors flex justify-between items-center group ${selectedClusterLabel === cluster.label ? 'bg-indigo-600 text-white shadow-md' : 'text-slate-600 hover:bg-slate-100'}`}
              >
                <span className="truncate pr-2">{cluster.label}</span>
                <span className={`text-xs px-1.5 py-0.5 rounded-full ${selectedClusterLabel === cluster.label ? 'bg-white/20' : 'bg-slate-200 text-slate-500'}`}>
                  {cluster.documents.length}
                </span>
              </button>
            ))}
          </div>
        </div>
        
        {activeCluster && (
           <div className="p-4 bg-indigo-50 border border-indigo-100 rounded-xl">
             <div className="flex items-center gap-2 text-indigo-700 font-bold text-sm mb-2">
               <Info className="w-4 h-4" />
               Về cụm "{activeCluster.label}"
             </div>
             <p className="text-xs text-indigo-600 leading-relaxed italic">
                Cụm này chứa {activeCluster.documents.length} tài liệu liên quan đến chủ đề {activeCluster.label.toLowerCase()}.
             </p>
           </div>
         )}
      </aside>

      <main className="lg:col-span-3 space-y-6">
        <div className="flex flex-col sm:flex-row gap-4 justify-between items-center bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
          <div className="relative flex-1 w-full">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
            <input 
              type="text" 
              placeholder="Tìm theo tên tệp hoặc từ khóa..."
              className="w-full pl-10 pr-4 py-2 text-sm border-none focus:ring-0 outline-none"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-400 whitespace-nowrap">
            <Filter className="w-3 h-3" />
            Hiển thị {filteredDocs.length} tài liệu
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4">
          {filteredDocs.map(doc => (
            <div key={doc.id} className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition-all group">
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-4 flex-1">
                  <div className={`p-3 rounded-lg flex-shrink-0 ${doc.type.includes('pdf') ? 'bg-rose-50 text-rose-500' : 'bg-sky-50 text-sky-500'}`}>
                    {doc.type.includes('pdf') ? <FileText className="w-6 h-6" /> : <FileCode className="w-6 h-6" />}
                  </div>
                  <div className="min-w-0">
                    <h4 className="font-bold text-slate-900 group-hover:text-indigo-600 transition-colors truncate">{doc.fileName}</h4>
                    <p className="text-sm text-slate-500 mt-1 line-clamp-1 italic">{doc.summary}</p>
                  </div>
                </div>
                <div className="text-right hidden sm:block flex-shrink-0 ml-4">
                  <span className="text-[10px] font-bold text-slate-300 uppercase tracking-tighter">Từ khóa</span>
                  <div className="flex flex-wrap gap-1 mt-1 justify-end max-w-[150px]">
                    {doc.keyphrases.slice(0, 3).map((kp, i) => (
                      <span key={i} className="px-1.5 py-0.5 bg-slate-50 text-slate-400 text-[9px] rounded uppercase font-medium border border-slate-100">
                        {kp}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ))}
          {filteredDocs.length === 0 && (
            <div className="py-20 text-center bg-white rounded-xl border-2 border-dashed border-slate-200">
              <Search className="w-12 h-12 text-slate-200 mx-auto mb-4" />
              <p className="text-slate-400 font-medium">Không tìm thấy tài liệu phù hợp</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};
