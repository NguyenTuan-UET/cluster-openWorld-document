
import React, { useState } from 'react';
import { FileUpload } from './components/FileUpload';
import { ResultsView } from './components/ResultsView';
import { processAndClusterNewDocuments } from './services/geminiService';
import { ProcessingStatus, AnalyzedDocument, DocumentCluster, ProcessingError } from './types';
import { BrainCircuit, RefreshCcw, Sparkles, FolderTree } from 'lucide-react';

export default function App() {
  const [status, setStatus] = useState<ProcessingStatus>(ProcessingStatus.IDLE);
  const [documents, setDocuments] = useState<AnalyzedDocument[]>([]);
  const [clusters, setClusters] = useState<DocumentCluster[]>([]);
  const [error, setError] = useState<ProcessingError | null>(null);
  const [progress, setProgress] = useState({ processed: 0, total: 0 });

  const handleReset = () => {
    if (confirm("Bạn có chắc chắn muốn xóa toàn bộ thư viện tài liệu không?")) {
      setDocuments([]);
      setClusters([]);
      setStatus(ProcessingStatus.IDLE);
      setError(null);
    }
  };

  const processWorkflow = async (files: File[]) => {
    if (files.length === 0) return;
    
    setStatus(ProcessingStatus.EXTRACTING);
    setProgress({ processed: 0, total: files.length });
    setError(null);

    try {
      // Step 1 is now integrated inside the main service call
      // which handles extraction.
      
      // We show "Clustering" as a general state for assignment and new cluster creation
      setStatus(ProcessingStatus.CLUSTERING);
      
      const { finalClusters, allDocuments } = await processAndClusterNewDocuments(
        files,
        clusters, // Pass existing clusters
        documents // Pass existing documents
      );

      setClusters(finalClusters);
      setDocuments(allDocuments);

      setStatus(ProcessingStatus.COMPLETED);
    } catch (err: any) {
      setError({ message: err.message || "Đã xảy ra lỗi hệ thống trong quá trình phân cụm." });
      setStatus(ProcessingStatus.ERROR);
    }
  };

  const isLoading = status === ProcessingStatus.EXTRACTING || status === ProcessingStatus.CLUSTERING;

  return (
    <div className="min-h-screen bg-[#fcfdfe] text-slate-900 pb-20 font-sans">
      
      <header className="bg-white/80 backdrop-blur-md border-b sticky top-0 z-30 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center text-white shadow-indigo-200 shadow-lg">
              <FolderTree className="w-6 h-6" />
            </div>
            <div>
              <h1 className="font-extrabold text-xl tracking-tight text-slate-900">TaxonomyAI</h1>
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest leading-none mt-0.5">Phân cụm Tài liệu Thông minh</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            {documents.length > 0 && (
              <FileUpload onFilesSelected={processWorkflow} compact disabled={isLoading} />
            )}
            {documents.length > 0 && (
              <button 
                onClick={handleReset}
                title="Xóa và làm lại"
                className="p-2 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-all"
              >
                <RefreshCcw className="w-5 h-5" />
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        
        {status === ProcessingStatus.IDLE && documents.length === 0 && (
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-12">
              <div className="inline-flex items-center gap-2 px-3 py-1 bg-indigo-50 text-indigo-700 rounded-full text-xs font-bold uppercase tracking-wider mb-6">
                <Sparkles className="w-3 h-3" />
                Hỗ trợ bởi Gemini 2.5 Flash
              </div>
              <h2 className="text-5xl font-black text-slate-900 tracking-tight mb-6">
                Phân cụm Tài liệu theo <span className="text-indigo-600 underline decoration-indigo-200 underline-offset-8">Chủ đề</span>
              </h2>
              <p className="text-lg text-slate-500 mb-10 max-w-2xl mx-auto leading-relaxed">
                Tải lên một loạt tài liệu. AI sẽ trích xuất các chủ đề chính từ mỗi tài liệu, sau đó tự động nhóm chúng vào các cụm có ý nghĩa.
              </p>
            </div>
            <div className="max-w-xl mx-auto">
              <FileUpload onFilesSelected={processWorkflow} />
            </div>
          </div>
        )}

        {isLoading && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/10 backdrop-blur-sm">
            <div className="bg-white p-8 rounded-2xl shadow-2xl border border-slate-100 max-w-sm w-full text-center">
              <div className="relative w-20 h-20 mx-auto mb-6">
                 <div className="absolute inset-0 border-4 border-slate-100 rounded-full"></div>
                 <div className="absolute inset-0 border-4 border-indigo-600 rounded-full border-t-transparent animate-spin"></div>
                 <div className="absolute inset-0 flex items-center justify-center">
                   <BrainCircuit className="w-8 h-8 text-indigo-600 animate-pulse" />
                 </div>
              </div>
              <h3 className="text-xl font-bold text-slate-900 mb-2">
                {status === ProcessingStatus.CLUSTERING ? "Cập nhật & Tạo cụm" : "Trích xuất Chủ đề"}
              </h3>
              <p className="text-slate-500 text-sm mb-6">
                {/* FIX: Use progress.total from state instead of files.length which is out of scope. */}
                {status === ProcessingStatus.CLUSTERING
                  ? "Phân tích và sắp xếp các tài liệu mới..."
                  : `Đang xử lý ${progress.total} tệp...`}
              </p>
            </div>
          </div>
        )}

        {clusters.length > 0 && !isLoading && (
          <ResultsView clusters={clusters} allDocuments={documents} />
        )}

        {status === ProcessingStatus.ERROR && (
          <div className="max-w-lg mx-auto mt-12 p-8 bg-white rounded-2xl shadow-xl border border-red-100 text-center">
            <h3 className="text-xl font-bold text-slate-900 mb-2">Lỗi Hệ thống</h3>
            <p className="text-slate-600 mb-6">{error?.message}</p>
            <button onClick={() => setStatus(ProcessingStatus.COMPLETED)} className="px-6 py-2 bg-indigo-600 text-white rounded-lg font-bold">
              Đóng
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
