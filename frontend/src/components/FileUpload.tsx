import { Upload } from 'lucide-react';

interface FileUploadProps {
  onFilesAdded: (files: File[]) => void;
  disabled?: boolean;
  compact?: boolean;
}

export default function FileUpload({ onFilesAdded, disabled, compact }: FileUploadProps) {
  const handleFiles = (files: FileList | null) => {
    if (!files || disabled) return;
    const selected = Array.from(files).filter(
      (f) => f.type === 'application/pdf' || f.type === 'text/plain'
    );
    if (selected.length > 0) onFilesAdded(selected);
    // Reset input to allow re-uploading same file
    (document.activeElement as HTMLInputElement)?.blur();
  };

  if (compact) {
    return (
      <label
        className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors cursor-pointer
          ${disabled
            ? 'bg-indigo-100 text-indigo-400 cursor-not-allowed'
            : 'bg-indigo-600 text-white hover:bg-indigo-700'
          }`}
      >
        <Upload size={16} />
        Thêm tài liệu
        <input
          type="file"
          accept=".pdf,.txt,application/pdf,text/plain"
          multiple
          onChange={(e) => handleFiles(e.target.files)}
          disabled={disabled}
          className="hidden"
        />
      </label>
    );
  }

  return (
    <label
      className={`flex flex-col items-center justify-center h-64 border-2 border-dashed rounded-2xl cursor-pointer transition-all
        ${disabled
          ? 'border-slate-200 bg-slate-50 cursor-not-allowed'
          : 'border-indigo-300 bg-indigo-50/50 hover:border-indigo-500 hover:bg-indigo-50'
        }`}
    >
      <div className="flex gap-6 mb-4">
        {/* PDF badge */}
        <div className="flex flex-col items-center gap-1">
          <div className="w-12 h-12 rounded-xl bg-rose-100 flex items-center justify-center">
            <span className="text-rose-600 font-bold text-sm">PDF</span>
          </div>
        </div>
        {/* TXT badge */}
        <div className="flex flex-col items-center gap-1">
          <div className="w-12 h-12 rounded-xl bg-sky-100 flex items-center justify-center">
            <span className="text-sky-600 font-bold text-sm">TXT</span>
          </div>
        </div>
      </div>

      <p className="text-slate-700 font-medium mb-1">
        Kéo & thả file vào đây
      </p>
      <p className="text-slate-500 text-sm mb-4">hoặc nhấn để chọn</p>

      <span className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700 transition-colors">
        Chọn file
      </span>

      <input
        type="file"
        accept=".pdf,.txt,application/pdf,text/plain"
        multiple
        onChange={(e) => handleFiles(e.target.files)}
        disabled={disabled}
        className="hidden"
      />
    </label>
  );
}
