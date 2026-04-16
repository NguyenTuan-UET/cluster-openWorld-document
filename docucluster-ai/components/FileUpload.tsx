
import React from 'react';
import { Upload, FileText, FileCode } from 'lucide-react';

interface FileUploadProps {
  onFilesSelected: (files: File[]) => void;
  disabled?: boolean;
  compact?: boolean;
}

export const FileUpload: React.FC<FileUploadProps> = ({ onFilesSelected, disabled, compact }) => {
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFilesSelected(Array.from(e.target.files));
      // Clear the input value to allow re-uploading the same file
      e.target.value = '';
    }
  };

  if (compact) {
    return (
      <label className={`flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg cursor-pointer transition-colors shadow-sm text-sm font-medium ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}>
        <Upload className="w-4 h-4" />
        Thêm tài liệu
        <input 
          type="file" 
          className="hidden" 
          multiple 
          accept="application/pdf,text/plain"
          onChange={handleFileChange}
          disabled={disabled}
        />
      </label>
    );
  }

  return (
    <div className="w-full">
      <label 
        className={`
          flex flex-col items-center justify-center w-full h-64 
          border-2 border-dashed rounded-xl cursor-pointer 
          transition-all duration-300
          ${disabled 
            ? 'bg-gray-100 border-gray-300 cursor-not-allowed opacity-60' 
            : 'bg-white border-blue-200 hover:bg-blue-50 hover:border-blue-400 shadow-sm'}
        `}
      >
        <div className="flex flex-col items-center justify-center pt-5 pb-6 text-center px-4">
          <div className="flex gap-2 mb-4">
            <div className="p-3 bg-red-100 text-red-600 rounded-lg"><FileText className="w-6 h-6" /></div>
            <div className="p-3 bg-blue-100 text-blue-600 rounded-lg"><FileCode className="w-6 h-6" /></div>
          </div>
          <p className="mb-2 text-lg font-semibold text-gray-700">
            {disabled ? "Đang xử lý..." : "Tải lên tệp PDF hoặc TXT"}
          </p>
          <p className="text-sm text-gray-500 max-w-sm">
            Tài liệu sẽ được tự động phân loại vào các cụm chủ đề động bằng Gemini AI.
          </p>
        </div>
        <input 
          type="file" 
          className="hidden" 
          multiple 
          accept="application/pdf,text/plain"
          onChange={handleFileChange}
          disabled={disabled}
        />
      </label>
    </div>
  );
};
