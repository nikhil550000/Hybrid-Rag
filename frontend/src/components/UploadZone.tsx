import { useState, useRef } from 'react';
import { UploadCloud, FileText, X, CheckCircle, AlertCircle } from 'lucide-react';
import type { UploadResponse } from '../types';

interface UploadZoneProps {
  onSessionCreated: (session: UploadResponse) => void;
}

export const UploadZone = ({ onSessionCreated }: UploadZoneProps) => {
  const [files, setFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      addFiles(Array.from(e.target.files));
    }
  };

  const addFiles = (newFiles: File[]) => {
    setError(null);
    const pdfFiles = newFiles.filter(file => file.type === 'application/pdf' || file.name.endsWith('.pdf'));
    
    if (pdfFiles.length !== newFiles.length) {
      setError("Only PDF files are supported.");
    }
    
    setFiles(prev => {
      const combined = [...prev, ...pdfFiles];
      if (combined.length > 5) {
        setError("Maximum 5 files allowed per session.");
        return combined.slice(0, 5);
      }
      return combined;
    });
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    if (files.length === 0) return;
    
    setIsUploading(true);
    setError(null);
    
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    try {
      const response = await fetch('http://127.0.0.1:8000/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        setError(data.detail?.message || data.detail || 'Upload failed');
      } else {
        onSessionCreated(data as UploadResponse);
      }
    } catch (err) {
      setError("Failed to connect to the server.");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="upload-container">
      <div 
        className={`upload-zone glass-panel ${isDragging ? 'drag-active' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input 
          type="file" 
          multiple 
          accept=".pdf,application/pdf" 
          ref={fileInputRef}
          onChange={handleFileInput}
          style={{ display: 'none' }}
        />
        
        <UploadCloud size={48} className="upload-icon" />
        <h3>Upload PDFs for Custom Session</h3>
        <p>Drag & drop or click to select files (Max 5)</p>
      </div>

      {error && (
        <div className="upload-error">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {files.length > 0 && (
        <div className="file-list">
          {files.map((file, idx) => (
            <div key={idx} className="file-item glass-panel">
              <div className="file-item-info">
                <FileText size={16} className="text-accent" />
                <span className="file-name" title={file.name}>{file.name}</span>
                <span className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
              </div>
              <button 
                className="remove-file-btn" 
                onClick={(e) => { e.stopPropagation(); removeFile(idx); }}
                disabled={isUploading}
              >
                <X size={16} />
              </button>
            </div>
          ))}

          <button 
            className="upload-submit-btn" 
            onClick={handleUpload}
            disabled={isUploading || files.length === 0}
          >
            {isUploading ? (
              <span className="flex-center gap-2"><div className="spinner-small" /> Processing...</span>
            ) : (
              <span className="flex-center gap-2"><CheckCircle size={18} /> Start Custom Session</span>
            )}
          </button>
        </div>
      )}
    </div>
  );
};
