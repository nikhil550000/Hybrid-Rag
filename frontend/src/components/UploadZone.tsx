import { useRef, useState, type ChangeEvent, type DragEvent } from 'react';
import { AlertCircle, CheckCircle2, FileText, LoaderCircle, UploadCloud, X } from 'lucide-react';
import { toErrorResponse, uploadDocuments } from '../api';
import type { UploadResponse } from '../types';

interface UploadZoneProps {
  onSessionCreated: (session: UploadResponse) => void;
  onUploadStateChange?: (isUploading: boolean) => void;
  disabled?: boolean;
}

const MAX_FILES = 5;
const MAX_FILE_BYTES = 20 * 1024 * 1024;
const MAX_TOTAL_BYTES = 50 * 1024 * 1024;

function formatFileSize(bytes: number): string {
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

function fileKey(file: File): string {
  return `${file.name}:${file.size}:${file.lastModified}`;
}

export function UploadZone({
  onSessionCreated,
  onUploadStateChange,
  disabled = false,
}: UploadZoneProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const addFiles = (newFiles: File[]) => {
    const nextFiles = [...files];
    const existingKeys = new Set(nextFiles.map(fileKey));
    let totalBytes = nextFiles.reduce((total, file) => total + file.size, 0);
    const problems: string[] = [];

    for (const file of newFiles) {
      if (!file.name.toLowerCase().endsWith('.pdf')) {
        problems.push(`${file.name} is not a PDF.`);
        continue;
      }
      if (file.size > MAX_FILE_BYTES) {
        problems.push(`${file.name} exceeds the 20 MB per-file limit.`);
        continue;
      }
      if (existingKeys.has(fileKey(file))) continue;
      if (nextFiles.length >= MAX_FILES) {
        problems.push('A custom session can contain at most 5 PDFs.');
        break;
      }
      if (totalBytes + file.size > MAX_TOTAL_BYTES) {
        problems.push('The selected PDFs exceed the 50 MB total limit.');
        continue;
      }

      nextFiles.push(file);
      existingKeys.add(fileKey(file));
      totalBytes += file.size;
    }

    setFiles(nextFiles);
    setError(problems.length > 0 ? problems[0] : null);
  };

  const handleDragOver = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    if (!disabled && !isUploading) setIsDragging(true);
  };

  const handleDragLeave = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setIsDragging(false);
    if (disabled || isUploading) return;
    addFiles(Array.from(event.dataTransfer.files));
  };

  const handleFileInput = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) addFiles(Array.from(event.target.files));
    event.target.value = '';
  };

  const removeFile = (index: number) => {
    setFiles((current) => current.filter((_, fileIndex) => fileIndex !== index));
    setError(null);
  };

  const handleUpload = async () => {
    if (files.length === 0 || disabled || isUploading) return;

    setIsUploading(true);
    onUploadStateChange?.(true);
    setError(null);

    try {
      const session = await uploadDocuments(files);
      onSessionCreated(session);
    } catch (caughtError: unknown) {
      const apiError = toErrorResponse(caughtError);
      setError(apiError.request_id
        ? `${apiError.message} Request ID: ${apiError.request_id}`
        : apiError.message);
    } finally {
      setIsUploading(false);
      onUploadStateChange?.(false);
    }
  };

  const totalSize = files.reduce((total, file) => total + file.size, 0);
  const controlsDisabled = disabled || isUploading;

  return (
    <section className="upload-panel" aria-labelledby="upload-title">
      <div className="upload-heading">
        <div>
          <p className="eyebrow">Isolated document session</p>
          <h2 id="upload-title">Build a custom corpus</h2>
        </div>
        <span>{files.length}/{MAX_FILES} PDFs · {formatFileSize(totalSize)}</span>
      </div>

      <label
        htmlFor="pdf-upload"
        className={`upload-zone ${isDragging ? 'drag-active' : ''} ${controlsDisabled ? 'disabled' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          id="pdf-upload"
          className="sr-only"
          type="file"
          multiple
          accept=".pdf,application/pdf"
          ref={fileInputRef}
          onChange={handleFileInput}
          disabled={controlsDisabled}
        />
        <span className="upload-icon" aria-hidden="true"><UploadCloud size={25} /></span>
        <strong>Drop PDFs here or choose files</strong>
        <span>20 MB per file · 50 MB total</span>
      </label>

      {error && (
        <div className="upload-error" role="alert">
          <AlertCircle size={17} aria-hidden="true" />
          <span>{error}</span>
        </div>
      )}

      {files.length > 0 && (
        <div className="selected-files" aria-label="Selected PDF files">
          {files.map((file, index) => (
            <div className="file-row" key={fileKey(file)}>
              <FileText size={17} aria-hidden="true" />
              <span className="file-name" title={file.name}>{file.name}</span>
              <span className="file-size">{formatFileSize(file.size)}</span>
              <button
                type="button"
                className="icon-button"
                onClick={() => removeFile(index)}
                disabled={controlsDisabled}
                aria-label={`Remove ${file.name}`}
                title={`Remove ${file.name}`}
              >
                <X size={16} aria-hidden="true" />
              </button>
            </div>
          ))}
        </div>
      )}

      {isUploading && (
        <div className="upload-progress" role="status">
          <LoaderCircle size={22} className="spinner" aria-hidden="true" />
          <div>
            <strong>Uploading and indexing {files.length} {files.length === 1 ? 'PDF' : 'PDFs'}</strong>
            <span>Extracting text, creating embeddings, and building retrieval indexes.</span>
          </div>
        </div>
      )}

      <button
        type="button"
        className="primary-button upload-submit"
        onClick={() => void handleUpload()}
        disabled={controlsDisabled || files.length === 0}
      >
        {isUploading
          ? <LoaderCircle size={18} className="spinner" aria-hidden="true" />
          : <CheckCircle2 size={18} aria-hidden="true" />}
        {isUploading ? 'Indexing documents' : 'Start custom session'}
      </button>
    </section>
  );
}
