import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import Papa from 'papaparse';
import { Upload, FileText, AlertCircle } from 'lucide-react';
import { parseCSVToTransactions } from '../utils/syntheticData';

export default function DataUploader({ onDataLoaded }) {
  const [fileInfo, setFileInfo]   = useState(null);
  const [error, setError]         = useState('');
  const [parsing, setParsing]     = useState(false);

  const onDrop = useCallback(acceptedFiles => {
    const file = acceptedFiles[0];
    if (!file) return;
    setError('');
    setParsing(true);

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (result) => {
        try {
          const txns = parseCSVToTransactions(result.data);
          if (txns.length === 0) {
            setError('No valid transactions found. Check your CSV columns.');
            setParsing(false);
            return;
          }
          setFileInfo({ name: file.name, rows: txns.length });
          onDataLoaded(txns, `CSV: ${file.name}`);
        } catch (e) {
          setError('Failed to parse CSV: ' + e.message);
        }
        setParsing(false);
      },
      error: (err) => {
        setError('Parse error: ' + err.message);
        setParsing(false);
      }
    });
  }, [onDataLoaded]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'text/csv': ['.csv'], 'text/plain': ['.txt'] }, maxFiles: 1,
  });

  return (
    <div className="upload-card">
      <div className="section-header" style={{ marginBottom: '1.25rem' }}>
        <div>
          <div className="section-title"><span>Upload</span> CSV Data</div>
          <div className="section-sub">customer_id · date · amount · category · channel</div>
        </div>
        <Upload size={18} color="var(--text3)" />
      </div>

      <div {...getRootProps()} className={`drop-zone ${isDragActive ? 'active' : ''}`}>
        <input {...getInputProps()} />
        <span className="drop-icon">📂</span>
        {isDragActive
          ? <div className="drop-title">Drop it!</div>
          : <div className="drop-title">{parsing ? 'Parsing...' : 'Drag & drop your CSV'}</div>
        }
        <div className="drop-sub">or click to browse · .csv files only</div>
      </div>

      {fileInfo && (
        <div className="file-info">
          <FileText size={14} />
          <span><strong>{fileInfo.name}</strong> — {fileInfo.rows.toLocaleString()} transactions loaded</span>
        </div>
      )}
      {error && (
        <div className="file-info" style={{ background: 'rgba(244,63,94,.07)', borderColor: 'rgba(244,63,94,.25)', color: '#f87171' }}>
          <AlertCircle size={14} />
          <span>{error}</span>
        </div>
      )}

      <div style={{ marginTop: '1rem', fontSize: '.75rem', color: 'var(--text3)', lineHeight: 1.6 }}>
        <strong style={{ color: 'var(--text2)' }}>Expected columns</strong><br />
        customer_id, transaction_date, amount<br />
        <em>Optional:</em> category, channel, quantity
      </div>
    </div>
  );
}
