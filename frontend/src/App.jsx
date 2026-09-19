import { useState, useRef, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import './index.css'
import './App.css'

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

function apiUrl(path) {
  return `${API_BASE}${path}`
}

function resolveApiUrl(path) {
  if (!path) return path
  if (/^https?:\/\//i.test(path)) return path
  return apiUrl(path)
}

function formatRenderText(content) {
  if (content == null) return ''
  if (typeof content === 'string') return content
  if (Array.isArray(content)) {
    return content
      .map((item) => (typeof item === 'object' && item !== null ? item.text || item.content || JSON.stringify(item) : String(item)))
      .join('\n')
  }
  if (typeof content === 'object') {
    return content.text || content.content || JSON.stringify(content)
  }
  return String(content)
}

async function readJsonResponse(res) {
  const text = await res.text()
  if (!text) return { data: {}, ok: true }
  try {
    return { data: JSON.parse(text), ok: true }
  } catch {
    return {
      data: null,
      ok: false,
      parseError: 'Something went wrong while generating the answer. Please try again.',
    }
  }
}

// ── Icons ────────────────────────────────────────────────────────────────────
const IconUpload = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
    <polyline points="17 8 12 3 7 8"/>
    <line x1="12" y1="3" x2="12" y2="15"/>
  </svg>
)
const IconSearch = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
  </svg>
)
const IconCheck = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12"/>
  </svg>
)
const IconAlert = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
  </svg>
)
const IconGlobe = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"/>
    <line x1="2" y1="12" x2="22" y2="12"/>
    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
  </svg>
)
const IconDoc = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
    <polyline points="14 2 14 8 20 8"/>
  </svg>
)

// ── DocuMind AI Brand Logo ───────────────────────────────────────────────────
const DocuMindLogo = () => (
  <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="DocuMind AI Logo">
    <defs>
      <linearGradient id="docuGrad" x1="2" y1="2" x2="30" y2="30" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stopColor="#38bdf8" />
        <stop offset="50%" stopColor="#818cf8" />
        <stop offset="100%" stopColor="#c084fc" />
      </linearGradient>
      <linearGradient id="docBorder" x1="8" y1="6" x2="24" y2="26" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stopColor="#ffffff" stopOpacity="0.9" />
        <stop offset="100%" stopColor="#cbd5e1" stopOpacity="0.5" />
      </linearGradient>
    </defs>
    {/* Base plate with rounded gradient */}
    <rect x="2.5" y="2.5" width="27" height="27" rx="8" fill="url(#docuGrad)" />
    <rect x="3.5" y="3.5" width="25" height="25" rx="7" fill="#0b0f19" fillOpacity="0.65" />
    
    {/* Stylized Document Outline with Fold */}
    <path
      d="M10 9.5C10 8.67157 10.6716 8 11.5 8H17.5L22 12.5V22.5C22 23.3284 21.3284 24 20.5 24H11.5C10.6716 24 10 23.3284 10 22.5V9.5Z"
      fill="url(#docuGrad)"
      fillOpacity="0.22"
      stroke="url(#docBorder)"
      strokeWidth="1.4"
      strokeLinejoin="round"
    />
    {/* Fold corner */}
    <path d="M17.5 8V12.5H22" stroke="url(#docBorder)" strokeWidth="1.4" strokeLinejoin="round" />
    
    {/* Neural AI Core Nodes & Connections */}
    <path d="M13 18.5L16 15.5L19 18.5" stroke="#38bdf8" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M16 12.5V15.5" stroke="#818cf8" strokeWidth="1.3" strokeLinecap="round" />
    <circle cx="16" cy="15.5" r="1.8" fill="#38bdf8" />
    <circle cx="13" cy="18.5" r="1.3" fill="#c084fc" />
    <circle cx="19" cy="18.5" r="1.3" fill="#c084fc" />
    <circle cx="16" cy="12.5" r="1.1" fill="#ffffff" />
  </svg>
)

// ── Spinner ──────────────────────────────────────────────────────────────────
const Spinner = () => <span className="spinner" aria-label="Loading" />

// ── Evidence Item Component ──────────────────────────────────────────────────
const EvidenceItem = ({ item }) => {
  const [showImage, setShowImage] = useState(false)
  const isWeb = item.type === 'web'
  const isImage = item.content_type === 'image' && !!item.image_url

  const rawText = formatRenderText(item.content || item.snippet)
  const previewText = rawText.length > 220 ? `${rawText.slice(0, 220).trim()}…` : rawText

  return (
    <li className="evidence-card">
      <div className="evidence-header">
        <div className="evidence-meta">
          <span className="evidence-icon">
            {isWeb ? <IconGlobe /> : <IconDoc />}
          </span>
          <span className="evidence-source" title={item.source || item.title}>
            {item.source || item.title || (isWeb ? 'Web Source' : 'Document')}
          </span>
          {!isWeb && item.page && (
            <span className="evidence-page">Page {item.page}</span>
          )}
        </div>
        <span className={`evidence-badge ${isWeb ? 'badge-web' : 'badge-doc'}`}>
          {isWeb ? 'Web' : item.content_type || 'Doc'}
        </span>
      </div>

      {previewText && (
        <p className="evidence-text">
          {previewText}
        </p>
      )}

      {isWeb && item.url && (
        <a href={item.url} target="_blank" rel="noopener noreferrer" className="evidence-link">
          Visit source ↗
        </a>
      )}

      {isImage && (
        <div className="evidence-image-container">
          <button
            type="button"
            className="evidence-image-toggle"
            onClick={() => setShowImage((prev) => !prev)}
          >
            {showImage ? 'Hide extracted image' : 'View extracted image'}
          </button>
          {showImage && (
            <img
              src={resolveApiUrl(item.image_url)}
              alt="Extracted evidence visual"
              className="evidence-image-preview"
            />
          )}
        </div>
      )}
    </li>
  )
}

// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  // Document state
  const [selectedFile, setSelectedFile] = useState(null)
  const [ingesting, setIngesting] = useState(false)
  const [ingestStatus, setIngestStatus] = useState(null)
  const fileInputRef = useRef(null)
  const ingestSeqRef = useRef(0)

  // Query state
  const [question, setQuestion] = useState('')
  const [querying, setQuerying] = useState(false)
  const [result, setResult] = useState(null)
  const [queryError, setQueryError] = useState(null)
  const querySeqRef = useRef(0)

  // Evidence view toggle
  const [showAllEvidence, setShowAllEvidence] = useState(false)
  const [dragging, setDragging] = useState(false)

  // ── Auto-Ingest on File Upload ──
  const handleIngestFile = async (fileToIngest) => {
    if (!fileToIngest || ingesting) return

    setSelectedFile(fileToIngest)
    const seq = ++ingestSeqRef.current
    setIngesting(true)
    setIngestStatus(null)
    try {
      const formData = new FormData()
      formData.append('file', fileToIngest)

      const res = await fetch(apiUrl('/ingest'), { method: 'POST', body: formData })
      const { data, ok: jsonOk } = await readJsonResponse(res)
      if (seq !== ingestSeqRef.current) return

      if (!jsonOk || !res.ok) {
        setIngestStatus({
          ok: false,
          msg: "Couldn't process this document. Please try another file.",
        })
      } else {
        setIngestStatus({
          ok: true,
          msg: `Ready: ${fileToIngest.name} analyzed and indexed.`,
        })
      }
    } catch {
      if (seq !== ingestSeqRef.current) return
      setIngestStatus({
        ok: false,
        msg: "Couldn't process this document. Please try another file.",
      })
    } finally {
      if (seq === ingestSeqRef.current) setIngesting(false)
    }
  }

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) handleIngestFile(file)
  }, [ingesting])

  const handleDragOver = (e) => { e.preventDefault(); setDragging(true) }
  const handleDragLeave = () => setDragging(false)

  // ── Query Execution ──
  const handleQuery = async () => {
    if (!question.trim() || querying) return
    const seq = ++querySeqRef.current
    setQuerying(true)
    setResult(null)
    setQueryError(null)
    setShowAllEvidence(false)

    try {
      const res = await fetch(apiUrl('/query'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question.trim() }),
      })
      const { data, ok: jsonOk } = await readJsonResponse(res)
      if (seq !== querySeqRef.current) return

      if (!jsonOk || !res.ok) {
        if (res.status === 400 && data?.detail?.includes('empty')) {
          setQueryError('Please upload a document before asking a question.')
        } else {
          setQueryError('Something went wrong while generating the answer. Please try again.')
        }
      } else {
        setResult(data)
      }
    } catch {
      if (seq !== querySeqRef.current) return
      setQueryError('Something went wrong while generating the answer. Please try again.')
    } finally {
      if (seq === querySeqRef.current) setQuerying(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      if (canQuery) handleQuery()
    }
  }

  const canQuery = !!question.trim() && !querying

  // ── Combine & Filter Evidence (Top 2-3 initially) ──
  const rawDocs = (result?.documents || []).map((d) => ({ ...d, type: 'doc' }))
  const rawWeb = (result?.web_results || []).map((w) => ({ ...w, type: 'web' }))
  const allEvidence = [...rawDocs, ...rawWeb]
  const displayedEvidence = showAllEvidence ? allEvidence : allEvidence.slice(0, 3)

  return (
    <main className="container" role="main">
      {/* ── DocuMind AI Header ── */}
      <header className="app-header">
        <div className="header-inner">
          <span className="header-logo-wrap" aria-hidden="true">
            <DocuMindLogo />
          </span>
          <div className="header-text-group">
            <h1 className="brand-title">
              DocuMind <span className="brand-ai">AI</span>
            </h1>
            <p className="subtitle">Intelligent Document Assistant</p>
          </div>
        </div>
      </header>

      {/* ── Document Dropzone ── */}
      <section className="section-card upload-section" aria-label="Upload document">
        <div
          id="file-dropzone"
          className={`file-dropzone${dragging ? ' dragging' : ''}${ingesting ? ' ingesting' : ''}`}
          onClick={() => !ingesting && fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          role="button"
          tabIndex={0}
          aria-label="Upload document"
          onKeyDown={(e) => e.key === 'Enter' && !ingesting && fileInputRef.current?.click()}
        >
          {ingesting ? (
            <span className="file-name-preview">
              <Spinner /> Analyzing {selectedFile?.name || 'document'}…
            </span>
          ) : selectedFile ? (
            <span className="file-name-preview">
              <IconDoc /> <strong>{selectedFile.name}</strong> <span className="change-hint">(Click to change)</span>
            </span>
          ) : (
            <span>Drop a document here or <strong>browse file</strong></span>
          )}
          {!ingesting && <IconUpload />}
        </div>
        <input
          ref={fileInputRef}
          type="file"
          id="file-input"
          style={{ display: 'none' }}
          accept=".pdf,.docx,.txt,.md,.png,.jpg,.jpeg"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) handleIngestFile(file)
          }}
        />

        {ingestStatus && (
          <div
            className={`status-msg ${ingestStatus.ok ? 'success' : 'error'}`}
            role="status"
            aria-live="polite"
          >
            {ingestStatus.ok ? <IconCheck /> : <IconAlert />}
            {ingestStatus.msg}
          </div>
        )}
      </section>

      {/* ── Question Section ── */}
      <section className="section-card query-section" aria-label="Ask a question">
        <div className="query-row">
          <textarea
            id="question-input"
            className="text-input question-input"
            placeholder="Ask a question about the document… (Ctrl+Enter to submit)"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={querying}
            rows={3}
          />
        </div>
        <div className="query-actions">
          <span className="hint-text">Press Ctrl + Enter to send</span>
          <button
            id="ask-btn"
            className="btn btn-primary"
            onClick={handleQuery}
            disabled={!canQuery}
            aria-busy={querying}
          >
            {querying ? <Spinner /> : <IconSearch />}
            {querying ? 'Analyzing…' : 'Ask'}
          </button>
        </div>
        {queryError && (
          <div className="status-msg error" role="alert">
            <IconAlert /> {queryError}
          </div>
        )}
      </section>

      {/* ── Priority Results ── */}
      {result && (
        <div className="results-container">
          {/* 1. Prominent Answer Section */}
          <section className="section-card answer-card" aria-label="Answer">
            <div className="answer-header">
              <span className="answer-title">Answer</span>
            </div>
            <div className="markdown-content">
              <ReactMarkdown>
                {formatRenderText(result.answer) || "I couldn't find enough information in the available sources to answer this question confidently."}
              </ReactMarkdown>
            </div>
          </section>

          {/* 2. Compact Evidence Section */}
          {allEvidence.length > 0 && (
            <section className="section-card evidence-section" aria-label="Evidence">
              <div className="evidence-title-row">
                <span className="section-heading">Evidence</span>
                <span className="evidence-count-badge">
                  {allEvidence.length} {allEvidence.length === 1 ? 'source' : 'sources'}
                </span>
              </div>

              <ul className="evidence-list" aria-label="Relevant evidence snippets">
                {displayedEvidence.map((item, i) => (
                  <EvidenceItem key={i} item={item} />
                ))}
              </ul>

              {allEvidence.length > 3 && (
                <button
                  type="button"
                  className="btn-toggle-evidence"
                  onClick={() => setShowAllEvidence((prev) => !prev)}
                >
                  {showAllEvidence
                    ? 'Show fewer evidence snippets'
                    : `Show all evidence (+${allEvidence.length - 3} more)`}
                </button>
              )}
            </section>
          )}

          {/* 3. Collapsed Technical Details */}
          <details className="tech-details">
            <summary className="tech-summary">▸ Answer details</summary>
            <div className="tech-details-body">
              <div className="tech-grid">
                <div className="tech-item">
                  <span className="tech-label">Retrieval mode:</span>
                  <span className="tech-val">{result.retrieval_mode || 'Hybrid RRF'}</span>
                </div>
                <div className="tech-item">
                  <span className="tech-label">Sources retrieved:</span>
                  <span className="tech-val">{result.documents?.length ?? 0}</span>
                </div>
                <div className="tech-item">
                  <span className="tech-label">Web search:</span>
                  <span className="tech-val">{result.web_fallback ? 'Used for supplemental context' : 'Not required'}</span>
                </div>
                {result.evaluation?.sufficient !== undefined && (
                  <div className="tech-item">
                    <span className="tech-label">Context sufficiency:</span>
                    <span className="tech-val">{result.evaluation.sufficient ? 'Sufficient' : 'Insufficient (supplemented)'}</span>
                  </div>
                )}
              </div>
            </div>
          </details>
        </div>
      )}
    </main>
  )
}
