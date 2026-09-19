import { useState, useRef, useCallback } from 'react'
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

function formatApiDetail(detail, fallback) {
  if (detail == null || detail === '') return fallback
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail.map((item) => item?.msg ?? JSON.stringify(item)).join('; ')
  }
  if (typeof detail === 'object') return JSON.stringify(detail)
  return String(detail)
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
      parseError: res.ok
        ? 'Invalid JSON in response.'
        : `Request failed (${res.status}). ${text.slice(0, 160)}`,
    }
  }
}

// ── Icons ────────────────────────────────────────────────────────────────────
const IconUpload = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
    <polyline points="17 8 12 3 7 8"/>
    <line x1="12" y1="3" x2="12" y2="15"/>
  </svg>
)
const IconLink = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>
    <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>
  </svg>
)
const IconSearch = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
  </svg>
)
const IconCheck = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12"/>
  </svg>
)
const IconX = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
  </svg>
)
const IconGlobe = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"/>
    <line x1="2" y1="12" x2="22" y2="12"/>
    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
  </svg>
)
const IconDoc = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
    <polyline points="14 2 14 8 20 8"/>
  </svg>
)
const IconImage = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
    <circle cx="8.5" cy="8.5" r="1.5"/>
    <polyline points="21 15 16 10 5 21"/>
  </svg>
)
const IconTable = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2"/>
    <line x1="3" y1="9" x2="21" y2="9"/>
    <line x1="3" y1="15" x2="21" y2="15"/>
    <line x1="9" y1="3" x2="9" y2="21"/>
    <line x1="15" y1="3" x2="15" y2="21"/>
  </svg>
)
const IconBolt = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
  </svg>
)

// ── Spinner ──────────────────────────────────────────────────────────────────
const Spinner = () => <span className="spinner" aria-label="Loading" />

// ── Badge ────────────────────────────────────────────────────────────────────
const ContentBadge = ({ type }) => {
  const map = {
    image: { cls: 'badge-image', icon: <IconImage />, label: 'Image' },
    table: { cls: 'badge-table', icon: <IconTable />, label: 'Table' },
    text:  { cls: 'badge-text',  icon: <IconDoc />,   label: 'Text'  },
  }
  const info = map[type] || map.text
  return (
    <span className={`badge ${info.cls}`} aria-label={`Content type: ${info.label}`}>
      {info.icon} {info.label}
    </span>
  )
}

// ── Source Card ──────────────────────────────────────────────────────────────
const SourceCard = ({ doc, index }) => (
  <li className="source-item">
    <div className="source-header">
      <span className="source-title">
        <span className="source-index">#{index + 1}</span>
        {doc.source}
        {doc.page && <span className="source-page">p.{doc.page}</span>}
      </span>
      <div className="source-badges">
        <ContentBadge type={doc.content_type} />
        {doc.score > 0 && (
          <span className="score-badge" aria-label={`Relevance score: ${(doc.score * 100).toFixed(1)}`}>
            {(doc.score * 100).toFixed(1)}
          </span>
        )}
      </div>
    </div>
    {doc.content_type === 'image' && doc.image_url ? (
      <img
        src={resolveApiUrl(doc.image_url)}
        alt={`Extracted image from ${doc.source}`}
        className="source-image-preview"
      />
    ) : doc.content ? (
      <div className="source-preview-snippet">
        {doc.content.slice(0, 300)}{doc.content.length > 300 ? '…' : ''}
      </div>
    ) : null}
  </li>
)

// ── Web Result ───────────────────────────────────────────────────────────────
const WebResultItem = ({ result, index }) => {
  const body = result.content || result.snippet
  return (
    <div className="web-result-item">
      <span className="web-result-num">#{index + 1}</span>
      {result.url ? (
        <a href={result.url} target="_blank" rel="noopener noreferrer" className="web-result-link">
          {result.title || result.url}
        </a>
      ) : (
        <span>{result.title || body || JSON.stringify(result)}</span>
      )}
      {body && <p className="web-result-snippet">{body}</p>}
    </div>
  )
}

// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  // Ingest state
  const [ingestMode, setIngestMode] = useState('file')
  const [selectedFile, setSelectedFile] = useState(null)
  const [urlInput, setUrlInput] = useState('')
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

  // Drag-and-drop
  const [dragging, setDragging] = useState(false)

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) setSelectedFile(file)
  }, [])

  const handleDragOver = (e) => { e.preventDefault(); setDragging(true) }
  const handleDragLeave = () => setDragging(false)

  // ── Ingest ──
  const handleIngest = async () => {
    if (ingesting) return
    if (ingestMode === 'file' && !selectedFile) return
    if (ingestMode === 'url' && !urlInput.trim()) return

    const seq = ++ingestSeqRef.current
    setIngesting(true)
    setIngestStatus(null)
    try {
      const formData = new FormData()
      if (ingestMode === 'file') {
        formData.append('file', selectedFile)
      } else {
        formData.append('url', urlInput.trim())
      }

      const res = await fetch(apiUrl('/ingest'), { method: 'POST', body: formData })
      const { data, ok: jsonOk, parseError } = await readJsonResponse(res)
      if (seq !== ingestSeqRef.current) return

      if (!jsonOk) {
        setIngestStatus({ ok: false, msg: parseError })
        return
      }

      if (!res.ok) {
        setIngestStatus({
          ok: false,
          msg: formatApiDetail(data?.detail, 'Ingestion failed.'),
        })
      } else {
        setIngestStatus({
          ok: true,
          msg: `${data.message || 'Ingestion successful.'} ${data.chunks_count ? `(${data.chunks_count} chunks)` : ''}`,
        })
      }
    } catch (err) {
      if (seq !== ingestSeqRef.current) return
      setIngestStatus({ ok: false, msg: `Network error: ${err.message}` })
    } finally {
      if (seq === ingestSeqRef.current) setIngesting(false)
    }
  }

  // ── Query ──
  const handleQuery = async () => {
    if (!question.trim()) return
    const seq = ++querySeqRef.current
    setQuerying(true)
    setResult(null)
    setQueryError(null)
    try {
      const res = await fetch(apiUrl('/query'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question.trim() }),
      })
      const { data, ok: jsonOk, parseError } = await readJsonResponse(res)
      if (seq !== querySeqRef.current) return

      if (!jsonOk) {
        setQueryError(parseError)
        return
      }

      if (!res.ok) {
        setQueryError(formatApiDetail(data?.detail, 'Query failed.'))
      } else {
        setResult(data)
      }
    } catch (err) {
      if (seq !== querySeqRef.current) return
      setQueryError(`Network error: ${err.message}`)
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

  const canIngest = ingestMode === 'file' ? !!selectedFile : !!urlInput.trim()
  const canQuery = !!question.trim()

  return (
    <main className="container" role="main">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-glow" aria-hidden="true" />
        <div className="header-inner">
          <span className="header-icon" aria-hidden="true"><IconBolt /></span>
          <div>
            <h1>MCP RAG Assistant</h1>
            <p className="subtitle">Multimodal · Hybrid Retrieval · MCP Tool Calling</p>
          </div>
        </div>
      </header>

      {/* ── Ingest Section ── */}
      <section className="section-card" aria-labelledby="ingest-title">
        <div className="section-title">
          <span id="ingest-title">Document</span>
          <div className="mode-toggle" role="group" aria-label="Ingest mode">
            <button
              id="toggle-file"
              className={`toggle-btn${ingestMode === 'file' ? ' active' : ''}`}
              onClick={() => setIngestMode('file')}
              aria-pressed={ingestMode === 'file'}
            >
              <IconUpload /> File
            </button>
            <button
              id="toggle-url"
              className={`toggle-btn${ingestMode === 'url' ? ' active' : ''}`}
              onClick={() => setIngestMode('url')}
              aria-pressed={ingestMode === 'url'}
            >
              <IconLink /> URL
            </button>
          </div>
        </div>

        <div className="input-row">
          {ingestMode === 'file' ? (
            <>
              <div
                id="file-dropzone"
                className={`file-dropzone${dragging ? ' dragging' : ''}`}
                onClick={() => fileInputRef.current?.click()}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                role="button"
                tabIndex={0}
                aria-label="Click or drop a file to upload"
                onKeyDown={(e) => e.key === 'Enter' && fileInputRef.current?.click()}
              >
                {selectedFile ? (
                  <span className="file-name-preview">{selectedFile.name}</span>
                ) : (
                  <span>Drop a file here or <strong>click to browse</strong></span>
                )}
                <IconUpload />
              </div>
              <input
                ref={fileInputRef}
                type="file"
                id="file-input"
                style={{ display: 'none' }}
                accept=".pdf,.docx,.txt,.md,.png,.jpg,.jpeg"
                onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
              />
            </>
          ) : (
            <input
              id="url-input"
              type="url"
              className="text-input"
              placeholder="https://example.com/document.pdf"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault()
                  if (!ingesting && canIngest) handleIngest()
                }
              }}
            />
          )}
          <button
            id="ingest-btn"
            className="btn btn-primary"
            onClick={handleIngest}
            disabled={ingesting || !canIngest}
            aria-busy={ingesting}
          >
            {ingesting ? <Spinner /> : <IconUpload />}
            {ingesting ? 'Ingesting…' : 'Ingest'}
          </button>
        </div>

        {ingestStatus && (
          <div
            className={`status-msg ${ingestStatus.ok ? 'success' : 'error'}`}
            role="status"
            aria-live="polite"
          >
            {ingestStatus.ok ? <IconCheck /> : <IconX />}
            {ingestStatus.msg}
          </div>
        )}
      </section>

      {/* ── Query Section ── */}
      <section className="section-card" aria-labelledby="query-title">
        <label id="query-title" className="section-title" htmlFor="question-input">
          Ask a Question
        </label>
        <div className="query-row">
          <textarea
            id="question-input"
            className="text-input question-input"
            placeholder="What does the document explain? (Ctrl+Enter to submit)"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={3}
          />
        </div>
        <div className="query-actions">
          <span className="hint-text">Ctrl + Enter to submit</span>
          <button
            id="ask-btn"
            className="btn btn-blue"
            onClick={handleQuery}
            disabled={querying || !canQuery}
            aria-busy={querying}
          >
            {querying ? <Spinner /> : <IconSearch />}
            {querying ? 'Thinking…' : 'Ask'}
          </button>
        </div>
        {queryError && (
          <div className="status-msg error" role="alert">
            <IconX /> {queryError}
          </div>
        )}
      </section>

      {/* ── Results ── */}
      {result && (
        <>
          {/* Answer */}
          <section className="section-card answer-section" aria-labelledby="answer-title">
            <div className="section-title" id="answer-title">Answer</div>
            <div className="answer-box" aria-live="polite">
              {result.answer}
            </div>
          </section>

          {/* Sources */}
          {result.documents?.length > 0 && (
            <section className="section-card" aria-labelledby="sources-title">
              <div className="section-title" id="sources-title">
                Sources
                <span className="count-badge">{result.documents.length}</span>
              </div>
              <ul className="sources-list" aria-label="Retrieved sources">
                {result.documents.map((doc, i) => (
                  <SourceCard key={i} doc={doc} index={i} />
                ))}
              </ul>
            </section>
          )}

          {/* Web Fallback */}
          {result.web_fallback && result.web_results?.length > 0 && (
            <section className="section-card web-section" aria-labelledby="web-title">
              <div className="section-title" id="web-title">
                <span className="web-label"><IconGlobe /> Web Fallback</span>
                <span className="count-badge">{result.web_results.length}</span>
              </div>
              <div className="web-results-list">
                {result.web_results.map((r, i) => (
                  <WebResultItem key={i} result={r} index={i} />
                ))}
              </div>
            </section>
          )}

          {/* Metadata Footer */}
          <div className="metadata-footer" role="contentinfo" aria-label="Query metadata">
            <span className="metadata-item">
              Retrieval
              <span className="metadata-value">{result.retrieval_mode || 'Hybrid RRF'}</span>
            </span>
            <span className="metadata-item">
              Sources
              <span className="metadata-value">{result.documents?.length ?? 0}</span>
            </span>
            <span className="metadata-item">
              Web Fallback
              <span className={`metadata-value ${result.web_fallback ? 'fallback-yes' : 'fallback-no'}`}>
                {result.web_fallback ? 'Yes' : 'No'}
              </span>
            </span>
          </div>
        </>
      )}
    </main>
  )
}
