import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

const STAGE_GROUPS = [
  {
    title: 'Translation Pipeline',
    caption: 'Speech -> Translation -> Voice',
    stages: [
      'A Whisper FR ASR',
      'A Whisper EN ASR',
      'B Helsinki-NLP FR->EN MT',
      'B Helsinki-NLP EN->FR MT',
      'C MMS-TTS EN synthesis',
      'C MMS-TTS FR synthesis'
    ]
  },
  {
    title: 'Mimi Latent Pipeline',
    caption: 'Latent transforms and decode',
    stages: [
      '[2] Mimi Encoder',
      '[3] Silence Gate',
      '[4] Squeezeformer (passthrough)',
      '[5] Helium TTT',
      '[6] BSM Merge',
      '[7] Depth Transformer',
      '[8] Unmerge + Rescale',
      '[9] Mimi Decoder'
    ]
  }
]

function App() {
  const [connected, setConnected] = useState(false)
  const [payload, setPayload] = useState(null)
  const [history, setHistory] = useState([])
  const timerRef = useRef(null)

  useEffect(() => {
    const host = window.location.hostname || '127.0.0.1'
    const url = `http://${host}:8000/api/latency`

    const poll = async () => {
      try {
        const res = await fetch(url, { cache: 'no-store' })
        if (!res.ok) {
          setConnected(false)
          return
        }
        const data = await res.json()
        setConnected(true)
        if (data?.data) {
          setPayload(data.data)
          setHistory((prev) => [data.data, ...prev].slice(0, 6))
        }
      } catch {
        setConnected(false)
      }
    }

    poll()
    timerRef.current = window.setInterval(poll, 1200)

    return () => {
      if (timerRef.current) {
        window.clearInterval(timerRef.current)
      }
    }
  }, [])

  const stages = payload?.stages ?? {}
  const totalMs = payload?.total_ms ?? 0
  const maxStage = useMemo(() => {
    const values = Object.values(stages)
    return values.length ? Math.max(...values, 1) : 1
  }, [stages])

  const meta = payload?.meta ?? {}
  const transcript = meta.text || '—'
  const translated = meta.translated || '—'

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">Realtime Latency Observatory</p>
          <h1>Pipeline Pulse</h1>
          <p className="subtitle">
            Watch live stage timings for speech, translation, and Mimi transforms.
          </p>
        </div>
        <div className="status">
          <span className={connected ? 'dot live' : 'dot'}></span>
          <div>
            <p className="status-title">API</p>
            <p className="status-sub">
              {connected ? 'Connected' : 'Disconnected'}
            </p>
          </div>
        </div>
      </header>

      <section className="summary">
        <div className="summary-card">
          <p className="label">Last transcript</p>
          <p className="value">{transcript}</p>
        </div>
        <div className="summary-card">
          <p className="label">Last translation</p>
          <p className="value">{translated}</p>
        </div>
        <div className="summary-card highlight">
          <p className="label">Total latency</p>
          <p className="value">{totalMs ? `${totalMs.toFixed(0)} ms` : '—'}</p>
        </div>
      </section>

      {STAGE_GROUPS.map((group) => (
        <section className="panel" key={group.title}>
          <div className="panel-header">
            <div>
              <h2>{group.title}</h2>
              <p>{group.caption}</p>
            </div>
          </div>
          <div className="grid">
            {group.stages.map((stage) => {
              const value = stages[stage]
              const width = value ? Math.max(12, (value / maxStage) * 100) : 12
              return (
                <div className="metric" key={stage}>
                  <div className="metric-row">
                    <span>{stage}</span>
                    <span className="metric-value">
                      {value ? `${value.toFixed(0)} ms` : '—'}
                    </span>
                  </div>
                  <div className="bar">
                    <div className="fill" style={{ width: `${width}%` }}></div>
                  </div>
                </div>
              )
            })}
          </div>
        </section>
      ))}

      <section className="panel history">
        <div className="panel-header">
          <div>
            <h2>Recent frames</h2>
            <p>Most recent latency payloads</p>
          </div>
        </div>
        <div className="history-list">
          {history.length === 0 && <p className="muted">No data yet.</p>}
          {history.map((item, idx) => (
            <div className="history-item" key={`${item.ts}-${idx}`}>
              <p>
                <span className="label">Total</span>
                <strong>{item.total_ms.toFixed(0)} ms</strong>
              </p>
              <p className="muted">
                {new Date(item.ts * 1000).toLocaleTimeString()}
              </p>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}

export default App
