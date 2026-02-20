import React, { useRef, useEffect, useState, useCallback } from "react";

const WS_URL = process.env.REACT_APP_WS_URL || "ws://localhost:8000/ws";

const EMOTION_META = {
  angry:    { color: "#ef4444", glow: "#ef444440", emoji: "😠", label: "ANGRY" },
  disgust:  { color: "#22c55e", glow: "#22c55e40", emoji: "🤢", label: "DISGUST" },
  fear:     { color: "#a855f7", glow: "#a855f740", emoji: "😨", label: "FEAR" },
  happy:    { color: "#eab308", glow: "#eab30840", emoji: "😊", label: "HAPPY" },
  neutral:  { color: "#94a3b8", glow: "#94a3b840", emoji: "😐", label: "NEUTRAL" },
  sad:      { color: "#3b82f6", glow: "#3b82f640", emoji: "😢", label: "SAD" },
  surprise: { color: "#f97316", glow: "#f9731640", emoji: "😲", label: "SURPRISE" },
};

const EMOTIONS = Object.keys(EMOTION_META);

// ── Webcam hook ───────────────────────────────────────────────────────────────
function useWebcam() {
  const videoRef = useRef(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let stream;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user" },
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => setReady(true);
        }
      } catch (e) {
        setError("Camera access denied or unavailable.");
      }
    })();
    return () => { if (stream) stream.getTracks().forEach(t => t.stop()); };
  }, []);

  return { videoRef, ready, error };
}

// ── WebSocket hook ────────────────────────────────────────────────────────────
function useWebSocket(url, onMessage) {
  const wsRef = useRef(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const ws = new WebSocket(url);
    wsRef.current = ws;
    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);
    ws.onmessage = (e) => onMessage(JSON.parse(e.data));
    return () => ws.close();
  }, [url]);

  const send = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  return { send, connected };
}

// ── Frame capture ─────────────────────────────────────────────────────────────
function useFrameCapture(videoRef, ready, send, connected) {
  const canvasRef = useRef(document.createElement("canvas"));
  const animRef = useRef(null);
  const lastSentRef = useRef(0);

  useEffect(() => {
    if (!ready || !connected) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const loop = () => {
      animRef.current = requestAnimationFrame(loop);
      const now = performance.now();
      if (now - lastSentRef.current < 80) return; // ~12fps to server
      lastSentRef.current = now;
      const video = videoRef.current;
      if (!video || video.readyState < 2) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);
      const frame = canvas.toDataURL("image/jpeg", 0.7);
      send({ frame });
    };

    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
  }, [ready, connected, send, videoRef]);
}

// ── Face overlay canvas ───────────────────────────────────────────────────────
function FaceOverlay({ faces, videoWidth, videoHeight }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    faces.forEach(({ box, emotion, confidence }) => {
      const meta = EMOTION_META[emotion] || EMOTION_META.neutral;
      const { x, y, w, h } = box;

      // Glow box
      ctx.shadowColor = meta.color;
      ctx.shadowBlur = 20;
      ctx.strokeStyle = meta.color;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x, y, w, h);
      ctx.shadowBlur = 0;

      // Corner marks
      const cLen = 18;
      ctx.lineWidth = 4;
      [
        [x, y, x + cLen, y, x, y + cLen],
        [x + w - cLen, y, x + w, y, x + w, y + cLen],
        [x, y + h - cLen, x, y + h, x + cLen, y + h],
        [x + w - cLen, y + h, x + w, y + h, x + w, y + h - cLen],
      ].forEach(([x1, y1, x2, y2, x3, y3]) => {
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.moveTo(x1, y1);
        ctx.lineTo(x3, y3);
        ctx.stroke();
      });

      // Label pill
      const label = `${meta.emoji}  ${meta.label}  ${(confidence * 100).toFixed(0)}%`;
      ctx.font = "bold 13px 'JetBrains Mono', monospace";
      const tw = ctx.measureText(label).width;
      const pillX = x;
      const pillY = y > 30 ? y - 32 : y + h + 6;
      const pH = 24, pPad = 10;

      ctx.fillStyle = meta.color + "dd";
      ctx.beginPath();
      ctx.roundRect(pillX, pillY, tw + pPad * 2, pH, 4);
      ctx.fill();

      ctx.fillStyle = "#000";
      ctx.fillText(label, pillX + pPad, pillY + 16);
    });
  }, [faces]);

  return (
    <canvas
      ref={canvasRef}
      width={videoWidth}
      height={videoHeight}
      style={{
        position: "absolute", top: 0, left: 0,
        width: "100%", height: "100%",
        pointerEvents: "none",
      }}
    />
  );
}

// ── Probability bar ───────────────────────────────────────────────────────────
function EmotionBar({ emotion, value, isTop }) {
  const meta = EMOTION_META[emotion];
  const pct = (value * 100).toFixed(1);
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
        <span style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 11,
          color: isTop ? meta.color : "#64748b",
          fontWeight: isTop ? 700 : 400,
          letterSpacing: "0.08em",
        }}>
          {meta.emoji} {meta.label}
        </span>
        <span style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 11,
          color: isTop ? meta.color : "#475569",
        }}>
          {pct}%
        </span>
      </div>
      <div style={{
        height: 4,
        background: "#1e293b",
        borderRadius: 2,
        overflow: "hidden",
      }}>
        <div style={{
          height: "100%",
          width: `${pct}%`,
          background: isTop
            ? `linear-gradient(90deg, ${meta.color}99, ${meta.color})`
            : "#334155",
          borderRadius: 2,
          transition: "width 0.15s ease",
          boxShadow: isTop ? `0 0 8px ${meta.color}80` : "none",
        }} />
      </div>
    </div>
  );
}

// ── Stats panel ───────────────────────────────────────────────────────────────
function StatsPanel({ faces, connected, fps }) {
  const primaryFace = faces[0] || null;
  const emotion = primaryFace?.emotion || null;
  const meta = emotion ? EMOTION_META[emotion] : null;
  const probs = primaryFace?.probs || {};

  const topEmotion = emotion
    ? Object.entries(probs).sort((a, b) => b[1] - a[1])[0]
    : null;

  return (
    <div style={{
      width: 280,
      display: "flex",
      flexDirection: "column",
      gap: 16,
      flexShrink: 0,
    }}>
      {/* Connection status */}
      <div style={{
        background: "#0f172a",
        border: "1px solid #1e293b",
        borderRadius: 12,
        padding: "12px 16px",
        display: "flex",
        alignItems: "center",
        gap: 10,
      }}>
        <div style={{
          width: 8, height: 8,
          borderRadius: "50%",
          background: connected ? "#22c55e" : "#ef4444",
          boxShadow: connected ? "0 0 8px #22c55e" : "0 0 8px #ef4444",
          animation: connected ? "pulse 2s infinite" : "none",
        }} />
        <span style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 11,
          color: connected ? "#22c55e" : "#ef4444",
          letterSpacing: "0.1em",
        }}>
          {connected ? "CONNECTED" : "OFFLINE"}
        </span>
        <span style={{
          marginLeft: "auto",
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 10,
          color: "#334155",
        }}>
          {fps} FPS
        </span>
      </div>

      {/* Primary emotion display */}
      <div style={{
        background: "#0f172a",
        border: `1px solid ${meta ? meta.color + "40" : "#1e293b"}`,
        borderRadius: 12,
        padding: 20,
        textAlign: "center",
        minHeight: 120,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        transition: "border-color 0.3s",
        boxShadow: meta ? `0 0 30px ${meta.glow}` : "none",
      }}>
        {emotion ? (
          <>
            <div style={{ fontSize: 48, marginBottom: 8, lineHeight: 1 }}>
              {meta.emoji}
            </div>
            <div style={{
              fontFamily: "'Space Mono', monospace",
              fontSize: 22,
              fontWeight: 700,
              color: meta.color,
              letterSpacing: "0.15em",
              textShadow: `0 0 20px ${meta.color}`,
            }}>
              {meta.label}
            </div>
            <div style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 12,
              color: "#475569",
              marginTop: 6,
            }}>
              {(primaryFace.confidence * 100).toFixed(1)}% confidence
            </div>
          </>
        ) : (
          <div style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 12,
            color: "#334155",
            letterSpacing: "0.1em",
          }}>
            {connected ? "NO FACE DETECTED" : "WAITING FOR SERVER"}
          </div>
        )}
      </div>

      {/* Probability bars */}
      <div style={{
        background: "#0f172a",
        border: "1px solid #1e293b",
        borderRadius: 12,
        padding: "16px 16px 12px",
      }}>
        <div style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 10,
          color: "#334155",
          letterSpacing: "0.15em",
          marginBottom: 12,
        }}>
          PROBABILITY DISTRIBUTION
        </div>
        {EMOTIONS.map(em => (
          <EmotionBar
            key={em}
            emotion={em}
            value={probs[em] || 0}
            isTop={em === emotion}
          />
        ))}
      </div>

      {/* Face count */}
      <div style={{
        background: "#0f172a",
        border: "1px solid #1e293b",
        borderRadius: 12,
        padding: "12px 16px",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}>
        <span style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 10,
          color: "#334155",
          letterSpacing: "0.1em",
        }}>FACES DETECTED</span>
        <span style={{
          fontFamily: "'Space Mono', monospace",
          fontSize: 20,
          fontWeight: 700,
          color: faces.length > 0 ? "#e2e8f0" : "#334155",
        }}>{faces.length}</span>
      </div>

      {/* Model info */}
      <div style={{
        background: "#0f172a",
        border: "1px solid #1e293b",
        borderRadius: 12,
        padding: "12px 16px",
      }}>
        <div style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 9,
          color: "#1e3a5f",
          letterSpacing: "0.1em",
          lineHeight: 1.8,
        }}>
          MODEL: ResNet-18 (fine-tuned)<br/>
          DATASET: FER2013 · 28,709 imgs<br/>
          ACCURACY: 68.95% test<br/>
          CLASSES: 7 universal emotions<br/>
          SMOOTHING: 8-frame window
        </div>
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const { videoRef, ready, error } = useWebcam();
  const [faces, setFaces] = useState([]);
  const [fps, setFps] = useState(0);
  const fpsRef = useRef({ count: 0, last: performance.now() });
  const [videoDims, setVideoDims] = useState({ w: 640, h: 480 });

  const handleMessage = useCallback((data) => {
    setFaces(data.faces || []);
    // fps counter
    fpsRef.current.count++;
    const now = performance.now();
    if (now - fpsRef.current.last > 1000) {
      setFps(fpsRef.current.count);
      fpsRef.current = { count: 0, last: now };
    }
  }, []);

  const { send, connected } = useWebSocket(WS_URL, handleMessage);
  useFrameCapture(videoRef, ready, send, connected);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const onMeta = () => setVideoDims({ w: v.videoWidth, h: v.videoHeight });
    v.addEventListener("loadedmetadata", onMeta);
    return () => v.removeEventListener("loadedmetadata", onMeta);
  }, [videoRef]);

  return (
    <>
      {/* Google Fonts */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=JetBrains+Mono:wght@400;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #020617; }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
        @keyframes scanline {
          0% { transform: translateY(-100%); }
          100% { transform: translateY(100vh); }
        }
      `}</style>

      <div style={{
        minHeight: "100vh",
        background: "#020617",
        display: "flex",
        flexDirection: "column",
        fontFamily: "'JetBrains Mono', monospace",
        color: "#e2e8f0",
        overflow: "hidden",
      }}>

        {/* Header */}
        <div style={{
          padding: "16px 32px",
          borderBottom: "1px solid #0f172a",
          display: "flex",
          alignItems: "center",
          gap: 16,
          background: "#020617",
          position: "relative",
          zIndex: 10,
        }}>
          <div style={{
            width: 8, height: 32,
            background: "linear-gradient(180deg, #3b82f6, #8b5cf6)",
            borderRadius: 2,
          }} />
          <div>
            <div style={{
              fontFamily: "'Space Mono', monospace",
              fontSize: 16,
              fontWeight: 700,
              color: "#f1f5f9",
              letterSpacing: "0.1em",
            }}>
              LIVE FACIAL EMOTION RECOGNITION
            </div>
            <div style={{
              fontSize: 10,
              color: "#334155",
              letterSpacing: "0.15em",
              marginTop: 2,
            }}>
              ResNet-18 · Transfer Learning · FER2013 · Real-Time WebSocket Inference
            </div>
          </div>
          <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
            {["angry", "happy", "surprise"].map(em => (
              <div key={em} style={{
                width: 8, height: 8,
                borderRadius: "50%",
                background: EMOTION_META[em].color,
                opacity: 0.6,
              }} />
            ))}
          </div>
        </div>

        {/* Main content */}
        <div style={{
          flex: 1,
          display: "flex",
          gap: 24,
          padding: 24,
          alignItems: "flex-start",
        }}>

          {/* Video area */}
          <div style={{
            flex: 1,
            position: "relative",
            borderRadius: 16,
            overflow: "hidden",
            background: "#0a0f1e",
            border: "1px solid #1e293b",
            aspectRatio: "4/3",
            maxHeight: "calc(100vh - 140px)",
          }}>

            {/* Scanline effect */}
            <div style={{
              position: "absolute", top: 0, left: 0, right: 0,
              height: "2px",
              background: "linear-gradient(90deg, transparent, #3b82f620, transparent)",
              zIndex: 5,
              animation: "scanline 4s linear infinite",
              pointerEvents: "none",
            }} />

            {error ? (
              <div style={{
                position: "absolute", inset: 0,
                display: "flex", flexDirection: "column",
                alignItems: "center", justifyContent: "center",
                gap: 12,
              }}>
                <div style={{ fontSize: 40 }}>📷</div>
                <div style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 12, color: "#ef4444",
                  letterSpacing: "0.1em",
                }}>{error}</div>
              </div>
            ) : (
              <>
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  style={{
                    width: "100%", height: "100%",
                    objectFit: "cover",
                    display: "block",
                    transform: "scaleX(-1)", // mirror
                    filter: ready ? "none" : "brightness(0)",
                    transition: "filter 0.5s",
                  }}
                />
                {ready && (
                  <FaceOverlay
                    faces={faces}
                    videoWidth={videoDims.w}
                    videoHeight={videoDims.h}
                  />
                )}

                {/* Corner decorations */}
                {["top-left", "top-right", "bottom-left", "bottom-right"].map(pos => (
                  <div key={pos} style={{
                    position: "absolute",
                    [pos.includes("top") ? "top" : "bottom"]: 12,
                    [pos.includes("left") ? "left" : "right"]: 12,
                    width: 20, height: 20,
                    borderTop: pos.includes("top") ? "2px solid #1e40af60" : "none",
                    borderBottom: pos.includes("bottom") ? "2px solid #1e40af60" : "none",
                    borderLeft: pos.includes("left") ? "2px solid #1e40af60" : "none",
                    borderRight: pos.includes("right") ? "2px solid #1e40af60" : "none",
                    pointerEvents: "none",
                  }} />
                ))}

                {/* Not ready overlay */}
                {!ready && (
                  <div style={{
                    position: "absolute", inset: 0,
                    display: "flex", alignItems: "center",
                    justifyContent: "center",
                    background: "#020617",
                  }}>
                    <div style={{
                      fontFamily: "'JetBrains Mono', monospace",
                      fontSize: 11, color: "#1e293b",
                      letterSpacing: "0.2em",
                      animation: "pulse 1.5s infinite",
                    }}>
                      INITIALIZING CAMERA...
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          {/* Stats panel */}
          <StatsPanel faces={faces} connected={connected} fps={fps} />
        </div>
      </div>
    </>
  );
}
