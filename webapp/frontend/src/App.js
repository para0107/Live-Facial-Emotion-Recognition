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

function useWebcam() {
  const videoRef = useRef(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let stream;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          // DOWNGRADED: Changed from 1280/720 to 640/480 for faster processing
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
          audio: false,
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
      if (now - lastSentRef.current < 80) return;
      lastSentRef.current = now;
      const video = videoRef.current;
      if (!video || video.readyState < 2) return;
      const W = video.videoWidth;
      const H = video.videoHeight;
      if (!W || !H) return;
      canvas.width = W;
      canvas.height = H;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.drawImage(video, 0, 0);

      // OPTIMIZED: Changed JPEG compression from 0.92 to 0.6 to reduce payload size
      send({ frame: canvas.toDataURL("image/jpeg", 0.6) });
    };

    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
  }, [ready, connected, send, videoRef]);
}

function FaceOverlay({ faces, videoWidth, videoHeight }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    faces.forEach(({ box, emotion, confidence, is_uncertain, secondary_emotion, secondary_confidence }) => {
      const meta = EMOTION_META[emotion] || EMOTION_META.neutral;
      const { x: rawX, y, w, h } = box;
      const x = canvas.width - rawX - w;

      // Make uncertain boxes look slightly different if you want (e.g., using yellow like the local webcam script)
      const renderColor = is_uncertain ? "#c8c832" : meta.color;
      const renderGlow = is_uncertain ? "#c8c83240" : meta.glow;

      ctx.shadowColor = renderColor;
      ctx.shadowBlur = is_uncertain ? 8 : 16;
      ctx.strokeStyle = renderColor;
      ctx.lineWidth = is_uncertain ? 1 : 2;
      ctx.strokeRect(x, y, w, h);
      ctx.shadowBlur = 0;

      const cLen = 14;
      ctx.lineWidth = is_uncertain ? 2 : 3;
      [
        [x, y, x + cLen, y, x, y + cLen],
        [x + w - cLen, y, x + w, y, x + w, y + cLen],
        [x, y + h - cLen, x, y + h, x + cLen, y + h],
        [x + w - cLen, y + h, x + w, y + h, x + w, y + h - cLen],
      ].forEach(([x1, y1, x2, y2, x3, y3]) => {
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
        ctx.moveTo(x1, y1); ctx.lineTo(x3, y3); ctx.stroke();
      });

      // Show dual label if uncertain, just like the local script
      let label = `${meta.emoji} ${meta.label} ${(confidence * 100).toFixed(0)}%`;
      if (is_uncertain && secondary_emotion) {
         const secMeta = EMOTION_META[secondary_emotion] || EMOTION_META.neutral;
         label = `${meta.label}/${secMeta.label} ${(confidence * 100).toFixed(0)}%/${(secondary_confidence * 100).toFixed(0)}%`;
      }

      ctx.font = "bold 12px monospace";
      const tw = ctx.measureText(label).width;
      const pillX = x;
      const pillY = y > 30 ? y - 28 : y + h + 4;
      ctx.fillStyle = renderColor + "dd";
      ctx.beginPath();
      ctx.roundRect(pillX, pillY, tw + 16, 22, 4);
      ctx.fill();
      ctx.fillStyle = "#000";
      ctx.fillText(label, pillX + 8, pillY + 15);
    });
  }, [faces]);

  return (
    <canvas ref={canvasRef} width={videoWidth} height={videoHeight}
      style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}
    />
  );
}

function EmotionBar({ emotion, value, isTop }) {
  const meta = EMOTION_META[emotion];
  const pct = (value * 100).toFixed(1);
  return (
    <div style={{ marginBottom: 5 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
        <span style={{ fontFamily: "monospace", fontSize: 10, color: isTop ? meta.color : "#64748b", fontWeight: isTop ? 700 : 400 }}>
          {meta.emoji} {meta.label}
        </span>
        <span style={{ fontFamily: "monospace", fontSize: 10, color: isTop ? meta.color : "#475569" }}>
          {pct}%
        </span>
      </div>
      <div style={{ height: 4, background: "#1e293b", borderRadius: 2, overflow: "hidden" }}>
        <div style={{
          height: "100%", width: `${pct}%`,
          background: isTop ? `linear-gradient(90deg, ${meta.color}99, ${meta.color})` : "#334155",
          borderRadius: 2, transition: "width 0.15s ease",
          boxShadow: isTop ? `0 0 6px ${meta.color}80` : "none",
        }} />
      </div>
    </div>
  );
}

function StatsPanel({ faces, connected, fps, mobile }) {
  const primaryFace = faces[0] || null;
  const emotion = primaryFace?.emotion || null;
  const meta = emotion ? EMOTION_META[emotion] : null;
  const probs = primaryFace?.probs || {};

  return (
    <div style={{
      display: "flex", flexDirection: "column", gap: 10,
      width: mobile ? "100%" : 256, flexShrink: 0,
    }}>
      {/* Status bar */}
      <div style={{
        background: "#0f172a", border: "1px solid #1e293b", borderRadius: 10,
        padding: "9px 14px", display: "flex", alignItems: "center", gap: 8,
      }}>
        <div style={{
          width: 7, height: 7, borderRadius: "50%", flexShrink: 0,
          background: connected ? "#22c55e" : "#ef4444",
          boxShadow: connected ? "0 0 6px #22c55e" : "0 0 6px #ef4444",
        }} />
        <span style={{ fontFamily: "monospace", fontSize: 10, color: connected ? "#22c55e" : "#ef4444", letterSpacing: "0.06em" }}>
          {connected ? "CONNECTED" : "OFFLINE"}
        </span>
        <span style={{ marginLeft: "auto", fontFamily: "monospace", fontSize: 10, color: "#334155" }}>
          {fps} FPS
        </span>
      </div>

      {/* Primary emotion */}
      <div style={{
        background: "#0f172a",
        border: `1px solid ${meta ? meta.color + "40" : "#1e293b"}`,
        borderRadius: 10, padding: "12px 14px", textAlign: "center",
        display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
        minHeight: mobile ? 80 : 105,
        boxShadow: meta ? `0 0 24px ${meta.glow}` : "none",
        transition: "border-color 0.3s, box-shadow 0.3s",
      }}>
        {emotion ? (
          <>
            <div style={{ fontSize: mobile ? 30 : 38, lineHeight: 1, marginBottom: 5 }}>{meta.emoji}</div>
            <div style={{
              fontFamily: "monospace", fontSize: mobile ? 15 : 19, fontWeight: 700,
              color: meta.color, letterSpacing: "0.1em",
              textShadow: `0 0 14px ${meta.color}`,
            }}>{meta.label}</div>
            <div style={{ fontFamily: "monospace", fontSize: 10, color: "#475569", marginTop: 3 }}>
              {(primaryFace.confidence * 100).toFixed(1)}% confidence
            </div>
          </>
        ) : (
          <div style={{ fontFamily: "monospace", fontSize: 10, color: "#334155", letterSpacing: "0.06em" }}>
            {connected ? "NO FACE DETECTED" : "WAITING FOR SERVER"}
          </div>
        )}
      </div>

      {/* Probability distribution */}
      <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 10, padding: "11px 13px" }}>
        <div style={{ fontFamily: "monospace", fontSize: 9, color: "#334155", letterSpacing: "0.1em", marginBottom: 9 }}>
          PROBABILITY DISTRIBUTION
        </div>
        {EMOTIONS.map(em => (
          <EmotionBar key={em} emotion={em} value={probs[em] || 0} isTop={em === emotion} />
        ))}
      </div>

      {/* Faces + model info */}
      <div style={{ display: "flex", gap: 10 }}>
        <div style={{
          background: "#0f172a", border: "1px solid #1e293b", borderRadius: 10,
          padding: "9px 12px", display: "flex", flexDirection: "column",
          alignItems: "center", justifyContent: "center", minWidth: 58,
        }}>
          <span style={{ fontFamily: "monospace", fontSize: 9, color: "#334155" }}>FACES</span>
          <span style={{
            fontFamily: "monospace", fontSize: 20, fontWeight: 700,
            color: faces.length > 0 ? "#e2e8f0" : "#334155", lineHeight: 1.2,
          }}>{faces.length}</span>
        </div>
        <div style={{ flex: 1, background: "#0f172a", border: "1px solid #1e293b", borderRadius: 10, padding: "9px 12px" }}>
          <div style={{ fontFamily: "monospace", fontSize: 9, color: "#1e3a5f", letterSpacing: "0.06em", lineHeight: 1.9 }}>
            ResNet-18 · FER2013<br/>
            Accuracy: 68.95%<br/>
            7 emotion classes
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const { videoRef, ready, error } = useWebcam();
  const [faces, setFaces] = useState([]);
  const [fps, setFps] = useState(0);
  const fpsRef = useRef({ count: 0, last: performance.now() });
  const [videoDims, setVideoDims] = useState({ w: 640, h: 480 });
  const [mobile, setMobile] = useState(window.innerWidth < 768);

  useEffect(() => {
    const onResize = () => setMobile(window.innerWidth < 768);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const handleMessage = useCallback((data) => {
    setFaces(data.faces || []);
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
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        html { -webkit-text-size-adjust: 100%; text-size-adjust: 100%; }
        body { background: #020617; overflow-x: hidden; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        @keyframes scanline { 0%{transform:translateY(-100%)} 100%{transform:translateY(100vh)} }
        .fer-video { 
          width: 100%; height: 100%; object-fit: cover; display: block;
          -webkit-transform: scaleX(-1); transform: scaleX(-1);
        }
      `}</style>

      <div style={{
        minHeight: "100svh", background: "#020617",
        display: "flex", flexDirection: "column", color: "#e2e8f0",
      }}>
        {/* Header */}
        <div style={{
          padding: mobile ? "11px 14px" : "13px 24px",
          borderBottom: "1px solid #0f172a",
          display: "flex", alignItems: "center", gap: 11,
          background: "#020617", flexShrink: 0,
        }}>
          <div style={{
            width: 4, height: mobile ? 26 : 32, flexShrink: 0,
            background: "linear-gradient(180deg, #3b82f6, #8b5cf6)", borderRadius: 2,
          }} />
          <div style={{ minWidth: 0 }}>
            <div style={{
              fontFamily: "'Space Mono', monospace",
              fontSize: mobile ? 10 : 13, fontWeight: 700,
              color: "#f1f5f9", letterSpacing: "0.07em",
              whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
            }}>
              LIVE FACIAL EMOTION RECOGNITION
            </div>
            {!mobile && (
              <div style={{ fontSize: 9, color: "#334155", letterSpacing: "0.1em", marginTop: 2 }}>
                ResNet-18 · Transfer Learning · FER2013 · WebSocket Inference
              </div>
            )}
          </div>
          <div style={{ marginLeft: "auto", display: "flex", gap: 6, flexShrink: 0 }}>
            {["angry", "happy", "surprise"].map(em => (
              <div key={em} style={{
                width: 7, height: 7, borderRadius: "50%",
                background: EMOTION_META[em].color, opacity: 0.7,
              }} />
            ))}
          </div>
        </div>

        {/* Body */}
        <div style={{
          flex: 1,
          display: "flex",
          flexDirection: mobile ? "column" : "row",
          gap: mobile ? 12 : 18,
          padding: mobile ? 12 : 18,
          alignItems: "flex-start",
          overflowY: "auto",
          WebkitOverflowScrolling: "touch",
        }}>
          {/* Video */}
          <div style={{
            position: "relative", borderRadius: 12, overflow: "hidden",
            background: "#0a0f1e", border: "1px solid #1e293b",
            width: "100%",
            aspectRatio: mobile ? "4/3" : "16/9",
            flexShrink: 0,
            ...(mobile ? {} : { flex: 1, maxHeight: "calc(100vh - 120px)" }),
          }}>
            <div style={{
              position: "absolute", top: 0, left: 0, right: 0, height: 2, zIndex: 5,
              background: "linear-gradient(90deg, transparent, #3b82f620, transparent)",
              animation: "scanline 4s linear infinite", pointerEvents: "none",
            }} />
            {["top-left","top-right","bottom-left","bottom-right"].map(pos => (
              <div key={pos} style={{
                position: "absolute", zIndex: 4, width: 14, height: 14, pointerEvents: "none",
                [pos.includes("top") ? "top" : "bottom"]: 9,
                [pos.includes("left") ? "left" : "right"]: 9,
                borderTop: pos.includes("top") ? "2px solid #1e40af50" : "none",
                borderBottom: pos.includes("bottom") ? "2px solid #1e40af50" : "none",
                borderLeft: pos.includes("left") ? "2px solid #1e40af50" : "none",
                borderRight: pos.includes("right") ? "2px solid #1e40af50" : "none",
              }} />
            ))}
            {error ? (
              <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 10 }}>
                <div style={{ fontSize: 32 }}>📷</div>
                <div style={{ fontFamily: "monospace", fontSize: 11, color: "#ef4444", textAlign: "center", padding: "0 20px" }}>{error}</div>
              </div>
            ) : (
              <>
                <video ref={videoRef} autoPlay playsInline muted className="fer-video"
                  style={{ filter: ready ? "none" : "brightness(0)", transition: "filter 0.5s" }}
                />
                {ready && <FaceOverlay faces={faces} videoWidth={videoDims.w} videoHeight={videoDims.h} />}
                {!ready && (
                  <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", background: "#020617" }}>
                    <div style={{ fontFamily: "monospace", fontSize: 10, color: "#1e293b", letterSpacing: "0.15em", animation: "pulse 1.5s infinite" }}>
                      INITIALIZING CAMERA...
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          {/* Stats */}
          <StatsPanel faces={faces} connected={connected} fps={fps} mobile={mobile} />
        </div>
      </div>
    </>
  );
}