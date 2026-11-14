/*!
 * Mini DLM 1.0 — multi-person body tracking for the browser.
 *
 * Built on Google's MediaPipe Pose Landmarker (Lite by default), with:
 *   - One Euro smoothing for low-jitter, low-lag motion
 *   - stable person IDs across frames (up to 6 people)
 *   - motion prediction between video frames
 *   - filling in hidden limbs so partial bodies still give a full skeleton
 *
 * License: Apache-2.0 (same as MediaPipe).
 */

export const VERSION = "1.0.0";

const TV = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MP_ROOT = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/";

export const MODEL_URLS = {
  lite:  MP_ROOT + "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
  full:  MP_ROOT + "pose_landmarker_full/float16/1/pose_landmarker_full.task",
  heavy: MP_ROOT + "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
};

/** Names of the 33 landmarks, in order. */
export const LANDMARKS = [
  "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
  "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
  "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel",
  "left_foot_index", "right_foot_index"
];

/** Pairs of landmark indices that make up the body skeleton. */
export const SKELETON = [
  [11, 12], [11, 23], [12, 24], [23, 24],
  [11, 13], [13, 15], [15, 19], [12, 14], [14, 16], [16, 20],
  [23, 25], [25, 27], [27, 31], [24, 26], [26, 28], [28, 32]
];

export const SMOOTHING = {
  off: null,
  light: { minCutoff: 2.2, beta: 6 },
  strong: { minCutoff: 0.8, beta: 3 }
};

const DEFAULTS = {
  maxPeople: 1,           // 1–6
  model: "lite",          // "lite" | "full" | "heavy", or a URL to a .task file
  smoothing: "light",     // "off" | "light" | "strong" | { minCutoff, beta }
  predict: true,          // extrapolate motion between video frames
  fillHiddenLimbs: true,  // estimate limbs the camera cannot see
  minConfidence: 0.5,
  delegate: "GPU",        // "GPU" falls back to "CPU" automatically
  visionModuleUrl: TV + "/vision_bundle.mjs",
  wasmBase: TV + "/wasm"
};

const L_SH = 11, R_SH = 12, L_HP = 23, R_HP = 24;
const ARMS = [[11, 13, 15, 19], [12, 14, 16, 20]];
const LEGS = [[23, 25, 27, 31], [24, 26, 28, 32]];
const ARM_LEN = [0.62, 0.58, 0.17], LEG_LEN = [0.92, 0.9, 0.26];
const VISIBLE = 0.3;

/* ---------------- One Euro filter ---------------- */
const TAU = Math.PI * 2;
const alpha = (dt, cutoff) => 1 / (1 + 1 / (TAU * cutoff * dt));
class OneEuro {
  constructor(minCutoff, beta) { this.minCutoff = minCutoff; this.beta = beta; this.x = null; }
  filter(x, tMs) {
    if (this.x === null) { this.x = x; this.dx = 0; this.t = tMs; return x; }
    const dt = Math.max((tMs - this.t) / 1000, 1e-3); this.t = tMs;
    this.dx += alpha(dt, 1) * ((x - this.x) / dt - this.dx);
    this.x += alpha(dt, this.minCutoff + this.beta * Math.abs(this.dx)) * (x - this.x);
    return this.x;
  }
}

/* ---------------- helpers ---------------- */
const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: ((a.z || 0) + (b.z || 0)) / 2 });
const seen = p => (p.visibility ?? 1) >= VISIBLE;

function centreOf(lm) {
  if (seen(lm[L_HP]) && seen(lm[R_HP])) return mid(lm[L_HP], lm[R_HP]);
  const s = mid(lm[L_SH], lm[R_SH]); return { x: s.x, y: s.y + 0.15 };
}

/**
 * Estimates hidden limbs. Works in a space where x is scaled by the frame's
 * aspect ratio so that lengths are the same in both directions.
 */
export function fillHiddenLimbs(landmarks, aspect = 1) {
  const P = landmarks.map(p => ({ x: p.x * aspect, y: p.y, z: p.z || 0, visibility: p.visibility ?? 1 }));
  const ok = p => p.visibility >= VISIBLE;
  const mark = (i, p) => { P[i] = { ...p, visibility: Math.max(p.visibility ?? 0, 0.6), filled: true }; };

  let hipsGuessed = !(ok(P[L_HP]) || ok(P[R_HP]));
  if (hipsGuessed) {
    if (!(ok(P[L_SH]) && ok(P[R_SH]))) return landmarks.map(p => ({ ...p }));
    const sw = Math.max(dist(P[L_SH], P[R_SH]), 0.05), drop = sw * 1.35;
    const plausible = (h, s) => { const r = dist(h, s) / drop; return r > 0.6 && r < 1.6 && h.y > s.y; };
    if (plausible(P[L_HP], P[L_SH]) && plausible(P[R_HP], P[R_SH])) { mark(L_HP, P[L_HP]); mark(R_HP, P[R_HP]); hipsGuessed = false; }
    else {
      mark(L_HP, { x: P[L_SH].x + (P[R_SH].x - P[L_SH].x) * 0.12, y: P[L_SH].y + drop, z: P[L_SH].z });
      mark(R_HP, { x: P[R_SH].x + (P[L_SH].x - P[R_SH].x) * 0.12, y: P[R_SH].y + drop, z: P[R_SH].z });
    }
  }
  if (!(ok(P[L_SH]) && ok(P[R_SH]))) return P.map(p => ({ ...p, x: p.x / aspect }));

  const sh = mid(P[L_SH], P[R_SH]), hp = mid(P[L_HP], P[R_HP]);
  const u = Math.max(dist(sh, hp), 0.02);
  const ul = dist(sh, hp) || 1;
  const down = { x: (hp.x - sh.x) / ul, y: (hp.y - sh.y) / ul };

  const chain = (ids, lens, isLeg, trustGuess) => {
    if (!ok(P[ids[0]])) return;
    for (let i = 1; i < ids.length; i++) {
      const prev = P[ids[i - 1]], p = P[ids[i]];
      if (ok(p)) continue;
      const L = lens[i - 1] * u, d = dist(prev, p);
      const downish = !isLeg || ((p.x - prev.x) * down.x + (p.y - prev.y) * down.y) / (d || 1) > 0.3;
      if (trustGuess && d > L * 0.55 && d < L * 1.6 && downish) { mark(ids[i], p); continue; }
      let dir = down;
      if (isLeg && i === 1) { const ox = prev.x - hp.x, oy = prev.y - hp.y, ol = Math.hypot(ox, oy) || 1; dir = { x: down.x + ox / ol * 0.12, y: down.y + oy / ol * 0.12 }; }
      if (i >= 2) { const pp = P[ids[i - 2]], dx = prev.x - pp.x, dy = prev.y - pp.y, dd = Math.hypot(dx, dy); if (dd > 1e-4) dir = { x: dx / dd, y: dy / dd }; }
      if (i === 3) dir = { x: dir.x * 0.3 + (prev.x >= P[ids[0]].x ? 0.9 : -0.9) * 0.7, y: dir.y * 0.3 + 0.2 };
      const n = Math.hypot(dir.x, dir.y) || 1;
      mark(ids[i], { x: prev.x + dir.x / n * L, y: prev.y + dir.y / n * L, z: prev.z });
    }
  };
  ARMS.forEach(ids => chain(ids, ARM_LEN, false, true));
  LEGS.forEach(ids => chain(ids, LEG_LEN, true, !hipsGuessed));
  return P.map(p => ({ ...p, x: p.x / aspect }));
}

/* ---------------- the tracker ---------------- */
export class MiniDLM {
  /**
   * Loads the model and returns a ready tracker.
   * @param {object} [options] see DEFAULTS above
   */
  static async create(options = {}) {
    const t = new MiniDLM(options);
    await t._load();
    return t;
  }

  constructor(options = {}) {
    this.options = { ...DEFAULTS, ...options };
    this.options.maxPeople = Math.max(1, Math.min(6, this.options.maxPeople | 0));
    this.backend = null;
    this.stats = { fps: 0, inferenceMs: 0, backend: null };
    this._detector = null;
    this._tracks = [];
    this._nextId = 1;
    this._lastTs = 0;
    this._lastFrameTime = -1;
    this._lastDetection = 0;
    this._source = null;
    this._aspect = 1;
    this._statCount = 0; this._statMs = 0; this._statT = now();
  }

  async _load() {
    const o = this.options;
    if (!this._vision) {
      const m = await import(o.visionModuleUrl);
      this._PoseLandmarker = m.PoseLandmarker;
      this._vision = await m.FilesetResolver.forVisionTasks(o.wasmBase);
    }
    const modelAssetPath = MODEL_URLS[o.model] || o.model;
    const make = delegate => this._PoseLandmarker.createFromOptions(this._vision, {
      baseOptions: { modelAssetPath, delegate }, runningMode: "VIDEO", numPoses: o.maxPeople,
      minPoseDetectionConfidence: o.minConfidence, minPosePresenceConfidence: o.minConfidence, minTrackingConfidence: o.minConfidence
    });
    let det, backend = o.delegate;
    try { det = await make(o.delegate); }
    catch (e) { if (o.delegate === "CPU") throw e; det = await make("CPU"); backend = "CPU"; }
    if (this._detector) this._detector.close();
    this._detector = det;
    this.backend = this.stats.backend = backend;
    this.reset();
  }

  /** Changes options. Changing maxPeople, model or delegate reloads the model. */
  async setOptions(partial) {
    const before = this.options;
    this.options = { ...before, ...partial };
    this.options.maxPeople = Math.max(1, Math.min(6, this.options.maxPeople | 0));
    if ("smoothing" in partial) this._tracks.forEach(t => { t.filters = null; });
    if (["maxPeople", "model", "delegate", "minConfidence"].some(k => k in partial && partial[k] !== before[k])) await this._load();
  }

  /** Forgets everyone being tracked, e.g. after loading a new video. */
  reset() { this._tracks = []; this._lastFrameTime = -1; }

  /**
   * Runs the model on the current frame of a video, image or canvas.
   * For videos, a frame that was already processed is skipped.
   * @returns {Person[]} the people found, smoothed and with stable ids
   */
  track(source, timestamp = now()) {
    if (!this._detector) throw new Error("Mini DLM is not loaded. Use MiniDLM.create().");
    const isVideo = typeof HTMLVideoElement !== "undefined" && source instanceof HTMLVideoElement;
    if (isVideo) {
      if (source.readyState < 2 || !source.videoWidth) return this.getPeople(timestamp);
      if (source.currentTime === this._lastFrameTime) return this.getPeople(timestamp);
      if (this._source === source && source.currentTime < this._lastFrameTime) this._tracks.forEach(t => { t.filters = null; t.prev = null; });
      this._lastFrameTime = source.currentTime;
    }
    this._source = source;
    const w = source.videoWidth || source.naturalWidth || source.width, h = source.videoHeight || source.naturalHeight || source.height;
    if (w && h) this._aspect = w / h;
    let ts = timestamp; if (ts <= this._lastTs) ts = this._lastTs + 1; this._lastTs = ts;
    const t0 = now();
    const result = this._detector.detectForVideo(source, ts);
    this._recordStats(now() - t0);
    const poses = (result.landmarks || []).map(a => a.map(p => ({ x: p.x, y: p.y, z: p.z, visibility: p.visibility ?? 1 })));
    this._ingest(poses, ts);
    return this.getPeople(ts);
  }

  /**
   * Returns the people currently tracked. Call it on every screen refresh:
   * with prediction on, poses keep moving smoothly between video frames.
   */
  getPeople(time = now()) {
    const live = this._tracks.filter(t => this._lastDetection - t.seen < 300);
    return live.map(t => this._person(t, this._predicted(t, time)));
  }

  /**
   * Tracks a video continuously and calls onFrame(people, stats) on every
   * screen refresh. Returns a function that stops it.
   */
  start(video, onFrame) {
    let running = true;
    const tick = time => {
      if (!running) return;
      let people;
      try { people = this.track(video, time); } catch (e) { running = false; throw e; }
      if (onFrame) onFrame(people, this.stats);
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    return () => { running = false; };
  }

  /** Frees the model. */
  close() { if (this._detector) this._detector.close(); this._detector = null; this._tracks = []; }

  /* ---------- internals (exposed for testing) ---------- */
  _ingest(poses, ts) {
    this._lastDetection = ts;
    const dets = poses.map(lm => ({ lm, c: centreOf(lm) }));
    const pairs = [];
    this._tracks.forEach((t, ti) => dets.forEach((d, di) => pairs.push([Math.hypot(t.c.x - d.c.x, t.c.y - d.c.y), ti, di])));
    pairs.sort((a, b) => a[0] - b[0]);
    const usedT = new Set(), usedD = new Set();
    for (const [dd, ti, di] of pairs) {
      if (dd > 0.3 || usedT.has(ti) || usedD.has(di)) continue;
      usedT.add(ti); usedD.add(di); this._feed(this._tracks[ti], dets[di], ts);
    }
    dets.forEach((d, di) => {
      if (usedD.has(di)) return;
      const used = new Set(this._tracks.map(t => t.slot)); let slot = 0; while (used.has(slot)) slot++;
      const t = { id: this._nextId++, slot, filters: null };
      this._tracks.push(t); this._feed(t, d, ts);
    });
    this._tracks = this._tracks.filter(t => ts - t.seen < 1000);
  }

  _feed(t, d, ts) {
    const s = this.options.smoothing, cfg = typeof s === "object" && s ? s : SMOOTHING[s];
    let lm = d.lm;
    if (cfg) {
      if (!t.filters) t.filters = lm.map(() => [0, 1, 2].map(() => new OneEuro(cfg.minCutoff, cfg.beta)));
      lm = lm.map((p, i) => ({ x: t.filters[i][0].filter(p.x, ts), y: t.filters[i][1].filter(p.y, ts), z: t.filters[i][2].filter(p.z || 0, ts), visibility: p.visibility }));
    }
    t.prev = t.lm; t.prevT = t.t; t.lm = lm; t.t = ts; t.seen = ts; t.c = d.c;
  }

  _predicted(t, time) {
    const src = this._source;
    const playing = src && typeof src.paused === "boolean" ? !src.paused : false;
    if (!this.options.predict || !playing || !t.prev) return t.lm;
    const span = t.t - t.prevT; if (!(span > 5 && span < 200)) return t.lm;
    const k = Math.min(Math.max(time - t.t, 0), 50) / span; if (k <= 0) return t.lm;
    return t.lm.map((p, i) => { const q = t.prev[i]; return { x: p.x + (p.x - q.x) * k, y: p.y + (p.y - q.y) * k, z: p.z + (p.z - q.z) * k, visibility: p.visibility }; });
  }

  _person(t, lm) {
    const landmarks = this.options.fillHiddenLimbs ? fillHiddenLimbs(lm, this._aspect) : lm.map(p => ({ ...p }));
    const keypoints = {};
    landmarks.forEach((p, i) => { keypoints[LANDMARKS[i]] = p; });
    const core = [11, 12, 23, 24].map(i => lm[i].visibility ?? 1);
    return { id: t.id, slot: t.slot, landmarks, keypoints, center: t.c, confidence: core.reduce((a, b) => a + b, 0) / core.length };
  }

  _recordStats(ms) {
    this._statCount++; this._statMs += ms;
    const t = now();
    if (t - this._statT >= 1000) {
      this.stats.fps = Math.round(this._statCount * 1000 / (t - this._statT));
      this.stats.inferenceMs = Math.round(this._statMs / this._statCount);
      this._statCount = 0; this._statMs = 0; this._statT = t;
    }
  }
}

/**
 * Draws a person's skeleton on a canvas. Landmarks are normalized (0–1), so
 * pass the size of the area the video is shown in.
 */
export function drawSkeleton(ctx, person, { width = ctx.canvas.width, height = ctx.canvas.height, color = "#ff8a3d", lineWidth = 4, jointColor = "#ffffff", showFilled = true } = {}) {
  const L = person.landmarks, show = p => (p.visibility ?? 1) >= VISIBLE && (showFilled || !p.filled);
  ctx.save();
  ctx.lineCap = "round"; ctx.lineWidth = lineWidth; ctx.strokeStyle = color;
  ctx.beginPath();
  SKELETON.forEach(([a, b]) => { if (!show(L[a]) || !show(L[b])) return; ctx.moveTo(L[a].x * width, L[a].y * height); ctx.lineTo(L[b].x * width, L[b].y * height); });
  ctx.stroke();
  ctx.fillStyle = jointColor;
  L.forEach((p, i) => { if (i < 11 || !show(p)) return; ctx.globalAlpha = p.filled ? 0.5 : 1; ctx.beginPath(); ctx.arc(p.x * width, p.y * height, lineWidth * 0.9, 0, TAU); ctx.fill(); });
  ctx.restore();
}

export default MiniDLM;
