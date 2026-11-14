// Tests for the parts of Mini DLM that don't need a browser or the model.
// Run with: node test.mjs
import assert from "node:assert/strict";
import { MiniDLM, fillHiddenLimbs, LANDMARKS, SKELETON } from "./mini-dlm.js";

let passed = 0;
const test = (name, fn) => { fn(); passed++; console.log("ok -", name); };

function body(dx = 0, dy = 0, hidden = []) {
  const lm = Array.from({ length: 33 }, () => ({ x: 0.5, y: 1.4, z: 0, visibility: 0.02 }));
  const s = (i, x, y) => { lm[i] = { x: x + dx, y: y + dy, z: 0, visibility: hidden.includes(i) ? 0.05 : 0.95 }; };
  s(0, .5, .17); s(7, .53, .16); s(8, .47, .16); s(11, .58, .3); s(12, .42, .3);
  s(13, .63, .42); s(14, .37, .42); s(15, .65, .53); s(16, .35, .53); s(19, .66, .56); s(20, .34, .56);
  s(23, .55, .55); s(24, .45, .55); s(25, .56, .72); s(26, .44, .72); s(27, .56, .9); s(28, .44, .9); s(31, .6, .93); s(32, .4, .93);
  return lm;
}
const fresh = (opts = {}) => { const t = new MiniDLM({ smoothing: "off", ...opts }); t._source = { paused: false }; return t; };

test("33 named landmarks and a valid skeleton", () => {
  assert.equal(LANDMARKS.length, 33);
  SKELETON.forEach(([a, b]) => assert.ok(a < 33 && b < 33));
});

test("people keep their ids when they move", () => {
  const t = fresh({ fillHiddenLimbs: false });
  t._ingest([body(-.25), body(.25)], 1000);
  const first = t.getPeople(1000).map(p => [p.id, p.center.x < .5]);
  t._ingest([body(.27), body(-.23)], 1033);   // order swapped, both moved a little
  const second = t.getPeople(1033);
  first.forEach(([id, isLeft]) => {
    const p = second.find(q => q.id === id);
    assert.ok(p, "person " + id + " still tracked");
    assert.equal(p.center.x < .5, isLeft, "person " + id + " stayed on the same side");
  });
});

test("a newcomer takes the lowest free slot, ids are never reused", () => {
  const t = fresh();
  t._ingest([body(-.3), body(.3)], 0);
  t._ingest([body(.3)], 1200);                  // left person gone for over a second
  t._ingest([body(.3), body(-.3)], 1233);
  const ps = t.getPeople(1233);
  assert.deepEqual(ps.map(p => p.slot).sort(), [0, 1]);
  assert.ok(Math.max(...ps.map(p => p.id)) === 3);
});

test("smoothing reduces jitter", () => {
  const raw = fresh(), smooth = fresh({ smoothing: "strong" });
  let rawMove = 0, smoothMove = 0, lastR, lastS;
  for (let i = 0; i < 30; i++) {
    const jitter = (i % 2 ? 1 : -1) * 0.01;
    raw._ingest([body(jitter)], i * 33); smooth._ingest([body(jitter)], i * 33);
    const r = raw.getPeople(i * 33)[0].landmarks[0].x, s = smooth.getPeople(i * 33)[0].landmarks[0].x;
    if (i > 5) { rawMove += Math.abs(r - lastR); smoothMove += Math.abs(s - lastS); }
    lastR = r; lastS = s;
  }
  assert.ok(smoothMove < rawMove * 0.5, `smoothed ${smoothMove.toFixed(3)} vs raw ${rawMove.toFixed(3)}`);
});

test("prediction moves poses forward between frames, only while playing", () => {
  const t = fresh();
  t._ingest([body(0)], 1000); t._ingest([body(.02)], 1033);
  const atFrame = t.getPeople(1033)[0].landmarks[0].x;
  const later = t.getPeople(1050)[0].landmarks[0].x;
  assert.ok(later > atFrame, "keeps moving right");
  t._source.paused = true;
  assert.equal(t.getPeople(1050)[0].landmarks[0].x, atFrame, "no prediction when paused");
  t.options.predict = false; t._source.paused = false;
  assert.equal(t.getPeople(1050)[0].landmarks[0].x, atFrame, "no prediction when turned off");
});

test("hidden legs are filled in below the hips", () => {
  const out = fillHiddenLimbs(body(0, 0, [25, 26, 27, 28, 31, 32]), 1);
  [25, 26, 27, 28].forEach(i => { assert.ok(out[i].filled, LANDMARKS[i] + " filled"); assert.ok(out[i].visibility >= .6); });
  assert.ok(out[25].y > out[23].y && out[27].y > out[25].y, "knee below hip, ankle below knee");
  assert.ok(!out[11].filled && !out[23].filled, "visible points untouched");
});

test("upper-body framing gets hips and legs", () => {
  const out = fillHiddenLimbs(body(0, 0, [23, 24, 25, 26, 27, 28, 31, 32]), 16 / 9);
  assert.ok(out[23].filled && out[23].y > out[11].y, "hip under shoulder");
  assert.ok(out[27].y > out[25].y && out[25].y > out[23].y, "legs point down");
  const legL = Math.abs(out[23].x - out[27].x);
  assert.ok(legL < .1, "legs hang straight, not crossed");
});

test("visible bodies are returned unchanged", () => {
  const lm = body();
  const out = fillHiddenLimbs(lm, 1);
  [0, 11, 15, 23, 27].forEach(i => { assert.ok(Math.abs(out[i].x - lm[i].x) < 1e-9 && Math.abs(out[i].y - lm[i].y) < 1e-9); });
});

test("maxPeople is clamped to 1–6", () => {
  assert.equal(new MiniDLM({ maxPeople: 20 }).options.maxPeople, 6);
  assert.equal(new MiniDLM({ maxPeople: 0 }).options.maxPeople, 1);
});

test("track() before create() explains what to do", () => {
  assert.throws(() => new MiniDLM().track({}), /MiniDLM.create/);
});

console.log(`\n${passed} tests passed`);
