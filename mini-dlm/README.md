# Mini DLM 1.0

Multi-person body tracking for the browser, in one JavaScript file.

Mini DLM finds up to 6 people in a video, image or canvas and gives you 33 body points for each one. It runs on Google's MediaPipe Pose model and adds four things on top:

- **Smoothing** that removes jitter without making movement lag (One Euro filter)
- **Stable IDs** so each person keeps the same number from frame to frame
- **Motion prediction** between video frames, so motion stays fluid at your screen's refresh rate
- **Hidden-limb filling**: if the camera only sees someone's upper body, you still get a full skeleton

Everything runs on the user's device, on the graphics chip when available. Videos are never uploaded.

**Demo:** [`demo.html`](demo.html)

## Quick start

```html
<video id="video" src="dance.mp4" playsinline muted loop autoplay></video>
<canvas id="canvas"></canvas>

<script type="module">
  import { MiniDLM, drawSkeleton } from "https://cdn.jsdelivr.net/gh/vendemiaraul-cyber/stickman-tracker-1@main/mini-dlm/mini-dlm.js";

  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");

  const dlm = await MiniDLM.create({ maxPeople: 2 });

  dlm.start(video, people => {
    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;
    for (const person of people) drawSkeleton(ctx, person);
  });
</script>
```

For production, pin a commit instead of `@main`, or copy `mini-dlm.js` into your own site.

## What you get for each person

```js
{
  id: 3,              // unique while the tracker runs
  slot: 0,            // 0 = first person on screen, 1 = second… reused when someone leaves
  confidence: 0.94,   // how sure the model is about the shoulders and hips
  center: { x, y },
  landmarks: [ { x, y, z, visibility, filled? }, … 33 points ],
  keypoints: { nose: {…}, left_wrist: {…}, … }   // the same points by name
}
```

- `x` and `y` go from 0 to 1 across the frame (multiply by your canvas size to draw)
- `z` is depth relative to the hips; smaller is closer to the camera
- `visibility` goes from 0 to 1
- `filled: true` means Mini DLM estimated the point because the camera could not see it

The 33 point names are exported as `LANDMARKS`, and the bone pairs as `SKELETON`.

## Options

```js
const dlm = await MiniDLM.create({
  maxPeople: 1,            // 1–6
  model: "lite",           // "lite" (fastest), "full", "heavy" (most accurate), or a URL to a .task file
  smoothing: "light",      // "off", "light", "strong", or { minCutoff, beta }
  predict: true,           // motion prediction between video frames
  fillHiddenLimbs: true,   // estimate limbs the camera can't see
  minConfidence: 0.5,
  delegate: "GPU"          // falls back to "CPU" automatically
});

await dlm.setOptions({ maxPeople: 4 });   // people, model and delegate changes reload the model
```

To host everything yourself, also set `visionModuleUrl` and `wasmBase` to your copy of `@mediapipe/tasks-vision@0.10.14`, and `model` to your copy of the `.task` file.

## Methods

| Method | What it does |
|---|---|
| `MiniDLM.create(options)` | Downloads the model and returns a ready tracker |
| `dlm.start(video, onFrame)` | Tracks a video on every screen refresh and calls `onFrame(people, stats)`. Returns a `stop()` function |
| `dlm.track(source)` | Runs the model once on a video, image or canvas and returns the people found. Skips a video frame it already processed |
| `dlm.getPeople()` | The people currently tracked, with prediction applied. Cheap to call every frame |
| `dlm.setOptions(options)` | Changes options |
| `dlm.reset()` | Forgets everyone, e.g. when loading a new video |
| `dlm.close()` | Frees the model |
| `dlm.stats` | `{ fps, inferenceMs, backend }` |

Helpers: `drawSkeleton(ctx, person, { color, lineWidth, width, height })` and `fillHiddenLimbs(landmarks, aspect)`.

## Tests

```sh
node test.mjs
```

The tests cover identity tracking, smoothing, prediction and limb filling with made-up poses, so they run without a browser or the model.

## Credits

Pose detection by [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) from Google, Apache-2.0. Mini DLM is released under the same license.
