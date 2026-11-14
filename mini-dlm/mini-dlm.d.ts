export declare const VERSION: string;
export declare const LANDMARKS: readonly string[];
export declare const SKELETON: ReadonlyArray<[number, number]>;
export declare const MODEL_URLS: { lite: string; full: string; heavy: string };
export declare const SMOOTHING: Record<"off" | "light" | "strong", { minCutoff: number; beta: number } | null>;

export interface Landmark {
  /** 0–1 across the frame width */
  x: number;
  /** 0–1 down the frame height */
  y: number;
  /** depth relative to the hips; smaller is closer to the camera */
  z: number;
  /** 0–1 */
  visibility: number;
  /** true when Mini DLM estimated this point because the camera could not see it */
  filled?: boolean;
}

export interface Person {
  /** unique for the life of the tracker */
  id: number;
  /** 0 for the first person, 1 for the second… reused when someone leaves */
  slot: number;
  /** 33 landmarks in the order of LANDMARKS */
  landmarks: Landmark[];
  /** the same landmarks by name, e.g. keypoints.left_wrist */
  keypoints: Record<string, Landmark>;
  center: { x: number; y: number };
  /** average visibility of shoulders and hips */
  confidence: number;
}

export interface Stats { fps: number; inferenceMs: number; backend: "GPU" | "CPU" | null }

export interface MiniDLMOptions {
  /** 1–6, default 1 */
  maxPeople?: number;
  /** default "lite" */
  model?: "lite" | "full" | "heavy" | string;
  /** default "light" */
  smoothing?: "off" | "light" | "strong" | { minCutoff: number; beta: number };
  /** default true */
  predict?: boolean;
  /** default true */
  fillHiddenLimbs?: boolean;
  /** default 0.5 */
  minConfidence?: number;
  /** default "GPU", falls back to "CPU" */
  delegate?: "GPU" | "CPU";
  visionModuleUrl?: string;
  wasmBase?: string;
}

type Source = HTMLVideoElement | HTMLImageElement | HTMLCanvasElement | ImageBitmap | VideoFrame;

export declare class MiniDLM {
  static create(options?: MiniDLMOptions): Promise<MiniDLM>;
  readonly options: Required<MiniDLMOptions>;
  readonly backend: "GPU" | "CPU" | null;
  readonly stats: Stats;
  track(source: Source, timestamp?: number): Person[];
  getPeople(time?: number): Person[];
  start(video: HTMLVideoElement, onFrame?: (people: Person[], stats: Stats) => void): () => void;
  setOptions(options: MiniDLMOptions): Promise<void>;
  reset(): void;
  close(): void;
}

export declare function fillHiddenLimbs(landmarks: Landmark[], aspect?: number): Landmark[];
export declare function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  person: Person,
  options?: { width?: number; height?: number; color?: string; lineWidth?: number; jointColor?: string; showFilled?: boolean }
): void;
export default MiniDLM;
