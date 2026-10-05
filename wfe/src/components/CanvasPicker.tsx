/** Resolution and engine as one grid, rather than two dropdowns that can disagree.
 *
 *  A row is a canvas, sorted by pixel count; a column is a denoiser quantization. Each cell
 *  shows what that combination costs to render at the current clip length, and cells that
 *  will not fit in VRAM are disabled rather than hidden, so the shape of the wall is visible.
 *  Cells we have actually run are served from measured timings and marked; the rest come
 *  from the model the server publishes at /api/plan. When the box's GPUs have been measured
 *  (`sycl-h3 plan measure`), each engine gets a column per GPU, timed from that GPU's curve.
 */
import { jsx } from "../jsx.js";
import { render, state } from "../state.js";
import type { Canvas, EngineInfo, GpuPlan, Plan } from "../types.js";
import { fmtT } from "../util.js";

/** Latent tokens for a canvas at a clip length. Mirrors tokens_for() in wfe/server.py. */
export function tokensFor(width: number, height: number, seconds: number): number {
  let n = Math.max(5, Math.round(seconds * 24));
  while (n % 17 !== 5) n += 1;
  const ltf = n <= 5 ? 2 : Math.floor((n - 5) / 17) * 5 + 2;
  return ltf * Math.floor((width * height) / 1024) + Math.round((n / 24) * 40) + 10 + 336;
}

export interface Estimate {
  tokens: number;
  gib: number;
  fits: boolean;
  secondsPerStep: number;
  total: number;
  measured: boolean;
}

/** Seconds per step at `t` tokens on one GPU: linear between its measured points, the end segments' slope
 *  beyond them (mirrors plan::interp in the engine), times its clip factor. */
export function gpuStep(g: GpuPlan, t: number): number {
  const p = g.points;
  if (!p.length) return g.step_a * t * t + g.step_b * t;
  const p0 = p[0];
  if (p.length === 1 && p0) return (p0[1] * t / p0[0]) * g.step_scale;
  let k = p.findIndex((q) => q[0] >= t);
  k = Math.min(Math.max(k < 0 ? p.length - 1 : k, 1), p.length - 1);
  const a = p[k - 1], b = p[k];
  if (!a || !b) return 0;
  return (a[1] + ((b[1] - a[1]) * (t - a[0])) / (b[0] - a[0])) * g.step_scale;
}

export function estimate(
  plan: Plan, canvas: Canvas, engine: EngineInfo, seconds: number, steps: number, gpu?: GpuPlan,
): Estimate {
  const tokens = tokensFor(canvas.width, canvas.height, seconds);
  const key = `${engine.quant}|${canvas.width}x${canvas.height}|${Math.round(seconds)}`;
  const hit = gpu ? undefined : plan.measured[key];
  const gib = hit ? hit.peak : engine.gib + tokens * plan.gib_per_token;
  const secondsPerStep = gpu ? gpuStep(gpu, tokens)
    : hit?.s_per_step ?? plan.step_a * tokens * tokens + plan.step_b * tokens;
  return {
    tokens,
    gib: Math.round(gib * 10) / 10,
    fits: gib + plan.margin_gib <= plan.cap_gib,
    secondsPerStep,
    total: secondsPerStep * steps,
    measured: !!hit || !!gpu,
  };
}

export interface CanvasPickerProps {
  width: number;
  height: number;
  engine: string;
  seconds: number;
  steps: number;
  onPick: (width: number, height: number, engine: string, seconds: number) => void;
}

/** The clip lengths the table's columns show, in seconds of video generated. */
const LENGTHS = [2, 4, 6, 8, 10, 12, 15];

/** render time over clip length: green under 15:1, amber to 30:1, red above */
const ratioClass = (r: number) => (r < 15 ? "r-good" : r < 30 ? "r-mid" : "r-bad");

export function CanvasPicker(props: CanvasPickerProps) {
  const plan = state.plan;
  if (!plan) return <div class="hint">loading the canvas table…</div>;
  const engines = plan.engines.filter((e) => e.ready);
  if (!engines.length) return <div class="hint">no denoiser weights found on the box</div>;
  const engine = engines.find((e) => e.quant === props.engine) ?? engines[0]!;

  // the GPU: the one picked, else the fastest measured (lowest quadratic term); none measured: the plan's own curve
  const gpus = plan.gpus ?? [];
  const fastest = gpus.reduce<GpuPlan | undefined>((a, g) => (!a || g.step_a < a.step_a ? g : a), undefined);
  const gpu = gpus.find((g) => g.gpu === state.pickGpu) ?? fastest;
  const short = (n: string) => n.replace(/^Intel\(R\) Arc\(TM\) /, "").replace(/ Graphics$/, "");
  const sw = gpus.length > 1 ? (
    <div class="gpusw">
      <span class="hint">timed for</span>
      {gpus.map((g) => (
        <button
          type="button"
          class={{ on: g === gpu }}
          attrs={{ title: `GPU ${g.gpu}${g.shared ? ", shared with the chat model" : ""} · a clip's step = ${g.step_scale.toFixed(2)} x the measured blocks` }}
          on={{ click: () => { state.pickGpu = g.gpu; render(); } }}
        >{`${short(g.name)}${g.shared ? " (shared)" : ""}`}</button>
      ))}
    </div>
  ) : null;

  const cur = Math.round(props.seconds);
  const header = (
    <tr>
      <th class="rh">canvas</th>
      {LENGTHS.map((sec) => <th class={{ cur: sec === cur }}>{`${sec} s`}</th>)}
    </tr>
  );

  const rows = plan.canvases.map((c) => {
    const chosenRow = c.width === props.width && c.height === props.height;
    return (
      <tr>
        <th class="rh">
          <span class={{ dim: !chosenRow }}>{`${c.width}×${c.height}`}</span>
          <small>
            {`${c.aspect} · ${c.mpx} MP${c.note ? ` · ${c.note}` : ""}`}
          </small>
        </th>
        {LENGTHS.map((sec) => {
          const est = estimate(plan, c, engine, sec, props.steps, gpu);
          const ratio = est.total / sec;
          const chosen = chosenRow && sec === cur;
          return (
            <td>
              <button
                type="button"
                class={{ sel: chosen, meas: est.measured, [ratioClass(ratio)]: est.fits }}
                attrs={{
                  disabled: !est.fits,
                  title: est.fits
                    ? `${est.tokens.toLocaleString()} tokens · ~${est.gib} GiB peak · ${est.secondsPerStep.toFixed(1)} s/step` +
                      (gpu ? ` on ${short(gpu.name)}` : "") + ` · ${ratio.toFixed(1)} s of sampling per second of clip`
                    : `needs ~${est.gib} GiB, over the ${plan.cap_gib} GiB cap`,
                }}
                on={{ click: () => props.onPick(c.width, c.height, engine.quant, sec) }}
              >
                {est.fits ? fmtT(est.total) : "—"}
                <small>{est.fits ? `${ratio.toFixed(0)}:1 · ${est.gib} GiB` : "will not fit"}</small>
              </button>
            </td>
          );
        })}
      </tr>
    );
  });

  return (
    <div class="pickwrap">
      {sw}
      <table class="pick">
        <thead>{header}</thead>
        <tbody>{rows}</tbody>
      </table>
      <div class="hint">
        {`Sampling time for ${props.steps} steps by clip length${gpu ? ` on the ${short(gpu.name)} (measured curve)` : " (fitted)"}; `}
        {"the colour is render time per second of clip - "}
        <span class="r-good-t">under 15:1</span>{", "}
        <span class="r-mid-t">to 30:1</span>{", "}
        <span class="r-bad-t">above</span>
        {`. The text encoder, decode and upscale come on top. Disabled cells exceed ${plan.cap_gib} GiB of VRAM.`}
      </div>
    </div>
  );
}
