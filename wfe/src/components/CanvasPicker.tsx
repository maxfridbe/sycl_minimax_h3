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
import { state } from "../state.js";
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
  onPick: (width: number, height: number, engine: string) => void;
}

export function CanvasPicker(props: CanvasPickerProps) {
  const plan = state.plan;
  if (!plan) return <div class="hint">loading the canvas table…</div>;
  const engines = plan.engines.filter((e) => e.ready);
  if (!engines.length) return <div class="hint">no denoiser weights found on the box</div>;

  // a column per engine, or per engine and GPU once the GPUs are measured
  const gpus = plan.gpus ?? [];
  const short = (n: string) => n.replace(/^Intel\(R\) Arc\(TM\) /, "").replace(/ Graphics$/, "");
  const cols: { e: EngineInfo; g?: GpuPlan }[] = engines.flatMap((e) => gpus.length ? gpus.map((g) => ({ e, g })) : [{ e }]);
  const header = (
    <tr>
      <th class="rh">canvas</th>
      {cols.map(({ e, g }) => (
        <th attrs={{ title: g ? `GPU ${g.gpu}${g.shared ? " (shared with the chat model)" : ""}` : "" }}>
          {g ? `${e.quant} · ${short(g.name)}` : e.quant}
        </th>
      ))}
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
        {cols.map(({ e, g }) => {
          const est = estimate(plan, c, e, props.seconds, props.steps, g);
          const chosen = chosenRow && e.quant === props.engine;
          return (
            <td>
              <button
                type="button"
                class={{ sel: chosen, meas: est.measured }}
                attrs={{
                  disabled: !est.fits,
                  title: est.fits
                    ? `${est.tokens.toLocaleString()} tokens · ~${est.gib} GiB peak · `
                      + `${est.secondsPerStep.toFixed(1)} s/step`
                      + (g ? ` on GPU ${g.gpu} (measured curve)` : est.measured ? " (measured)" : " (estimated)")
                    : `needs ~${est.gib} GiB, over the ${plan.cap_gib} GiB cap`,
                }}
                on={{ click: () => props.onPick(c.width, c.height, e.quant) }}
              >
                {est.fits ? fmtT(est.total) : "—"}
                <small>{est.fits ? `${est.gib} GiB` : "will not fit"}</small>
              </button>
            </td>
          );
        })}
      </tr>
    );
  });

  return (
    <div class="pickwrap">
      <table class="pick">
        <thead>{header}</thead>
        <tbody>{rows}</tbody>
      </table>
      <div class="hint">
        {`Time is for ${props.steps} steps at ${props.seconds}s. `}
        <b class="meas-key">Bold</b>
        {gpus.length
          ? " cells come from each GPU's measured step times (sycl-h3 plan measure). Disabled cells exceed "
          : " cells were measured on this box; the rest are fitted. Disabled cells exceed "}
        {`${plan.cap_gib} GiB of VRAM.`}
      </div>
    </div>
  );
}
