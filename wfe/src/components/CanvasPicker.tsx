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
  // a cell measured on this GPU (sycl-h3 plan measure) beats the GPU's curve
  const cell = gpu?.cells?.[`${canvas.width}x${canvas.height}|${seconds}`];
  if (cell) {
    const gibc = cell.over ? engine.gib + tokens * plan.gib_per_token : cell.gib ?? 0;
    const sps = cell.s_per_step ?? gpuStep(gpu!, tokens);
    return {
      tokens,
      gib: Math.round(gibc * 10) / 10,
      fits: !cell.over && gibc + plan.margin_gib <= plan.cap_gib,
      secondsPerStep: sps,
      total: sps * steps,
      measured: true,
    };
  }
  const gib = hit ? hit.peak : engine.gib + tokens * plan.gib_per_token;
  const secondsPerStep = gpu ? gpuStep(gpu, tokens)
    : hit?.s_per_step ?? plan.step_a * tokens * tokens + plan.step_b * tokens;
  return {
    tokens,
    gib: Math.round(gib * 10) / 10,
    fits: gib + plan.margin_gib <= plan.cap_gib,
    secondsPerStep,
    total: secondsPerStep * steps,
    measured: !!hit,
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
const LENGTHS = [5, 8, 10, 12, 15];

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
  const sw = (
    <div class="gpusw">
      <span class="k"><span class="i dim" props={{ innerHTML: "&#xf2db;" }} /> gpu</span>
      {gpus.length
        ? gpus.map((g) => (
          <button
            type="button"
            class={{ on: g === gpu }}
            attrs={{ title: `GPU ${g.gpu}${g.shared ? ", shared with the chat model" : ""} · a clip's step = ${g.step_scale.toFixed(2)} x the measured blocks` }}
            on={{ click: () => { state.pickGpu = g.gpu; render(); } }}
          >{`${short(g.name)}${g.shared ? " (shared)" : ""}`}</button>
        ))
        : <span class="hint">not measured yet (sycl-h3 plan measure)</span>}
      <span class="hint">{`engine ${engine.quant}`}</span>
    </div>
  );

  // every cell first: the colours run from the cheapest ratio in the table to the dearest
  const cells = plan.canvases.map((c) => LENGTHS.map((sec) => {
    const est = estimate(plan, c, engine, sec, props.steps, gpu);
    return { c, sec, est, ratio: est.total / sec };
  }));
  const fit = cells.flat().filter((x) => x.est.fits).map((x) => x.ratio);
  // the scale runs from this table's cheapest cell to its dearest
  const lo = Math.log(fit.length ? Math.min(...fit) : 1), hi = Math.log(fit.length ? Math.max(...fit) : 2);
  const heat = (r: number) => {
    const t = hi > lo ? Math.min(1, Math.max(0, (Math.log(r) - lo) / (hi - lo))) : 0;
    const hue = 120 * (1 - t);                    // green (cheap) -> red (dear)
    return { background: `hsl(${hue}, 45%, 21%)`, borderColor: `hsl(${hue}, 50%, 33%)` };
  };

  const header = (
    <tr>
      <th class="rh">canvas</th>
      {LENGTHS.map((sec) => <th>{`${sec}s`}</th>)}
    </tr>
  );
  const rows = cells.map((row) => {
    const c = row[0]!.c;
    const fits = row.filter((x) => x.est.fits);
    const best = fits.length ? Math.min(...fits.map((x) => x.ratio)) : -1;
    return (
      <tr>
        <th class="rh">
          <b>{`${c.width}x${c.height}`}</b>
          <span class="dim">{` ${c.mpx} Mp · ${c.aspect}`}</span>
        </th>
        {row.map(({ sec, est, ratio }) => {
          const chosen = c.width === props.width && c.height === props.height && sec === Math.round(props.seconds);
          return (
            <td>
              <button
                type="button"
                class={{ sel: chosen, best: est.fits && ratio === best, over: !est.fits }}
                style={est.fits ? heat(ratio) : {}}
                attrs={{
                  disabled: !est.fits,
                  title: est.fits
                    ? `${est.tokens.toLocaleString()} tokens · ~${est.gib} GiB peak · ${est.secondsPerStep.toFixed(1)} s/step` +
                      (gpu ? ` on ${short(gpu.name)}` : "") + ` · est. ${fmtT(est.total)} of sampling`
                    : `needs ~${est.gib} GiB, over the ${plan.cap_gib} GiB cap`,
                }}
                on={{ click: () => props.onPick(c.width, c.height, engine.quant, sec) }}
              >
                {est.fits ? `${Math.round(ratio)}×${est.measured ? "*" : ""}` : "—"}
                <small>{est.fits ? `${est.gib}G` : "over"}</small>
              </button>
            </td>
          );
        })}
      </tr>
    );
  });

  // the chosen cell, in words: tokens, VRAM against the cap, s/step, the total
  const cur = estimate(plan, { width: props.width, height: props.height } as Canvas, engine, props.seconds, props.steps, gpu);
  const pct = Math.min(100, (cur.gib / plan.cap_gib) * 100);
  const summary = (
    <div class={{ picksum: true, bad: !cur.fits }}>
      <span class="i" props={{ innerHTML: cur.fits ? "&#xf00c;" : "&#xf00d;" }} />
      <b>{`${props.width}x${props.height}`}</b>
      {` · ${props.seconds}s · ${props.steps} steps · ${engine.quant}${gpu ? ` · ${short(gpu.name)}` : ""} · `}
      <b>{cur.tokens.toLocaleString()}</b>{" tokens · VRAM "}<b>{`${cur.gib}`}</b>{`/${plan.cap_gib} GiB`}
      <span class="vbar"><span style={{ width: `${pct}%` }} /></span>
      <b>{`${cur.secondsPerStep.toFixed(1)}s`}</b>{"/step · est. total "}<b>{fmtT(cur.total)}</b>
    </div>
  );

  return (
    <div class="pickwrap">
      {sw}
      <table class="pick heat">
        <thead>{header}</thead>
        <tbody>{rows}</tbody>
      </table>
      <div class="hint">
        {"cell = "}<b>seconds of GPU per second of video</b>
        {` at ${props.steps} steps, and peak VRAM. Green is cheaper, red dearer; `}
        <span class="best-key">dashed</span>
        {" = best ratio in that row. "}
        {gpu
          ? `* = measured on the ${short(gpu.name)} (sycl-h3 plan measure), everything else from its measured step curve. Sampling only - the text encoder, decode and upscale come on top. `
          : "* = measured, everything else extrapolated from those. "}
        {`Greyed = past the ${plan.cap_gib} GiB cap.`}
      </div>
      {summary}
    </div>
  );
}
