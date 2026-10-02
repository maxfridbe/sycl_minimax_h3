/** Resolution and engine as one grid, rather than two dropdowns that can disagree.
 *
 *  A row is a canvas, sorted by pixel count; a column is a denoiser quantization. Each cell
 *  shows what that combination costs to render at the current clip length, and cells that
 *  will not fit in VRAM are disabled rather than hidden, so the shape of the wall is visible.
 *  Cells we have actually run are served from measured timings and marked; the rest come
 *  from the fitted model the server publishes at /api/plan.
 */
import { jsx } from "../jsx.js";
import { state } from "../state.js";
import type { Canvas, EngineInfo, Plan } from "../types.js";
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

export function estimate(
  plan: Plan, canvas: Canvas, engine: EngineInfo, seconds: number, steps: number,
): Estimate {
  const tokens = tokensFor(canvas.width, canvas.height, seconds);
  const key = `${engine.quant}|${canvas.width}x${canvas.height}|${Math.round(seconds)}`;
  const hit = plan.measured[key];
  const gib = hit ? hit.peak : engine.gib + tokens * plan.gib_per_token;
  const secondsPerStep = hit?.s_per_step ?? plan.step_a * tokens * tokens + plan.step_b * tokens;
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
  onPick: (width: number, height: number, engine: string) => void;
}

export function CanvasPicker(props: CanvasPickerProps) {
  const plan = state.plan;
  if (!plan) return <div class="hint">loading the canvas table…</div>;
  const engines = plan.engines.filter((e) => e.ready);
  if (!engines.length) return <div class="hint">no denoiser weights found on the box</div>;

  const header = (
    <tr>
      <th class="rh">canvas</th>
      {engines.map((e) => <th>{e.quant}</th>)}
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
        {engines.map((e) => {
          const est = estimate(plan, c, e, props.seconds, props.steps);
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
                      + `${est.secondsPerStep.toFixed(1)} s/step${est.measured ? " (measured)" : " (estimated)"}`
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
        {" cells were measured on this box; the rest are fitted. Disabled cells exceed "}
        {`${plan.cap_gib} GiB of VRAM.`}
      </div>
    </div>
  );
}
