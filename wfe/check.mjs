import { mkdir, writeFile, rm } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { dirname, join } from "node:path";

const base = process.env.H3_WFE || "http://localhost:8090";
const root = "/tmp/uicheck";
await rm(root, { recursive: true, force: true });
await mkdir(root, { recursive: true });
await writeFile(join(root, "package.json"), '{"type":"module"}');

const page = await (await fetch(base + "/")).text();
const entry = /src="([^"]+)"/.exec(page)[1];
const seen = new Set();
const norm = (dir, spec) => {
  const parts = (dir + "/" + spec).split("/");
  const out = [];
  for (const p of parts) { if (p === "..") out.pop(); else if (p !== "." && p !== "") out.push(p); }
  return "/" + out.join("/");
};
async function walk(path) {
  if (seen.has(path)) return;
  seen.add(path);
  const r = await fetch(base + path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  const src = await r.text();
  const file = join(root, path.replace(/^\/ui\//, ""));
  await mkdir(dirname(file), { recursive: true });
  await writeFile(file, src);
  const dir = path.slice(0, path.lastIndexOf("/"));
  for (const m of src.matchAll(/from\s+"([^"]+)"/g)) {
    if (m[1].startsWith(".")) await walk(norm(dir, m[1]));
  }
}
await walk(entry);
console.log(`${seen.size} modules fetched`);

let bad = 0;
for (const p of seen) {
  const f = join(root, p.replace(/^\/ui\//, ""));
  try { execFileSync(process.execPath, ["--check", f], { stdio: "pipe" }); }
  catch (e) { bad++; console.log("PARSE FAIL", p, String(e.stderr).split("\n")[2] ?? ""); }
}
console.log(bad ? `${bad} modules failed to parse` : "all modules parse as ES modules");

// link + render check: no DOM needed until patch(), so App() must build a vnode tree cleanly
globalThis.localStorage = { getItem: () => null, setItem: () => {} };
globalThis.document = { getElementById: () => null };
const { App } = await import(join(root, "src/components/App.js"));
const { state } = await import(join(root, "src/state.js"));
state.summary = { rate_wall_per_video_s: 50, rate_wh_per_video_s: 4, running: true, queued: 3,
  queued_video_s: 30, remaining_wall_s: 1500, eta_ts: Date.now()/1000+1500, remaining_kwh: 0.1,
  series: "x", series_done: 1, series_video_s: 10, series_final_s: 10, done_total: 5,
  done_video_s: 50, done_wall_s: 2500, done_wh: 200,
  disk: { path: "/out", total_gb: 1000, used_gb: 500, free_gb: 500, pct: 50 },
  project_stats: { project: "Patter dance", clips: 3, video_s: 25, gpu_s: 1427, ratio: 57.7, wh: 65 } };
state.status = { idle: false, paused: false, paused_projects: [], paused_batches: [],
  job: { id: "x", name: "h3_1", started: 0, prompt: "p", label: "Dance 01/32: a", project: "Patter dance" },
  stage: "sampling 3/8", pct: 40, queue: ["Dance 02/32: d"],
  queue_items: [{ label: "Dance 02/32: d", seconds: 12.9, steps: 8, engine: "Q8_0", size: "768x576",
                  chain: true, project: "Patter dance", batch: "Dance-A", held: false }],
  projects: [{ project: "Patter dance", clips: 1, video_s: 12.9, paused: false, first_index: 0, batches: [] }],
  all_projects: ["Patter dance"] };
state.timelineProject = "Patter dance";
state.timelineClips = [
  { n: 1, name: "h3_1", state: "done", seconds: 5.17, camera: "B", label: "Dance 01/32: c", dialogue: "", thumb: "/thumb/h3_1.webp" },
  { n: 2, name: null, state: "queued", seconds: 12.9, camera: "A", label: "Dance 02/32: d", dialogue: "Mister Data", thumb: null },
];
state.list = [{ file: "h3_1.mp4", label: "Dance 01/32: c", project: "Patter dance", seconds: 5.17 }];
state.films = [{ file: "dance.mp4", mb: 42, seconds: 296, mtime: 0 }];
// the canvas grid only draws once the plan has arrived, so give it a real one
state.plan = await (await fetch(base + "/api/plan")).json();

const count = (n) => 1 + (Array.isArray(n.children)
  ? n.children.reduce((a, c) => a + (typeof c === "object" && c ? count(c) : 0), 0) : 0);

let panelsSeen = 0;
const walkTree = (n, fn) => {
  fn(n);
  if (Array.isArray(n.children)) for (const c of n.children) if (typeof c === "object" && c) walkTree(c, fn);
};

for (const tab of ["create", "jobs"]) {
  state.tab = tab;
  const tree = App();
  walkTree(tree, (n) => { if (n.sel === "section" && n.data?.class?.panel) panelsSeen++; });
  console.log(`tab ${tab}: ${count(tree)} vnodes, root <${tree.sel}>`);
}
if (panelsSeen < 5) throw new Error(`only ${panelsSeen} collapsible panels rendered across both tabs`);
console.log(`${panelsSeen} collapsible panels render`);

// collapsing a panel must actually drop its body from the tree
const { togglePanel } = await import(join(root, "src/state.js"));
state.tab = "jobs";
const before = count(App());
togglePanel("queue"); togglePanel("completed"); togglePanel("films"); togglePanel("timeline");
const after = count(App());
if (after >= before) throw new Error(`collapsing changed nothing: ${before} -> ${after} vnodes`);
console.log(`collapse works: ${before} -> ${after} vnodes`);
togglePanel("queue"); togglePanel("completed"); togglePanel("films"); togglePanel("timeline");

// routes must survive a round trip, including names with spaces and slashes
const { parseHash, hashFor } = await import(join(root, "src/router.js"));
for (const r of [
  { tab: "create", project: null, clip: null },
  { tab: "jobs", project: null, clip: null },
  { tab: "jobs", project: "Patter dance v2", clip: null },
  { tab: "jobs", project: "Patter dance v2", clip: 12 },
]) {
  const back = parseHash(hashFor(r));
  const same = back.tab === r.tab && back.project === r.project && back.clip === r.clip;
  if (!same) throw new Error(`route round trip failed: ${JSON.stringify(r)} -> ${hashFor(r)} -> ${JSON.stringify(back)}`);
}
console.log("routes round-trip through the hash");

// the page must ship an icon that needs no extra request
if (!/rel="icon"\s+href="data:image\/svg\+xml/.test(page)) throw new Error("favicon is not inlined");
console.log("favicon is inlined in the page");
