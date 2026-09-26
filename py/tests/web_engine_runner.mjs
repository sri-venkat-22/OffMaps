// Node side of tests/test_web_engine.py: runs docs/engine/*.js on a JSON job and writes JSON.
//   node web_engine_runner.mjs job.json out.json
// NaN travels as null both ways (JSON has no NaN).
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ENG = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "docs", "engine");
const load = (f) => import(pathToFileURL(path.join(ENG, f)).href);
const { SpeedNet } = await load("speednet.js");
const { FusionHead } = await load("head.js");
const { Filter, SpeedCal, Align, gqTrust, gqR, gqSpoof } = await load("core.js");
const { stageEngine, run, SharedNet } = await load("engine.js");

const job = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const nan = (a) => a.map((x) => (x === null ? NaN : Array.isArray(x) ? nan(x) : x));
const readJson = (f) => JSON.parse(fs.readFileSync(path.join(ENG, "model", f), "utf8"));
const bin = fs.readFileSync(path.join(ENG, "model", "speednet.bin"));
const net = new SpeedNet(readJson("speednet.json"), bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
let out;

if (job.mode === "speednet") {
  out = job.x.map((x) => net.forward(Float64Array.from(x.flat())));
} else if (job.mode === "core") {
  const f = new Filter(), sc = new SpeedCal(), al = new Align(), rows = [];
  for (const [op, ...a] of job.ops) {
    if (op === "noise") f.setNoise(...a);
    else if (op === "init") f.init(...a);
    else if (op === "predict") f.predict(...a);
    else if (op === "speed") f.updateSpeed(...a);
    else if (op === "zupt") f.updateZupt(...a);
    else if (op === "curv") f.updateCurvature(...a);
    else if (op === "gpos") f.updateGnssPos(...a);
    else if (op === "gvel") f.updateGnssVel(...a);
    else if (op === "cross") f.updateCrosstrack(...a);
    else if (op === "heading") f.updateHeading(...a);
    else if (op === "keep") f.setMapKeepSpeed(...a);
    else if (op === "hsig") f.setHeadingSigma(...a);
    rows.push([...f.state(), ...f.cov()]);
  }
  const cal = [];
  for (const [vn, vd, w, lam] of job.cal) { sc.setLambda(lam === null ? Infinity : lam); sc.push(vn, vd, w); cal.push(sc.fit()); }
  const aln = job.aln.map((a) => { al.update(...a); return al.get(); });
  const gq = job.gq.map(([cn0, sv, navic, dop, chi2]) => [gqTrust(cn0, sv, navic, dop, chi2), gqR(gqTrust(cn0, sv, navic, dop, chi2)), gqSpoof(cn0, sv, chi2, 0)]);
  out = { filter: rows, cal, aln, gq };
} else if (job.mode === "loop") {
  const profile = readJson("profile.json"), head = new FusionHead(readJson("fusion_head.json"));
  const imu = { t: job.imu.t, acc: job.imu.acc, gyro: job.imu.gyro, mag: nan(job.imu.mag) };
  const gn = Object.fromEntries(Object.entries(job.gnss).map(([k, v]) => [k, nan(v)]));
  const shared = new SharedNet(net);
  out = {};
  const t0 = Date.now();
  for (const s of job.stages) {
    const eng = stageEngine(s, { profile, net: shared, head, roads: job.roads || null });
    out[s] = run(eng, imu, gn, job.outages || [], { every: job.every || 1 });
  }
  out.ms = Date.now() - t0;
}
fs.writeFileSync(process.argv[3], JSON.stringify(out));
