// Equal-environment-step CliffWalking comparison using official transitions.
// The protocol is documented in research/2026-11-25_cliffwalking_protocol.md.
const fs = require('fs');

const source = JSON.parse(fs.readFileSync('experiments/cliffwalking_transitions.json', 'utf8'));
if (source.environment !== 'CliffWalking-v1' || source.is_slippery !== false ||
    source.gymnasium_version !== '1.4.0') throw Error('Unexpected transition source');
const next = new Int16Array(48 * 4), reward = new Int16Array(48 * 4);
const terminal = new Uint8Array(48 * 4);
for (let s = 0; s < 48; s++) for (let a = 0; a < 4; a++) {
  const i = 4 * s + a, entry = source.transitions[s][a];
  [next[i], reward[i], terminal[i]] = entry;
}

function U(seed, step, channel) {
  let x = (seed + Math.imul(step + 1, 0x9e3779b9) +
           Math.imul(channel + 1, 0x85ebca6b)) >>> 0;
  x ^= x >>> 16; x = Math.imul(x, 0x7feb352d);
  x ^= x >>> 15; x = Math.imul(x, 0x846ca68b);
  return ((x ^ (x >>> 16)) >>> 0) / 4294967296;
}

function greedy(q, state) {
  let best = 0, value = q[4 * state];
  for (let a = 1; a < 4; a++) if (q[4 * state + a] > value) {
    best = a; value = q[4 * state + a];
  }
  return best;
}

function maxQ(q, state) {
  let value = q[4 * state];
  for (let a = 1; a < 4; a++) value = Math.max(value, q[4 * state + a]);
  return value;
}

function greedyReturn(q, tracker = null) {
  let state = 36, total = 0;
  for (let k = 0; k < 500; k++) {
    if (tracker) tracker.steps++;
    const index = 4 * state + greedy(q, state);
    total += reward[index];
    if (terminal[index]) return total;
    state = next[index];
  }
  return total;
}

function prepare(seed) {
  const q = new Float64Array(48 * 4);
  let state = 36, episodeSteps = 0;
  for (let step = 0; step < 2500; step++) {
    const action = U(seed, step, 0) < 0.2 ? Math.floor(4 * U(seed, step, 1)) :
                   greedy(q, state);
    const index = 4 * state + action, ns = next[index];
    const target = reward[index] + (terminal[index] ? 0 : maxQ(q, ns));
    q[index] += 0.5 * (target - q[index]);
    episodeSteps++;
    if (terminal[index] || episodeSteps === 500) {
      state = 36; episodeSteps = 0;
    } else state = ns;
  }
  return q;
}

function hit(initialQ, seed) {
  const q = new Float64Array(initialQ);
  let state = 36, episodeSteps = 0, completed = 0;
  const assessment = {steps: 0};
  for (let step = 1; step <= 3000; step++) {
    const action = U(seed, step - 1, 0) < 0.1 ?
                   Math.floor(4 * U(seed, step - 1, 1)) : greedy(q, state);
    const index = 4 * state + action, ns = next[index];
    const target = reward[index] + (terminal[index] ? 0 : maxQ(q, ns));
    q[index] += 0.5 * (target - q[index]);
    episodeSteps++;
    if (terminal[index] || episodeSteps === 500) {
      state = 36; episodeSteps = 0; completed++;
    } else state = ns;
    if (step % 25 === 0 && greedyReturn(q, assessment) >= -13) {
      return {episodes: completed, steps: step, censored: 0,
              assessment_steps: assessment.steps};
    }
  }
  return {episodes: completed, steps: 3000, censored: 1,
          assessment_steps: assessment.steps};
}

function summarize(rows, cohorts, seed) {
  const admitted = rows.filter(r => r.admitted);
  const d = admitted.map(r => r.strong_episodes - r.weak_episodes);
  const ds = admitted.map(r => r.strong_steps - r.weak_steps);
  const average = values => values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
  const se = values => values.length > 1 ? Math.sqrt(values.reduce((v, x) =>
    v + (x - average(values)) ** 2, 0) / (values.length * (values.length - 1))) : null;
  return {experiment: 'equal_step_cliffwalking_baseline',
          transition_source: 'Gymnasium 1.4.0 CliffWalking-v1, is_slippery=false',
          seed, cohorts, admitted: admitted.length,
          preparation_steps_total: cohorts * 64 * 2500,
          selection_assessment_steps_total: rows.reduce((n, r) =>
            n + r.selection_assessment_steps, 0),
          continuation_assessment_steps_total: admitted.reduce((n, r) =>
            n + r.strong_assessment_steps + r.weak_assessment_steps, 0),
          mean_strong_episodes: average(admitted.map(r => r.strong_episodes)),
          mean_weak_episodes: average(admitted.map(r => r.weak_episodes)),
          mean_strong_steps: average(admitted.map(r => r.strong_steps)),
          mean_weak_steps: average(admitted.map(r => r.weak_steps)),
          gap_episodes_strong_minus_weak: average(d), gap_episodes_se: se(d),
          gap_steps_strong_minus_weak: average(ds), gap_steps_se: se(ds),
          weak_first_episodes: d.filter(x => x > 0).length,
          strong_first_episodes: d.filter(x => x < 0).length,
          tied_episodes: d.filter(x => x === 0).length,
          weak_first_steps: ds.filter(x => x > 0).length,
          strong_first_steps: ds.filter(x => x < 0).length,
          tied_steps: ds.filter(x => x === 0).length,
          strong_censored: admitted.reduce((n, r) => n + r.strong_censored, 0),
          weak_censored: admitted.reduce((n, r) => n + r.weak_censored, 0)};
}

function main() {
  const cohorts = Number(process.argv[2] ?? 1000), seed = Number(process.argv[3] ?? 20261126);
  const output = process.argv[4] ?? 'results/cliffwalking_equal_steps';
  if (!Number.isInteger(cohorts) || cohorts < 1 || !Number.isInteger(seed))
    throw Error('Usage: node experiments/cliffwalking_equal_steps.js [cohorts] [seed] [output]');
  const rows = [];
  for (let cohort = 0; cohort < cohorts; cohort++) {
    let strong = null, weak = null, strongIndex = -1, weakIndex = -1;
    const selectionAssessment = {steps: 0};
    for (let j = 0; j < 64; j++) {
      const q = prepare((seed + Math.imul(cohort + 1, 7919) + Math.imul(j + 1, 104729)) >>> 0);
      const quality = greedyReturn(q, selectionAssessment);
      if (quality === -15 && strong === null) {strong = q; strongIndex = j;}
      if (quality === -17 && weak === null) {weak = q; weakIndex = j;}
    }
    const admitted = strong !== null && weak !== null;
    const row = {cohort, admitted: Number(admitted), strong_candidate: strongIndex,
                 weak_candidate: weakIndex, strong_initial_return: strong ? -15 : '',
                 weak_initial_return: weak ? -17 : '',
                 selection_assessment_steps: selectionAssessment.steps,
                 strong_episodes: '',
                 weak_episodes: '', strong_steps: '', weak_steps: '',
                 strong_assessment_steps: '', weak_assessment_steps: '',
                 strong_censored: '', weak_censored: ''};
    if (admitted) {
      const futureSeed = (seed + Math.imul(cohort + 1, 2654435761)) >>> 0;
      const s = hit(strong, futureSeed), w = hit(weak, futureSeed);
      row.strong_episodes = s.episodes; row.weak_episodes = w.episodes;
      row.strong_steps = s.steps; row.weak_steps = w.steps;
      row.strong_assessment_steps = s.assessment_steps;
      row.weak_assessment_steps = w.assessment_steps;
      row.strong_censored = s.censored; row.weak_censored = w.censored;
    }
    rows.push(row);
  }
  const header = Object.keys(rows[0]);
  fs.mkdirSync(require('path').dirname(output), {recursive: true});
  const csv = [header.join(','), ...rows.map(r => header.map(k => r[k]).join(','))].join('\n') + '\n';
  fs.writeFileSync(output + '.csv', csv);
  const summary = summarize(rows, cohorts, seed);
  fs.writeFileSync(output + '.json', JSON.stringify(summary, null, 2) + '\n');
  console.log(JSON.stringify(summary, null, 2));
}

if (require.main === module) main();

module.exports = {U, greedy, greedyReturn, prepare, hit, next, reward, terminal};
