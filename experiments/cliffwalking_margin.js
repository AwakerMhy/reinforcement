// Fixed no-Q* action-margin selector, evaluated with the official transition table.
const fs = require('fs');
const pathModule = require('path');
const {greedy, greedyReturn, prepare, hit, next, terminal} =
  require('./cliffwalking_equal_steps.js');

function referencePath(q) {
  let state = 36;
  const path = [];
  for (let step = 0; step < 500; step++) {
    const action = greedy(q, state), index = 4 * state + action;
    path.push([state, action]);
    if (terminal[index]) {
      if (path.length !== 13) throw Error('Expected a 13-step reference path');
      return path;
    }
    state = next[index];
  }
  throw Error('Reference policy did not terminate');
}

function margin(q, path) {
  let sum = 0;
  for (const [state, action] of path) {
    let alternative = -Infinity;
    for (let a = 0; a < 4; a++) if (a !== action)
      alternative = Math.max(alternative, q[4 * state + a]);
    sum += q[4 * state + action] - alternative;
  }
  return sum;
}

function mean(values) {
  return values.length ? values.reduce((s, x) => s + x, 0) / values.length : null;
}

function se(values) {
  if (values.length < 2) return null;
  const center = mean(values);
  return Math.sqrt(values.reduce((s, x) => s + (x - center) ** 2, 0) /
                   (values.length * (values.length - 1)));
}

function describe(rows, prefix) {
  const s = rows.map(r => r[prefix + '_strong_steps']);
  const w = rows.map(r => r[prefix + '_weak_steps']);
  const d = s.map((x, i) => x - w[i]);
  return {mean_strong_steps: mean(s), mean_weak_steps: mean(w),
          gap_strong_minus_weak: mean(d), gap_se: se(d),
          weak_first: d.filter(x => x > 0).length,
          strong_first: d.filter(x => x < 0).length,
          tied: d.filter(x => x === 0).length,
          strong_censored: rows.reduce((n, r) => n + r[prefix + '_strong_censored'], 0),
          weak_censored: rows.reduce((n, r) => n + r[prefix + '_weak_censored'], 0)};
}

function main() {
  const cohorts = Number(process.argv[2] ?? 6000), seed = Number(process.argv[3] ?? 20261128);
  const output = process.argv[4] ?? 'results/cliffwalking_margin';
  if (!Number.isInteger(cohorts) || cohorts < 1 || !Number.isInteger(seed))
    throw Error('Usage: node experiments/cliffwalking_margin.js [cohorts] [seed] [output]');
  const rows = [];
  for (let cohort = 0; cohort < cohorts; cohort++) {
    const pool = [], strong = [], weak = [];
    let reference = null;
    const assessment = {steps: 0};
    for (let j = 0; j < 64; j++) {
      const q = prepare((seed + Math.imul(cohort + 1, 7919) +
                         Math.imul(j + 1, 104729)) >>> 0);
      const quality = greedyReturn(q, assessment);
      pool.push(q);
      if (quality === -13 && reference === null) reference = j;
      if (quality === -15) strong.push(j);
      if (quality === -17) weak.push(j);
    }
    const row = {cohort, reference: reference ?? '', n_strong: strong.length,
                 n_weak: weak.length, admitted: 0,
                 selection_assessment_steps: assessment.steps,
                 positive_strong_index: '', positive_weak_index: '',
                 positive_margin_gap: '', negative_strong_index: '',
                 negative_weak_index: '', negative_margin_gap: '',
                 positive_strong_steps: '', positive_weak_steps: '',
                 negative_strong_steps: '', negative_weak_steps: '',
                 positive_strong_censored: '', positive_weak_censored: '',
                 negative_strong_censored: '', negative_weak_censored: '',
                 continuation_assessment_steps: ''};
    if (reference !== null && strong.length && weak.length) {
      const path = referencePath(pool[reference]);
      const scores = pool.map(q => margin(q, path));
      let positive = null, negative = null;
      for (const si of strong) for (const wi of weak) {
        const gap = scores[wi] - scores[si];
        if (gap > 1 && (positive === null || gap > positive.gap))
          positive = {si, wi, gap};
        if (gap < -1 && (negative === null || gap < negative.gap))
          negative = {si, wi, gap};
      }
      if (positive !== null && negative !== null) {
        row.admitted = 1;
        row.positive_strong_index = positive.si;
        row.positive_weak_index = positive.wi;
        row.positive_margin_gap = positive.gap;
        row.negative_strong_index = negative.si;
        row.negative_weak_index = negative.wi;
        row.negative_margin_gap = negative.gap;
        const futureSeed = (seed + Math.imul(cohort + 1, 2654435761)) >>> 0;
        const ps = hit(pool[positive.si], futureSeed);
        const pw = hit(pool[positive.wi], futureSeed);
        const ns = hit(pool[negative.si], futureSeed);
        const nw = hit(pool[negative.wi], futureSeed);
        row.positive_strong_steps = ps.steps; row.positive_weak_steps = pw.steps;
        row.negative_strong_steps = ns.steps; row.negative_weak_steps = nw.steps;
        row.positive_strong_censored = ps.censored;
        row.positive_weak_censored = pw.censored;
        row.negative_strong_censored = ns.censored;
        row.negative_weak_censored = nw.censored;
        row.continuation_assessment_steps = ps.assessment_steps + pw.assessment_steps +
                                            ns.assessment_steps + nw.assessment_steps;
      }
    }
    rows.push(row);
  }
  const admitted = rows.filter(r => r.admitted);
  const positive = describe(admitted, 'positive');
  const negative = describe(admitted, 'negative');
  const interaction = admitted.map(r =>
    (r.positive_strong_steps - r.positive_weak_steps) -
    (r.negative_strong_steps - r.negative_weak_steps));
  const summary = {experiment: 'cliffwalking_fixed_action_margin', seed, cohorts,
    transition_source: 'Gymnasium 1.4.0 CliffWalking-v1, is_slippery=false',
    reference_available: rows.filter(r => r.reference !== '').length,
    quality_classes_available: rows.filter(r => r.n_strong && r.n_weak).length,
    admitted: admitted.length, preparation_steps_total: cohorts * 64 * 2500,
    selection_assessment_steps_total: rows.reduce((n, r) =>
      n + r.selection_assessment_steps, 0),
    continuation_assessment_steps_total: admitted.reduce((n, r) =>
      n + r.continuation_assessment_steps, 0),
    positive, negative, interaction_mean: mean(interaction),
    interaction_se: se(interaction)};
  const header = Object.keys(rows[0]);
  fs.mkdirSync(pathModule.dirname(output), {recursive: true});
  fs.writeFileSync(output + '.csv', [header.join(','), ...rows.map(r =>
    header.map(key => r[key]).join(','))].join('\n') + '\n');
  fs.writeFileSync(output + '.json', JSON.stringify(summary, null, 2) + '\n');
  console.log(JSON.stringify(summary, null, 2));
}

main();
