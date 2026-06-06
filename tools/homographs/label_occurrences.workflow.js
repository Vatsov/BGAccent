export const meta = {
  name: 'homograph-labeling',
  description: 'Label Bulgarian homograph occurrences from corpus + cross-model agreement check',
  phases: [
    { title: 'Discover', detail: 'list task files' },
    { title: 'Label', detail: 'one Sonnet agent per homograph -> labels/' },
    { title: 'Agreement', detail: 'Haiku re-labels every 10th homograph -> labels_agreement/' },
  ],
}

// ROOT is discovered at runtime (pwd of the project working dir) so the workflow
// is machine-independent; Read/Write require absolute paths, hence we resolve it.
let ROOT = '.'

const SUMMARY = {
  type: 'object',
  additionalProperties: false,
  required: ['idx', 'surface', 'labeled', 'escapes', 'low_conf', 'distribution'],
  properties: {
    idx: { type: 'integer' },
    surface: { type: 'string' },
    labeled: { type: 'integer' },
    escapes: { type: 'integer' },
    low_conf: { type: 'integer' },
    distribution: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['ordinal', 'count'],
        properties: { ordinal: { type: 'integer' }, count: { type: 'integer' } },
      },
    },
  },
}

const DISCOVER = {
  type: 'object',
  additionalProperties: false,
  required: ['root', 'ids'],
  properties: {
    root: { type: 'string' },
    ids: { type: 'array', items: { type: 'integer' } },
  },
}

function pad(n) {
  return String(n).padStart(4, '0')
}

function labelPrompt(idx, outDir) {
  const p = pad(idx)
  return `Ти си експерт по българско ударение и омографи.

Прочети файла (използвай Read tool):
${ROOT}/data/homographs/tasks/${p}.json

Файлът съдържа:
- "surface": дума, която се пише еднакво, но има няколко възможни ударения
- "options": възможните варианти — всеки с "ordinal" (коя поред гласна е ударена, 0-базирано), "pos" (част на речта), "lemma" (основна форма), "gloss" (значение на английски)
- "occurrences": реални изречения, в които целевата дума е оградена с « »

За ВСЯКА occurrence реши кой option отговаря на употребата в КОНКРЕТНОТО изречение (по значение + част на речта).

Запиши резултата във файла (използвай Write tool):
${ROOT}/data/homographs/${outDir}/${p}.jsonl
— по ЕДИН JSON обект на ред (JSONL), точно в този формат:
{"id":"<occ id от occurrences>","ordinal":<int или null>,"escape":<true|false>,"confidence":"high|med|low","reason":"<до 8 думи>"}

Правила:
- "ordinal" = ordinal-ът на избрания option.
- Ако НИТО един option не пасва (думата в това изречение е друга лема/значение, което липсва в options — напр. възвратен глагол, който го няма), сложи "escape": true и "ordinal": null.
- "confidence": high = недвусмислено; med = вероятно; low = трудно/гранично.
- Запази точно същите "id" стойности като в occurrences.

Накрая върни StructuredOutput summary: idx, surface, labeled (брой редове), escapes, low_conf, distribution (брой по ordinal).`
}

phase('Discover')
const disc = await agent(
  `Изпълни с Bash: \`pwd\` за абсолютния път до проекта (върни го като "root") и ` +
    `\`ls data/homographs/tasks/ | grep -oE '[0-9]+'\` за индексите на всички NNNN.json файлове ` +
    `(върни ги като целочислен масив "ids").`,
  { label: 'discover tasks', phase: 'Discover', schema: DISCOVER },
)

ROOT = disc.root
const ids = [...new Set(disc.ids)].sort((a, b) => a - b)
const agree = new Set(ids.filter((_, i) => i % 10 === 0))
log(`task файлове: ${ids.length}; agreement извадка: ${agree.size} (Haiku)`)

const jobs = []
for (const idx of ids) {
  jobs.push(() => agent(labelPrompt(idx, 'labels'), {
    label: `A:${idx}`, phase: 'Label', model: 'sonnet', schema: SUMMARY,
  }))
}
for (const idx of ids) {
  if (agree.has(idx)) {
    jobs.push(() => agent(labelPrompt(idx, 'labels_agreement'), {
      label: `B:${idx}`, phase: 'Agreement', model: 'haiku', schema: SUMMARY,
    }))
  }
}

const res = (await parallel(jobs)).filter(Boolean)
const labelRes = res.filter((r) => !agree.has(r.idx) || res.filter((x) => x.idx === r.idx).length >= 1)

const totalLabeled = res.reduce((s, r) => s + (r.labeled || 0), 0)
const totalEscapes = res.reduce((s, r) => s + (r.escapes || 0), 0)
const totalLowConf = res.reduce((s, r) => s + (r.low_conf || 0), 0)

return {
  homographs_done: res.length,
  total_labeled: totalLabeled,
  total_escapes: totalEscapes,
  total_low_conf: totalLowConf,
  agreement_sample: agree.size,
  note: 'Per-occurrence labels written to data/homographs/labels/ and labels_agreement/. Aggregate + agreement computed post-run.',
}
