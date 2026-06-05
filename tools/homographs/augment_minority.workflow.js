export const meta = {
  name: 'synth-augment-minority-homographs',
  description: 'Generate + adversarially verify synthetic sentences for starved minority homograph variants (floor 20, no accents in text)',
  phases: [
    { title: 'Generate', detail: 'one agent per variant emits need+buffer candidate sentences forcing the minority sense' },
    { title: 'Verify', detail: 'adversarial check of every candidate; force the minority stress, no stray accents' },
    { title: 'Topup', detail: 'conditional regenerate-and-self-verify when verification left a variant short' },
  ],
}

// Worklist is injected by tools/homographs/build_augment_workflow.py from the gitignored,
// corpus-derived data/homographs/augment_worklist.json (literary excerpts — NOT committed).
// This committed file is a TEMPLATE: running it directly throws (WORK is undefined).
// Generate and run the gitignored tools/homographs/augment_minority.local.js instead.
const WORK = __WORKLIST__
const FLOOR_BUFFER = 8

const GEN_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['sentences'],
  properties: {
    sentences: {
      type: 'array',
      items: { type: 'string' },
      description: 'Synthetic Bulgarian sentences, each with the target surface wrapped in « ».',
    },
  },
}

const VERIFY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['results'],
  properties: {
    results: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['sentence', 'valid', 'reason'],
        properties: {
          sentence: { type: 'string' },
          valid: { type: 'boolean' },
          reason: { type: 'string' },
        },
      },
    },
  },
}

function sensesBlock(w) {
  const lines = w.senses.map((s) => {
    const tag = s.target ? 'TARGET (minority — generate for THIS sense)' : 'OTHER sense (must NOT be evoked)'
    return `  - ord${s.ord} [${s.pos}, lemma=${s.lemma}] — ${tag}\n      attested: ${s.ex}`
  })
  return lines.join('\n')
}

function genPrompt(w, count) {
  const tgt = w.senses.find((s) => s.target)
  return `You are generating training data for a Bulgarian homograph stress disambiguator.

The written surface form «${w.surface}» is a homograph: identical letters, different stressed syllable depending on meaning. Here is the CLOSED set of attested meanings for this exact surface form (do not invent others):

${sensesBlock(w)}

Generate exactly ${count} NEW Bulgarian sentences that each use the surface form «${w.surface}» in the TARGET sense only:
  ord${tgt.ord} — part of speech: ${tgt.pos}, lemma: ${tgt.lemma}

HARD REQUIREMENTS (a sentence that violates ANY is useless):
1. Each sentence MUST contain the exact string «${w.surface}» — the surface wrapped in « and » (guillemets), exactly once, spelled letter-for-letter as given, with no accent on it.
2. The surrounding context MUST make the TARGET sense the ONLY natural reading. A literate Bulgarian must be unable to read it as any of the OTHER senses listed above. Lean on disambiguating cues: syntactic role, agreement, collocations, articles/prepositions, neighbouring words.
3. PLAIN TEXT ONLY — absolutely no stress marks anywhere: no combining acute (U+0301), no precomposed accented vowels (à, á, ѝ, ѝ, etc.). Just ordinary Bulgarian orthography.
4. Natural, grammatical, idiomatic Bulgarian. Vary length, register, topic, and sentence structure. No two sentences should feel templated.
5. Do NOT copy or lightly reword the attested examples above; write fresh contexts.

Return the ${count} sentences. Your output is consumed by a machine, not a person — return only the structured data.`
}

function verifyPrompt(w, candidates) {
  const tgt = w.senses.find((s) => s.target)
  const numbered = candidates.map((c, i) => `${i + 1}. ${c}`).join('\n')
  return `Adversarial review of synthetic training sentences for a Bulgarian homograph disambiguator.

Surface form: «${w.surface}». Closed set of attested meanings:

${sensesBlock(w)}

The TARGET sense these sentences must force is ord${tgt.ord} [${tgt.pos}, lemma=${tgt.lemma}].

For EACH candidate below, decide valid=true ONLY if ALL hold; otherwise valid=false with a one-clause reason:
  (a) it contains «${w.surface}» exactly once, spelled exactly, wrapped in guillemets;
  (b) a literate Bulgarian reads it UNAMBIGUOUSLY in the TARGET sense — it cannot plausibly be read as any OTHER listed sense (be strict: if another sense is even mildly plausible, reject);
  (c) no stress marks of any kind (no U+0301, no precomposed accented letters);
  (d) grammatical, natural, non-templated Bulgarian.

Default to valid=false when uncertain. Return one result per candidate, echoing the sentence verbatim. Candidates:

${numbered}`
}

phase('Generate')

const results = await pipeline(
  WORK,
  // Stage 1 — generate candidates
  (w) =>
    agent(genPrompt(w, w.need + FLOOR_BUFFER), {
      label: `gen:${w.surface}`,
      phase: 'Generate',
      schema: GEN_SCHEMA,
    }).then((g) => ({ w, candidates: (g?.sentences || []).filter((s) => typeof s === 'string') })),

  // Stage 2 — adversarial verify
  ({ w, candidates }) => {
    if (!candidates.length) return { w, kept: [] }
    return agent(verifyPrompt(w, candidates), {
      label: `verify:${w.surface}`,
      phase: 'Verify',
      schema: VERIFY_SCHEMA,
    }).then((v) => {
      const kept = (v?.results || []).filter((r) => r && r.valid).map((r) => r.sentence)
      return { w, kept }
    })
  },

  // Stage 3 — conditional top-up (generate + self-verify the shortfall)
  ({ w, kept }) => {
    const short = w.need - kept.length
    if (short <= 0) return { w, kept: kept.slice(0, w.need) }
    const tgt = w.senses.find((s) => s.target)
    const p = `${genPrompt(w, short + 4)}

You previously produced too few that survived strict review. Be MORE careful: before returning, silently reject any sentence where «${w.surface}» could be read in a non-target sense (target = ord${tgt.ord} ${tgt.pos} ${tgt.lemma}), then return only the survivors.`
    return agent(p, { label: `topup:${w.surface}`, phase: 'Topup', schema: GEN_SCHEMA })
      .then((g) => {
        const extra = (g?.sentences || []).filter((s) => typeof s === 'string')
        return { w, kept: kept.concat(extra).slice(0, w.need) }
      })
      .catch(() => ({ w, kept: kept.slice(0, w.need) }))
  },
)

const out = results.filter(Boolean).map((r) => ({
  surface: r.w.surface,
  ord: r.w.ord,
  need: r.w.need,
  sentences: r.kept,
}))

const totalKept = out.reduce((n, r) => n + r.sentences.length, 0)
log(`generated ${totalKept} verified sentences across ${out.length} variants`)

return { variants: out, totalKept }
