import Foundation
import MLX

/// The composite on-device assistant — a faithful Swift port of canonical
/// `hypernet_sp/app_session_torch.AppSession.turn` (decision-for-decision): BGE 3-band intent
/// routing → specificity pinning → tiered retrieval (L1/L2/pins/web) → context-injection templates
/// → bounded SP-evict generation (DecodePolicy) or isolated clean-quote recall → groundedness gate
/// with rolled-back retries → calculator repair → memory writes.
public final class Assistant {
    public let sp: SPModel
    public let bge: Embedder
    public let intentHead: LinearHead
    public let specHead: LinearHead
    public let mem: TieredMemory
    /// L3 web tier. Settable so the app can switch to fully-offline mode (set to nil → lookups
    /// fall back to local files / honest miss, never touching the network).
    public var web: WebSearch?
    /// Optional local-file knowledge tier (Mac/iPhone). Nil → behaviour unchanged.
    public var localIndex: LocalFileIndex?
    /// Attention-mass retrieval trigger (multi-turn). Nil → disabled.
    public var trigger: TriggerScorer?
    /// v1 ships with the trigger OFF (owner decision 2026-06-12): core memory works without it
    /// (intent routing + pin/log injection + L1/L2 search), and the v3 head (attn_trigger3) does not
    /// separate in the production layout. Code is retained; flip on with the v4 head when ready.
    public var triggerEnabled = false
    public private(set) var lastTriggerScore: Float = 0
    public private(set) var lastTriggerFired = false
    /// Block Recall (BLOCK_RECALL_SPEC): re-inject the top-2 query-relevant evicted blocks verbatim
    /// so precise values survive SP compression. Ensemble (Phase B) when the indexer is present, else
    /// BGE (Phase A). OFF or empty archive ⇒ identical to before.
    public var archive: RecallArchive?
    public var blockRecallEnabled = true
    public private(set) var lastRecallTokens = 0
    /// TRIGGER_V3 recall gate (APP_SPEC_ADDENDUM): fire block-recall retrieve only on turns that
    /// actually need an evicted fact. Nil → ungated (retrieve every turn, the prior behaviour).
    public var gate: TriggerGate?
    public var recallGateEnabled = true
    public private(set) var lastGateScore: Float = 0
    public private(set) var lastGateFired = true
    /// RECALL V2 (trigger_experiment/ondevice_recall, chitchat-fixed): retrained gate (thresh −2.82,
    /// `[BOS][SP][window]` last-position layout) + BGE bridge-head retriever (BGE-base 768). When set
    /// and `useRecallV2`, `recallTurn` routes detection+retrieval through this instead of gate+archive.
    public var recallV2: RecallV2?
    public var useRecallV2 = false
    /// Full generation body of the last turn (includes the <think> CoT for compute turns).
    public private(set) var lastBody = ""
    /// Accumulated SP-evict conversation length (tokens). Blocks evict to the archive past rw.
    public var genTokenCount: Int { genState.gen.count }
    public var keptTokenCount: Int { genState.kept.count }

    private var genState = SPModel.GenState()    // bounded conversation (gen/kept/absorbed)
    private var v2BlockEmb: [[Float]] = []       // RECALL_V2 bridge: cached block BGE vectors
    // running conversation, for the trigger's SP and for re-injection when it fires
    private var convTokens: [Int] = []
    private var convTurns: [(user: String, assistant: String)] = []

    public init(sp: SPModel, bge: Embedder, intentHead: LinearHead, specHead: LinearHead,
                l2URL: URL? = nil, web: WebSearch? = WebSearch()) {
        self.sp = sp; self.bge = bge; self.intentHead = intentHead; self.specHead = specHead
        self.mem = TieredMemory(bge: bge, l2URL: l2URL)
        self.web = web
    }

    public struct Paths {
        public let sp: SPModel.Paths
        public let bgeDir: URL
        public let intentHead: URL
        public let specHead: URL
        public let l2URL: URL?
        public let triggerHead: URL?
        public let phasebIndexer: URL?
        public let gateHead: URL?
        public let bgeBaseDir: URL?        // BGE-base (768) for the V2 bridge-head retriever
        public let recallGateV2: URL?      // recall_gate_v2.safetensors
        public let recallBgeHead: URL?     // recall_bge_head.safetensors
        public init(sp: SPModel.Paths, bgeDir: URL, intentHead: URL, specHead: URL,
                    l2URL: URL? = nil, triggerHead: URL? = nil, phasebIndexer: URL? = nil,
                    gateHead: URL? = nil, bgeBaseDir: URL? = nil, recallGateV2: URL? = nil,
                    recallBgeHead: URL? = nil) {
            self.sp = sp; self.bgeDir = bgeDir; self.intentHead = intentHead
            self.specHead = specHead; self.l2URL = l2URL; self.triggerHead = triggerHead
            self.phasebIndexer = phasebIndexer; self.gateHead = gateHead
            self.bgeBaseDir = bgeBaseDir; self.recallGateV2 = recallGateV2
            self.recallBgeHead = recallBgeHead
        }
    }

    public static func load(_ p: Paths) async throws -> Assistant {
        let sp = try await SPModel.load(p.sp)
        let bge = try await Embedder.load(directory: p.bgeDir)
        let intent = try LinearHead.load(p.intentHead)
        let spec = try LinearHead.load(p.specHead)
        let a = Assistant(sp: sp, bge: bge, intentHead: intent, specHead: spec, l2URL: p.l2URL)
        if let th = p.triggerHead {
            a.trigger = try? TriggerScorer(model: sp.model, bos: sp.tokenizer.bosTokenId ?? 151646,
                                           nSP: sp.pooler.numSoftTokens, paramsURL: th)
        }
        // Phase B (ensemble indexer) when bundled, else Phase A (BGE).
        if let ix = p.phasebIndexer,
           let ens = try? EnsembleArchive(model: sp.model, tokenizer: sp.tokenizer, bge: bge,
                                          bos: sp.tokenizer.bosTokenId ?? 151646, indexerURL: ix) {
            a.archive = ens
        } else {
            a.archive = BlockArchive(tokenizer: sp.tokenizer, bge: bge)
        }
        if let gh = p.gateHead {
            a.gate = try? TriggerGate(model: sp.model, pooler: sp.pooler, tokenizer: sp.tokenizer,
                                      embedDtype: sp.embedDtype, bos: sp.tokenizer.bosTokenId ?? 151646,
                                      headURL: gh)
        }
        // RECALL V2 (chitchat-fixed): needs BGE-base (768) + retrained gate + bridge head.
        if let bbd = p.bgeBaseDir, let gv2 = p.recallGateV2, let bh = p.recallBgeHead,
           let bgeBase = try? await Embedder.load(directory: bbd, dim: 768) {
            a.recallV2 = try? RecallV2(model: sp.model, pooler: sp.pooler, tokenizer: sp.tokenizer,
                                       bge: bgeBase, embedDtype: sp.embedDtype,
                                       bos: sp.tokenizer.bosTokenId ?? 151646, gateURL: gv2, headURL: bh)
        }
        return a
    }

    public struct TurnResult {
        public let intent: String
        public let answer: String
        public let source: String?         // tier label ("L1·same-session", "L3·web", …)
        public let injected: [String]      // chunks injected
        public let acked: Bool             // stored/answered without bounded generation
        public let tokens: Int
        public let tokensPerSecond: Double
    }

    /// `specific_spans`: regex candidates scored by the specificity head (≥0.6).
    public func specificSpans(_ text: String, minP: Float = 0.6, cap: Int = 6) async -> [String] {
        let cands = Array(specificCandidates(text).prefix(24))
        guard !cands.isEmpty else { return [] }
        var scored = [(span: String, p: Float)]()
        for c in cands { scored.append((c, specHead.scorePositive(await bge.encode(c, isQuery: false)))) }
        return scored.filter { $0.p >= minP }.sorted { $0.p > $1.p }.prefix(cap).map { $0.span }
    }

    private func ack(_ msg: String, _ intent: String, _ src: String? = nil, _ chunks: [String] = []) -> TurnResult {
        TurnResult(intent: intent, answer: msg, source: src, injected: chunks, acked: true, tokens: 0, tokensPerSecond: 0)
    }

    /// #25 `_stitch`: record a NON-GENERATING turn (fact ack / honest miss / closest note / isolated
    /// recall quote) into the conversation stream so the next casual turn has it to follow. A short
    /// (<4-word) answer is wrapped in prose — a BARE-TOKEN record (e.g. "8042") is a copy attractor
    /// the next creative turn echoes verbatim. The user still SEES the terse answer; only the RECORD
    /// is wrapped. Tokens only; no generation.
    private func stitch(_ userMsg: String, _ answer: String) {
        var rec = answer
        if answer.split(whereSeparator: { $0 == " " || $0 == "\n" }).count < 4 {
            rec = "The answer to your question is \(answer)."
        }
        let text = (genState.gen.isEmpty ? "" : "<｜end▁of▁sentence｜>")
            + "<｜User｜>\(userMsg)<｜Assistant｜>\(rec)"
        genState.gen.append(contentsOf: sp.tokenizer.encode(text: text, addSpecialTokens: false))
    }

    public func turn(_ userMsg: String, store: String = "session", ackOnly: Bool = false,
                     retries: Int = 2, options: SPModel.Options = .default) async -> TurnResult {
        if ackOnly {
            if store == "persist" { mem.persist(userMsg) } else { mem.remember_session(userMsg) }
            stitch(userMsg, "Got it — saved."); return ack("Got it — saved.", "fact")
        }
        let doc = await bge.encode(userMsg, isQuery: false)
        let intent = routeIntent(userMsg, intentHead: intentHead, docEmbedding: doc)
        let computeLike = intent == "math" || intent == "command"

        // store-request phrased as a question → persist + pin
        if mcWantsPersist(userMsg), mcFactlike(userMsg), !(await specificSpans(userMsg)).isEmpty {
            mem.persist(userMsg); mem.pin(userMsg)
            stitch(userMsg, "Got it — saved."); return ack("Got it — saved.", "fact")
        }
        if intent != "recall", intent != "lookup", !(await specificSpans(userMsg)).isEmpty {
            mem.pin(userMsg)
        }
        if intent == "fact" {
            if store == "persist" || mcWantsPersist(userMsg) { mem.persist(userMsg) }
            else { mem.remember_session(userMsg) }
            stitch(userMsg, "Got it — saved."); return ack("Got it — saved.", "fact")
        }

        // ---- retrieval trigger (multi-turn) --------------------------------------------
        // Does this turn reference a value now living only in the compressed SP history? One short
        // eager forward over [BOS|SP|turn] reads the model's own attention back onto the SP. When it
        // fires, re-inject the recent transcript so a bare follow-up ("is that right?", "さっきの〜")
        // is answered from context instead of being treated as a brand-new question.
        var referential = false
        lastTriggerScore = 0
        if triggerEnabled, let trig = trigger, !convTokens.isEmpty {
            let hist = Array(convTokens.suffix(4096))
            let emb = sp.model.embed(MLXArray(hist.map { Int32($0) }, [1, hist.count])).asType(.float32)
            let spVec = sp.pooler.forward(emb)
            let toks = sp.tokenizer.encode(text: userMsg, addSpecialTokens: false)
            lastTriggerScore = trig.score(sp: spVec, turnIds: toks)
            referential = lastTriggerScore >= trig.fireThreshold
        }
        lastTriggerFired = referential
        let recentChunks = convTurns.suffix(3).map {
            "Earlier the user said \"\($0.user)\" and you answered \"\($0.assistant)\"."
        }

        var src: String? = nil
        var chunks: [String] = []
        if intent == "recall" {
            (src, chunks) = await mem.retrievePersonal(userMsg)
            if chunks.isEmpty, let li = localIndex {
                // the user's own indexed files are personal knowledge too
                let hits = await li.search(userMsg, k: 3, minSim: 0.55)
                if !hits.isEmpty { src = "L·files"; chunks = hits.map { "\($0.source): \($0.text)" } }
            }
            if chunks.isEmpty {
                // referential follow-up the memory doesn't hold → answer from the recent transcript
                if referential, !recentChunks.isEmpty { src = "↩︎recall"; chunks = recentChunks }
                else {
                    let miss = "I don't have that saved — you haven't told me yet."
                    stitch(userMsg, miss); return ack(miss, intent)
                }
            }
        } else if intent == "lookup" {
            (src, chunks) = await mem.retrieveKnown(userMsg)
            if chunks.isEmpty {
                let wmOnly = mem.pins.filter { !mem.session.contains($0) }
                var localHits: [(text: String, source: String, sim: Float)] = []
                if let li = localIndex { localHits = await li.search(userMsg, k: 3, minSim: 0.55) }
                if !wmOnly.isEmpty, let mp = await mem.semMatches(userMsg, wmOnly, minSim: 0.5), !mp.isEmpty {
                    src = "WM·pins"; chunks = mp
                } else if !localHits.isEmpty {
                    // prefer the user's own local documents over the open web
                    src = "L·files"; chunks = localHits.map { "\($0.source): \($0.text)" }
                } else if let web {
                    let q = expandWebQuery(userMsg, pins: mem.pins, session: mem.session)
                    let raw = await web.search(q)
                    // rank snippets by relevance to the actual question before taking the top 2
                    let ranked = await rerankByCosine(raw, query: userMsg, bge: bge)
                    let ch = guardChunks(Array(ranked.prefix(2)))
                    if !ch.isEmpty { src = "L3·web"; chunks = ch }
                }
            }
        } else if intent == "command", !mem.session.isEmpty || !mem.pins.isEmpty {
            let log = Array(mem.session.suffix(TieredMemory.LOGCAP))
            chunks = log + mem.pins.filter { p in !log.contains { $0.hasPrefix(p) } }
            src = "L1·same-session" + (mem.pins.isEmpty ? "" : "+WM·pins")
        } else if intent == "math", mem.pins.contains(where: { $0 != userMsg }) {
            let prev = mem.pins.filter { $0 != userMsg && !mcIsQuestion($0) }
            let log = Array(mem.session.suffix(TieredMemory.LOGCAP)).filter { !mcIsQuestion($0) }
            chunks = log + prev.filter { c in !log.contains { $0.hasPrefix(c) } }
            src = "WM·pins" + (log.isEmpty ? "" : "+L1")
        }
        // trigger fired but nothing else retrieved (e.g. a bare chitchat follow-up) → recent transcript
        if chunks.isEmpty, referential, !recentChunks.isEmpty, !computeLike {
            src = "↩︎context"; chunks = recentChunks
        }
        let triggerCtx = (src?.hasPrefix("↩︎") ?? false) && !chunks.isEmpty

        // ---- context-injection template -------------------------------------------------
        let aug: String
        if triggerCtx {
            // trigger-injected recent transcript → answer the follow-up conversationally (not a
            // strict verbatim quote, and skip the retrieval-confidence hedge — this is context, not a fact lookup)
            aug = "Recent conversation:\n\(chunks.joined(separator: "\n"))\n\n"
                + "The user now says: \(userMsg)\nThis refers back to that conversation. Answer briefly using it."
        } else if !chunks.isEmpty, computeLike {
            // canonical: rel = sem_matches(...) or chunks[-2:]  (never empty when chunks non-empty)
            var rel = (await mem.semMatches(userMsg, chunks, minSim: 0.4, cap: 3)) ?? []
            if rel.isEmpty { rel = Array(chunks.suffix(2)) }
            rel = withAmendments(rel, chunks)
            rel = markSuperseded(rel)
            let multi = !regexMatches(userMsg, "\\([a-c]\\)").isEmpty
            let tail = multi
                ? "Answer EVERY lettered part; end with one line listing each part's result."
                : "End with the final number."
            aug = "\(userMsg)\n\n(Earlier in this conversation: \(rel.joined(separator: " ; ")))\n"
                + "Use those earlier values if the question refers to them. Ignore lines marked (outdated). \(tail)"
        } else if !chunks.isEmpty {
            // retrieval-confidence hedge band (0.62): below the true-match floor, don't ASSERT
            let qv = await bge.encode(userMsg, isQuery: true)
            var sims = [Float]()
            for c in chunks { sims.append(cosine(await bge.encode(c, isQuery: false), qv)) }
            if (sims.max() ?? 0) < 0.62 {
                let bestI = sims.firstIndex(of: sims.max()!) ?? 0
                let note = "I don't have that saved exactly — the closest note I have: \"\(chunks[bestI])\""
                stitch(userMsg, note); return ack(note, intent, src, chunks)
            }
            aug = "Context (retrieved from \(src ?? "")): \(chunks.joined(separator: " ; "))\n\n"
                + "Question: \(userMsg)\nThe answer is stated EXPLICITLY in the Context above. Do NOT "
                + "calculate, reason about, or transform it, and ignore anything earlier in the conversation "
                + "— just read the matching value from the Context and reply with ONLY that value, verbatim "
                + "(keep letter prefixes/punctuation, e.g. 'EMP-1234' not '1234'; use the most recent value "
                + "if it was corrected)."
        } else {
            aug = userMsg
        }

        // ---- generation (canonical AppSession.turn routing) -----------------------------
        // recall/lookup with retrieved context → isolated clean-quote (fresh cache, no SP, no
        // history). Everything else → bounded SP-evict genOnce, with a snapshot rolled back before
        // each groundedness retry so a bad attempt doesn't pollute the conversation stream.
        let quoteRecall = !chunks.isEmpty && !computeLike
        let checkChunks = (computeLike || triggerCtx) ? [] : chunks
        let t0 = Date()
        var answer: String
        var body: String? = nil
        if quoteRecall {
            answer = sp.cleanQuote(aug, shouldContinue: options.shouldContinue)
            var tries = 0
            while !answerOk(answer, checkChunks, userMsg), tries < retries {
                tries += 1
                answer = sp.cleanQuote(aug, seed: UInt64(tries), shouldContinue: options.shouldContinue)
            }
        } else {
            // Block Recall (BLOCK_RECALL_SPEC §3): at turn start, seal the tokens already past the
            // window into the archive, then retrieve the top-2 query-relevant blocks to re-inject.
            var recIds: [Int] = []
            if blockRecallEnabled, let archive {
                let absorbedTarget = max(0, genState.gen.count - options.rw)
                if absorbedTarget > 0 { await archive.sync(absorbed: Array(genState.gen.prefix(absorbedTarget))) }
                // TRIGGER_V3 gate (APP_SPEC_ADDENDUM): retrieve only when this turn needs an evicted
                // fact — skips block-recall injection on chitchat/window-local turns (cuts over-injection).
                var fire = true
                if recallGateEnabled, let gate {
                    let window = Array(genState.gen.suffix(options.rw))
                    fire = gate.fires(kept: genState.kept, window: window, query: userMsg)
                    lastGateScore = gate.lastScore
                }
                lastGateFired = fire
                if fire { recIds = await archive.retrieve(query: userMsg) }
            }
            lastRecallTokens = recIds.count
            // #25/#20: compute keeps the open think + numeric convergence (k=3) + "Final answer:"
            // salvage; non-compute gets a pre-closed think (force_think=false), no convergence
            // forcing, natural salvage, and rejects a numbers-only reply (copy artifact) — escalating
            // temperature on retry to escape a bare-token copy attractor from the prior turn.
            let pol = DecodePolicy(k: computeLike ? 3 : 1_000_000_000)
            let salvage = computeLike ? "Final answer: " : ""
            let salvageBudget = computeLike ? 48 : 200
            func okAnswer(_ a: String) -> Bool {
                if !computeLike, regexMatches(a, "[A-Za-z]{2,}").isEmpty { return false }
                return answerOk(a, checkChunks, userMsg)
            }
            let snap = genState
            let g0 = sp.genOnce(aug, state: &genState, options: options, firstTurn: snap.gen.isEmpty,
                                policy: pol, recIds: recIds, forceThink: computeLike,
                                salvage: salvage, salvageBudget: salvageBudget)
            answer = g0.answer; body = g0.body
            var tries = 0
            while !okAnswer(answer), tries < retries {
                genState = snap
                let tempOv: Float? = computeLike ? nil : (tries == 0 ? 0.85 : 1.0)
                let g1 = sp.genOnce(aug, state: &genState, options: options, firstTurn: snap.gen.isEmpty,
                                    policy: pol, recIds: recIds, forceThink: computeLike,
                                    salvage: salvage, salvageBudget: salvageBudget, tempOverride: tempOv)
                answer = g1.answer; body = g1.body
                tries += 1
            }
        }
        let secs = Date().timeIntervalSince(t0)
        lastBody = body ?? answer

        // post-hoc calculator repair (compute turns)
        if computeLike, !answer.isEmpty {
            let (fixed, corr) = repairAnswer(answer, fullBody: body)
            if !corr.isEmpty { answer = fixed }
        }
        // #25: an isolated clean-quote answer leaves a trace in the stream too (prose-wrapped if
        // terse), so the next casual turn has continuity AND a bare value can't become a copy attractor.
        if quoteRecall { stitch(userMsg, answer) }
        // memory writes
        if store == "persist" { mem.persist(userMsg) }
        // self-log the RESULT of a compute turn (declarative, supersedes previous)
        if computeLike, answerOk(answer, [], userMsg) {
            let nums = regexMatches(answer, "\\$?\\d[\\d,]*(?:\\.\\d+)?")
            if let val = lastBoxed(answer) ?? nums.last {
                mem.dropComputedResults()
                mem.remember_session("Earlier computed result: \(val).")
            }
        }

        // record this turn for the next turn's trigger (SP + re-injection). English core text.
        if !answer.isEmpty {
            convTurns.append((user: userMsg, assistant: answer))
            if convTurns.count > 8 { convTurns.removeFirst(convTurns.count - 8) }
            let turnToks = sp.tokenizer.encode(text: "User: \(userMsg)\nAssistant: \(answer)\n",
                                               addSpecialTokens: false)
            convTokens.append(contentsOf: turnToks)
            if convTokens.count > 6000 { convTokens.removeFirst(convTokens.count - 6000) }
        }

        let ntok = sp.tokenizer.encode(text: answer, addSpecialTokens: false).count
        return TurnResult(intent: intent, answer: answer, source: src, injected: chunks, acked: false,
                          tokens: ntok, tokensPerSecond: secs > 0 ? Double(ntok) / secs : 0)
    }

    /// ROUTING-FREE recall measurement (`ルーティングは切って 単純にリコールだけの性能を見たい`).
    ///
    /// Bypasses `routeIntent` entirely — no intent classification, no cleanQuote/recall/lookup/fact
    /// branching, no memory writes. EVERY message goes straight through the bounded SP-evict `genOnce`
    /// path, accumulating in `genState`. Recall is handled SOLELY by the gate (`TriggerGate.fires`) +
    /// archive (`RecallArchive.retrieve`): the gate decides whether this turn needs an evicted fact,
    /// and if so the top-2 query-relevant blocks are re-injected verbatim. This is the path the
    /// gate-style recall was developed to occupy — measuring it in isolation, without the router that
    /// it was built to replace. `lastRecallTokens` / `lastGateFired` / `lastGateScore` report what the
    /// gate+archive did each turn.
    public func recallTurn(_ userMsg: String, options: SPModel.Options = .default) async -> TurnResult {
        // RECALL_V2 (in-generation gate): the gate is a per-position classifier, so detection+retrieval
        // happen DURING generation (genOnceRecall), not as a single pre-gen check. Candidate blocks =
        // the prior conversation (genState.gen), 128-aligned.
        if useRecallV2, let rv = recallV2 {
            let t0 = Date()
            let snap = genState
            let blocks = rv.makeBlocks(genState.gen)
            let pol = DecodePolicy(k: 1_000_000_000)
            func okAnswer(_ a: String) -> Bool {
                if regexMatches(a, "[A-Za-z]{2,}").isEmpty { return false }
                return answerOk(a, [], userMsg)
            }
            var emb = v2BlockEmb
            let g0 = await sp.genOnceRecall(userMsg, state: &genState, options: options,
                                            firstTurn: snap.gen.isEmpty, recall: rv, recallBlocks: blocks,
                                            blockEmb: &emb, policy: pol, forceThink: false,
                                            salvage: "", salvageBudget: 200)
            var answer = g0.answer, body = g0.body
            var fired = g0.fired, recTok = g0.recallTokens, score = g0.maxScore
            var tries = 0
            while !okAnswer(answer), tries < 2 {
                genState = snap
                var e2 = v2BlockEmb
                let g1 = await sp.genOnceRecall(userMsg, state: &genState, options: options,
                                                firstTurn: snap.gen.isEmpty, recall: rv, recallBlocks: blocks,
                                                blockEmb: &e2, policy: pol, forceThink: false,
                                                salvage: "", salvageBudget: 200,
                                                tempOverride: tries == 0 ? 0.85 : 1.0)
                answer = g1.answer; body = g1.body; fired = g1.fired; recTok = g1.recallTokens
                score = g1.maxScore; emb = e2; tries += 1
            }
            v2BlockEmb = emb
            lastGateFired = fired; lastGateScore = score; lastRecallTokens = recTok
            lastBody = body
            let secs = Date().timeIntervalSince(t0)
            if !answer.isEmpty {
                convTurns.append((user: userMsg, assistant: answer))
                if convTurns.count > 8 { convTurns.removeFirst(convTurns.count - 8) }
            }
            let nt = sp.tokenizer.encode(text: answer, addSpecialTokens: false).count
            return TurnResult(intent: "recall-only", answer: answer, source: nil, injected: [], acked: false,
                              tokens: nt, tokensPerSecond: secs > 0 ? Double(nt) / secs : 0)
        }
        var recIds: [Int] = []
        if blockRecallEnabled, let archive {
            let absorbedTarget = max(0, genState.gen.count - options.rw)
            if absorbedTarget > 0 { await archive.sync(absorbed: Array(genState.gen.prefix(absorbedTarget))) }
            var fire = true
            if recallGateEnabled, let gate {
                let window = Array(genState.gen.suffix(options.rw))
                fire = gate.fires(kept: genState.kept, window: window, query: userMsg)
                lastGateScore = gate.lastScore
            }
            lastGateFired = fire
            if fire { recIds = await archive.retrieve(query: userMsg) }
        }
        lastRecallTokens = recIds.count

        let t0 = Date()
        let snap = genState
        // Pure chitchat decode policy (no compute convergence, pre-closed think, natural salvage),
        // reject a numbers-only copy-artifact reply, escalate temperature on retry.
        let pol = DecodePolicy(k: 1_000_000_000)
        func okAnswer(_ a: String) -> Bool {
            if regexMatches(a, "[A-Za-z]{2,}").isEmpty { return false }
            return answerOk(a, [], userMsg)
        }
        let g0 = sp.genOnce(userMsg, state: &genState, options: options, firstTurn: snap.gen.isEmpty,
                            policy: pol, recIds: recIds, forceThink: false,
                            salvage: "", salvageBudget: 200)
        var answer = g0.answer; var body = g0.body
        var tries = 0
        while !okAnswer(answer), tries < 2 {
            genState = snap
            let g1 = sp.genOnce(userMsg, state: &genState, options: options, firstTurn: snap.gen.isEmpty,
                                policy: pol, recIds: recIds, forceThink: false,
                                salvage: "", salvageBudget: 200,
                                tempOverride: tries == 0 ? 0.85 : 1.0)
            answer = g1.answer; body = g1.body
            tries += 1
        }
        lastBody = body ?? answer
        let secs = Date().timeIntervalSince(t0)

        if !answer.isEmpty {
            convTurns.append((user: userMsg, assistant: answer))
            if convTurns.count > 8 { convTurns.removeFirst(convTurns.count - 8) }
        }
        let ntok = sp.tokenizer.encode(text: answer, addSpecialTokens: false).count
        return TurnResult(intent: "recall-only", answer: answer, source: nil, injected: [], acked: false,
                          tokens: ntok, tokensPerSecond: secs > 0 ? Double(ntok) / secs : 0)
    }

    /// PURE SP turn (final build): no memory, no recall, no web — just bounded SP-evict generation
    /// with CoT OPEN and a hard 4000-token think cap (force-close `</think>`). Multi-turn continuity
    /// rides `genState` (the SP-compressed running conversation). Answer length bounded only by
    /// `options.genLen` (kept large). This is "just an on-device SP chat model".
    public func turnSP(_ userMsg: String, options: SPModel.Options = .default) async -> TurnResult {
        let t0 = Date()
        let g = sp.genOnce(userMsg, state: &genState, options: options, firstTurn: genState.gen.isEmpty,
                           policy: DecodePolicy(k: 1_000_000_000), forceThink: true,
                           salvage: "", salvageBudget: 200, thinkCap: 4000)
        lastBody = g.body
        let answer = normalizeDisplay(g.answer)
        let secs = Date().timeIntervalSince(t0)
        if !answer.isEmpty {
            convTurns.append((user: userMsg, assistant: answer))
            if convTurns.count > 8 { convTurns.removeFirst(convTurns.count - 8) }
        }
        let nt = sp.tokenizer.encode(text: answer, addSpecialTokens: false).count
        return TurnResult(intent: "sp", answer: answer, source: nil, injected: [], acked: false,
                          tokens: nt, tokensPerSecond: secs > 0 ? Double(nt) / secs : 0)
    }

    /// Always-on persona (OPERATIONS_SPEC FIX3): a short declarative profile in the never-evicted MQ
    /// prefix so every turn — chitchat included — answers directly and never refuses benign requests.
    public var persona = "You are a warm, helpful assistant chatting with the user. You answer "
        + "directly, conversationally, and concisely, and you never refuse a reasonable request."

    /// MQ-prefix token ids: BOS + persona + a declarative line of in-session persisted facts (never
    /// question-shaped). Used to prime every router-free generation.
    private func profileIds() -> [Int] {
        // Persona ONLY — no persisted facts in the prefix. The first turn must be clean (no SP, no
        // injected context); fact recall is the gate's job, not the always-on prefix. (Avoids stale
        // facts bleeding into every turn, e.g. a prior session's "locker code".)
        let bos = sp.tokenizer.bosTokenId ?? 151646
        return [bos] + sp.tokenizer.encode(text: persona, addSpecialTokens: false)
    }

    private func isRefusal(_ a: String) -> Bool {
        let l = a.lowercased()
        return l.contains("can't assist") || l.contains("cannot assist") || l.contains("can't help with that")
            || l.contains("i'm sorry, but i can") || l.contains("i am unable to") || l.contains("i'm not able to")
    }

    /// Display normalization (OPERATIONS_SPEC FIX2/FIX8): unwrap `\boxed{…}`, strip leaked special
    /// tokens / stray think tags.
    private func normalizeDisplay(_ a: String) -> String {
        var s = a
        if let m = regexMatches(s, "\\\\boxed\\{([^}]*)\\}").last { s = s.replacingOccurrences(of: "\\boxed{\(m)}", with: m) }
        for tok in ["<think>", "</think>", "<｜Assistant｜>", "<｜User｜>", "<｜end▁of▁sentence｜>", "<｜begin▁of▁sentence｜>"] {
            s = s.replacingOccurrences(of: tok, with: "")
        }
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// ROUTER-FREE turn (OPERATIONS_SPEC §3, the unified production pipeline). No intent classifier:
    /// every turn runs ONE generation path with memory + Web injected on-demand via the retrievability
    /// gate. Math / local-file / cross-session (L2-disk) paths are intentionally removed for this build.
    ///   1. explicit "remember/save X" + a specific value → persist+pin+ack (no generation)
    ///   2. specificity pin (cap handled by TieredMemory)
    ///   3. recall gate — BGE search over session + pins, BEFORE saving this turn; discard <0.62
    ///   4. web gate — no memory hit + a question + web on → DuckDuckGo→Wikipedia + injection guard
    ///   5. generate — SP-evict genOnce with the always-on persona; verbatim-quote template on a hit
    ///   6. post — refusal → high-temp retry; display normalization; then log this turn to session
    public func turnRF(_ userMsg: String, webEnabled: Bool = true,
                       options: SPModel.Options = .default) async -> TurnResult {
        // 1. explicit save
        if mcWantsPersist(userMsg), mcFactlike(userMsg), !(await specificSpans(userMsg)).isEmpty {
            mem.persist(userMsg); mem.pin(userMsg)
            stitch(userMsg, "Got it — saved."); return ack("Got it — saved.", "save")
        }
        // 2. recall gate — search BEFORE saving/pinning this turn (spec: avoid self-match).
        // FULLY OFFLINE build: Web tier removed — recall is in-session memory (session + pins) only;
        // everything else answers from the model's own knowledge.
        var (src, chunks) = await mem.retrievePersonal(userMsg)
        if !chunks.isEmpty {
            let qv = await bge.encode(userMsg, isQuery: true)
            var best: Float = 0
            for c in chunks { best = max(best, cosine(await bge.encode(c, isQuery: false), qv)) }
            if best < 0.62 { chunks = []; src = nil }     // low-confidence → normal answer
        }
        // 3. specificity pin — AFTER retrieval (so the current turn can't self-match), and only
        // DECLARATIVE values (pinning a question like "capital of Australia?" makes every later
        // "capital of X" match it).
        if !mcIsQuestion(userMsg), !(await specificSpans(userMsg)).isEmpty { mem.pin(userMsg) }
        // 4. generation — RAW. Recall wiring above is the only thing we do; otherwise just let the
        // model generate WITH CoT open (forceThink:true → real <think> reasoning, which the 1.5B R1
        // needs for calculation). No persona, no retry loop, no salvage forcing, no rewriting — bound
        // length with options.genLen. Only recalled chunks get a short "read it verbatim" lead.
        let aug: String
        if !chunks.isEmpty {
            aug = "Context (retrieved from \(src ?? "")): \(chunks.joined(separator: " ; "))\n\n"
                + "Question: \(userMsg)\nRead the matching value from the Context and answer with it verbatim."
        } else {
            aug = userMsg
        }
        let t0 = Date()
        let g = sp.genOnce(aug, state: &genState, options: options, firstTurn: genState.gen.isEmpty,
                           policy: DecodePolicy(k: 1_000_000_000), forceThink: true,
                           salvage: "", salvageBudget: 120)   // only closes a still-open think; no prompt hacks
        lastBody = g.body
        // artifact cleanup only (leaked special tokens / \boxed wrapper) — content untouched.
        let answer = normalizeDisplay(g.answer)
        let secs = Date().timeIntervalSince(t0)
        // 6. log this turn to session AFTER retrieval — DECLARATIVE statements only. Logging a
        // QUESTION pollutes recall: a later "capital of Australia?" matches a stored "capital of
        // France?" at ≥0.62 and injects the wrong answer. Questions/chitchat continuity ride genState.
        if !mcIsQuestion(userMsg) { mem.remember_session(userMsg) }
        if !answer.isEmpty {
            convTurns.append((user: userMsg, assistant: answer))
            if convTurns.count > 8 { convTurns.removeFirst(convTurns.count - 8) }
        }
        let nt = sp.tokenizer.encode(text: answer, addSpecialTokens: false).count
        return TurnResult(intent: "rf", answer: answer, source: src, injected: chunks, acked: false,
                          tokens: nt, tokensPerSecond: secs > 0 ? Double(nt) / secs : 0)
    }

    /// RECALL_V2 §4: DYNAMIC gate-threshold calibration for the running (e.g. 4-bit) base model.
    /// 4-bit quantisation shifts the gate logit DOWN by a roughly constant amount (separation/AUC
    /// unchanged), so a fixed fp32 threshold under-fires. At startup we run a handful of short needle
    /// dialogues through THIS model, generate the answer in measure-only mode (gate scored every chunk,
    /// never fired), and take each dialogue's PEAK gate score = the recall-position signal. The
    /// threshold is set to the `percentile`-th percentile of those peaks (10 → ~90% of recall moments
    /// exceed it ⇒ ~90% recall). Returns the chosen threshold (also applied to `recallV2`).
    @discardableResult
    public func calibrateRecallThreshold(percentile: Double = 10) async -> Float? {
        guard let rv = recallV2 else { return nil }
        // Buried-fact needle dialogues: fact stated, pushed past the 512 window by neutral filler, then
        // a question + a primed answer stub ending EXACTLY where the fact is emitted — a deterministic
        // recall position (avoids the bimodal floor of letting the model answer generically). The gate
        // (robust linear head) read at that position is the "想起位置でのゲートスコア".
        let facts: [(stmt: String, q: String, lead: String)] = [
            ("The vault code at the cabin is 7741.", "What is the vault code at the cabin?", "The vault code at the cabin is "),
            ("Dr. Sato's clinic extension is 3092.", "What is Dr. Sato's clinic extension?", "Dr. Sato's clinic extension is "),
            ("The drone weighed 12.6 kilograms at checkout.", "How much did the drone weigh?", "The drone weighed "),
            ("Our booth number at the expo is C-48.", "What is our booth number at the expo?", "Our booth number at the expo is "),
            ("The rare vinyl cost 9400 yen.", "How much did the rare vinyl cost?", "The rare vinyl cost "),
            ("The ferry to the island departs at 06:35.", "What time does the ferry to the island depart?", "The ferry to the island departs at "),
            ("The server rack model is RX-220.", "What model is the server rack?", "The server rack model is "),
            ("The recipe needs 480 grams of sugar.", "How many grams of sugar does the recipe need?", "The recipe needs "),
            ("We parked on level 5, spot 73.", "Which spot did we park in?", "We parked in spot "),
            ("The trailhead is at marker 26.", "Which marker is the trailhead at?", "The trailhead is at marker "),
        ]
        let filler = "We talked about the weather, which had been changing quickly all week, and about a "
            + "quiet ramen place near the station that does a good late bowl. Work stayed busy but "
            + "manageable, and the garden tomatoes were finally turning red after a slow start."
        var scores = [Float]()
        for f in facts {
            var ids = sp.tokenizer.encode(text: "<｜User｜>\(f.stmt)<｜Assistant｜>Noted, thanks.",
                                          addSpecialTokens: false)
            while ids.count < 900 {
                ids += sp.tokenizer.encode(text: "<｜end▁of▁sentence｜><｜User｜>\(filler)<｜Assistant｜>Got it.",
                                           addSpecialTokens: false)
            }
            let pseq = ids + sp.tokenizer.encode(
                text: "<｜end▁of▁sentence｜><｜User｜>\(f.q)<｜Assistant｜><think>\n\n</think>\n\n\(f.lead)",
                addSpecialTokens: false)
            _ = await rv.recall(seq: pseq, ignoreGate: true)   // sets rv.lastScore at the primed position
            scores.append(rv.lastScore)
        }
        guard !scores.isEmpty else { return nil }
        let s = scores.sorted()
        let rank = (percentile / 100.0) * Double(s.count - 1)
        let lo = Int(rank.rounded(.down)); let frac = Float(rank - Double(lo))
        let thr = lo + 1 < s.count ? s[lo] + frac * (s[lo + 1] - s[lo]) : s[lo]
        rv.threshOverride = thr
        lastCalibration = (peaks: s, thresh: thr)
        return thr
    }
    /// Last calibration result (peaks + chosen threshold), for logging/inspection.
    public private(set) var lastCalibration: (peaks: [Float], thresh: Float)? = nil

    /// Reset multi-turn state (new conversation).
    public func resetConversation() {
        genState = SPModel.GenState(); convTokens = []; convTurns = []; v2BlockEmb = []
        lastTriggerScore = 0; lastTriggerFired = false
        archive?.reset(); lastRecallTokens = 0
    }
}
