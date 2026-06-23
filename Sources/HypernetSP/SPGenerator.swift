import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom
import Tokenizers

/// Bounded SP-evict generator — Swift port of `sp_mlx.generate_sp` / `tiered_rag_mlx.ChatSession`.
///
/// Distant context is compressed into `Pooler.numSoftTokens` soft prompts; only a raw window of
/// `rw` recent tokens is kept verbatim. Each chunk the KV cache is trimmed back to the prompt and
/// re-prefilled with `[soft prompts | raw window]`, giving O(1) bounded KV regardless of length.
public final class SPModel {

    public let model: SPQwen2Model
    public let tokenizer: Tokenizer
    public let pooler: Pooler
    public let H: Int
    public let eos: Int
    public let embedDtype: DType
    public let thinkOpen: [Int]    // "<think>\n"
    public let thinkClose: [Int]   // "\n</think>\n\n"

    public init(model: SPQwen2Model, tokenizer: Tokenizer, pooler: Pooler, eos: Int) {
        self.model = model
        self.tokenizer = tokenizer
        self.pooler = pooler
        self.H = pooler.H
        self.eos = eos
        // dtype of the model's token embeddings (4-bit weights dequantize to this)
        let probe = model.embed(MLXArray([Int32(0)], [1, 1]))
        self.embedDtype = probe.dtype
        self.thinkOpen = tokenizer.encode(text: "<think>\n", addSpecialTokens: false)
        self.thinkClose = tokenizer.encode(text: "\n</think>\n\n", addSpecialTokens: false)
    }

    // MARK: - loading

    public struct Paths: Sendable {
        public let modelDir: URL          // fft_mlx4 (MLX 4-bit Qwen2)
        public let poolerWeights: URL     // pooler.safetensors
        public let poolerConfig: URL      // pooler_config.json
        public init(modelDir: URL, poolerWeights: URL, poolerConfig: URL) {
            self.modelDir = modelDir
            self.poolerWeights = poolerWeights
            self.poolerConfig = poolerConfig
        }
    }

    /// Select the MLX compute device. The iOS Simulator's Metal cannot initialize MLX's GPU
    /// (`mlx::core::metal::Device` aborts), so validation there runs on CPU — numerically
    /// equivalent. On a real device (or macOS) use `.gpu`. Controlled by `SP_DEVICE=cpu|gpu`,
    /// defaulting to GPU.
    public static func configureDevice() {
        switch ProcessInfo.processInfo.environment["SP_DEVICE"]?.lowercased() {
        case "cpu": MLX.Device.setDefault(device: .cpu)
        case "gpu": MLX.Device.setDefault(device: .gpu)
        default: break
        }
    }

    public static func load(_ paths: Paths) async throws -> SPModel {
        configureDevice()
        // model config + weights (4-bit affine, group 64)
        let cfgData = try Data(contentsOf: paths.modelDir.appendingPathComponent("config.json"))
        let qcfg = try JSONDecoder().decode(SPQwen2Configuration.self, from: cfgData)
        let qwen = SPQwen2Model(qcfg)
        try loadWeights(
            modelDirectory: paths.modelDir,
            model: qwen,
            quantization: .init(groupSize: 64, bits: 4))

        let pooler = try Pooler(weightsURL: paths.poolerWeights, configURL: paths.poolerConfig)
        let tok = try await AutoTokenizer.from(modelFolder: paths.modelDir, strict: false)

        // eos: tokenizer's eos if available, else DeepSeek <｜end▁of▁sentence｜> = 151643
        let eos = tok.eosTokenId ?? 151643
        return SPModel(model: qwen, tokenizer: tok, pooler: pooler, eos: eos)
    }

    // MARK: - options / result

    public struct Options: Sendable {
        public var genLen: Int = 2000
        public var temp: Float = 0.6
        public var topP: Float = 0.0
        public var rw: Int = 1024
        public var chunk: Int = 64          // C
        public var maxD: Int = 4096
        public var chat: Bool = false
        public var greedy: Bool = false     // deterministic argmax (for validation)
        public var seed: UInt64 = 0
        /// Checked at the start of every chunk. Return false to stop generation early and return
        /// the partial result — e.g. when an iOS app goes to the background (the GPU rejects work
        /// submitted off-foreground, which would otherwise abort the process).
        public var shouldContinue: (@Sendable () -> Bool)? = nil
        public init() {}
        public static let `default` = Options()
    }

    public struct Result {
        public let text: String
        public let tokens: [Int]
        public let eosHit: Bool
        public let evicts: Int
        public let maxKept: Int
        public let seconds: Double
        public var tokensPerSecond: Double { seconds > 0 ? Double(tokens.count) / seconds : 0 }
    }

    /// Result of `continueStream` — a bounded-KV continuation of an arbitrary token stream.
    public struct StreamResult {
        public let newTokens: [Int]
        public let text: String
        public let eosHit: Bool
        public let evicts: Int
        public let maxKept: Int       // largest compressed-history size seen (≤ maxD)
        public let keptFinal: Int     // final compressed-history size (the bounded working set)
        public let streamLen: Int     // total conversation length (grows unbounded across turns)
        public let seconds: Double
        public var tokensPerSecond: Double { seconds > 0 ? Double(newTokens.count) / seconds : 0 }
    }

    // MARK: - generation

    private func embIds(_ ids: ArraySlice<Int>) -> MLXArray {
        if ids.isEmpty { return MLXArray.zeros([1, 0, H], dtype: embedDtype) }
        let arr = MLXArray(ids.map { Int32($0) }, [1, ids.count])
        return model.embed(arr)
    }
    private func embIds(_ ids: [Int]) -> MLXArray { embIds(ids[...]) }

    public func generate(prompt: String, options: Options = .default) -> Result {
        MLXRandom.seed(options.seed)
        let rw = options.rw, C = options.chunk, maxD = options.maxD, genLen = options.genLen
        var evicts = 0, maxKept = 0

        func evict(_ kept: [Int]) -> [Int] {
            maxKept = max(maxKept, kept.count)
            guard maxD > 0, kept.count > maxD else { return kept }
            let (_, mass) = pooler.forwardWithMass(embIds(kept).asType(.float32))
            let m = mass[0].asArray(Float.self)                       // (L,)
            let order = (0 ..< m.count).sorted { m[$0] > m[$1] }      // desc by mass
            let top = order.prefix(maxD).sorted()                    // keep order
            evicts += 1
            return top.map { kept[$0] }
        }

        let text = options.chat ? "<｜User｜>\(prompt)<｜Assistant｜>" : prompt
        let qIds = tokenizer.encode(text: text, addSpecialTokens: true)

        let cache = model.newCache(parameters: nil)                  // [KVCacheSimple]
        let prefill = model.callAsFunction(
            MLXArray(qIds.map { Int32($0) }, [1, qIds.count]), cache: cache)
        eval(prefill)                                                // materialize prompt KV
        let MQ = cache[0].offset

        var gen = [Int](); var kept = [Int](); var absorbed = 0; var done = false
        let t0 = Date()
        let nChunks = (genLen + C - 1) / C

        outer: for _ in 0 ..< nChunks {
            if let sc = options.shouldContinue, !sc() { break outer }   // app backgrounded → stop cleanly
            let c0 = gen.count
            let R = min(c0, rw)
            let ndEnd = c0 - R
            if ndEnd > absorbed {
                kept.append(contentsOf: gen[absorbed ..< ndEnd])
                absorbed = ndEnd
                kept = evict(kept)
            }
            let sp = pooler.forward(embIds(kept).asType(.float32)).asType(embedDtype)
            var block = sp
            if R > 0 {
                block = concatenated([sp, embIds(gen[(c0 - R) ..< c0])], axis: 1)
            }
            for c in cache { c.trim(c.offset - MQ) }
            var logits = model.callAsFunction(inputEmbeddings: block, cache: cache)
            var last = logits[0..., -1, 0...]                         // (1, vocab)
            eval(last)                                                // materialize KV; bound graph

            let inner = min(C, genLen - c0)
            for _ in 0 ..< inner {
                let nxt = sample(last, temp: options.temp, topP: options.topP, greedy: options.greedy)
                gen.append(nxt)
                if nxt == eos { done = true; break }
                logits = model.callAsFunction(inputEmbeddings: embIds([nxt]), cache: cache)
                last = logits[0..., -1, 0...]
                eval(last)                                            // materialize cache each step
            }
            if done { break outer }
        }

        let secs = Date().timeIntervalSince(t0)
        let out = tokenizer.decode(tokens: gen)
        return Result(text: out, tokens: gen, eosHit: done,
                      evicts: evicts, maxKept: maxKept, seconds: secs)
    }

    /// Bounded SP-evict conversation state (the evictable token stream + compressed history),
    /// mirroring `AppSession.gen/kept/absorbed`.
    public struct GenState { public var gen: [Int] = []; public var kept: [Int] = []; public var absorbed: Int = 0; public init() {} }

    /// Faithful port of canonical `AppSession._gen_once`: one bounded SP-evict generation that
    /// forces `<think>`, switches to greedy after `</think>` (DecodePolicy two-phase temp),
    /// force-closes a converged/looping think stream, and runs the 2-pass "Final answer:" salvage
    /// if the think never closed. Updates `state` (the bounded conversation). Returns (answer, body).
    public func genOnce(_ aug: String, state: inout GenState, options: Options, firstTurn: Bool,
                        policy: DecodePolicy = DecodePolicy(),
                        recIds: [Int] = [], forceThink: Bool = true,
                        salvage: String = "Final answer: ", salvageBudget: Int = 48,
                        tempOverride: Float? = nil, profileIds: [Int]? = nil,
                        thinkCap: Int = 0) -> (answer: String, body: String) {
        let rw = options.rw, C = options.chunk, maxD = options.maxD
        let cap = options.genLen
        var gen = state.gen, kept = state.kept, absorbed = state.absorbed

        func evict(_ k: [Int]) -> [Int] {
            guard maxD > 0, k.count > maxD else { return k }
            let (_, mass) = pooler.forwardWithMass(embIds(k).asType(.float32))
            let m = mass[0].asArray(Float.self)
            let order = (0 ..< m.count).sorted { m[$0] > m[$1] }
            return order.prefix(maxD).sorted().map { k[$0] }
        }
        func logitsFor(emb: MLXArray) -> MLXArray {
            let l = model.callAsFunction(inputEmbeddings: emb, cache: cache)
            let last = l[0..., -1, 0...]; eval(last); return last
        }

        let prefix = (firstTurn ? "" : "<｜end▁of▁sentence｜>") + "<｜User｜>\(aug)<｜Assistant｜>"
        // #20: non-compute turns get a PRE-CLOSED empty think — leaving think out entirely isn't
        // enough (the FFT model re-opens its own <think>, drafts inside, emits a self-review). The
        // pre-closed think pins it to answer the message directly (fixes greetings regurgitating the
        // prior turn). Compute turns keep the open think for reasoning.
        let thinkPart = forceThink ? thinkOpen
            : tokenizer.encode(text: "<think>\n\n</think>\n\n", addSpecialTokens: false)
        var feed = tokenizer.encode(text: prefix, addSpecialTokens: false) + thinkPart
        let start = gen.count
        var fi = 0, new = 0, thinkTok = 0
        let cache = model.newCache(parameters: nil)
        let bos = tokenizer.bosTokenId ?? 151646
        // OPERATIONS_SPEC FIX3: a persona/profile sits in the NEVER-EVICTED MQ prefix so every turn
        // knows who it is talking to and won't refuse/collapse on benign chitchat. `profileIds` must
        // start with BOS; nil → bare BOS (prior behaviour).
        let prime = profileIds ?? [bos]
        _ = model.callAsFunction(MLXArray(prime.map { Int32($0) }, [1, prime.count]), cache: cache)
        let MQ = cache[0].offset
        var inThink = forceThink, done = false

        while !done {
            let c0 = gen.count, R = min(c0, rw), ndEnd = c0 - R
            if ndEnd > absorbed { kept.append(contentsOf: gen[absorbed ..< ndEnd]); absorbed = ndEnd; kept = evict(kept) }
            for c in cache { c.trim(c.offset - MQ) }
            // Block layout = [SP | rec | recent window]. While `kept` is empty, don't prepend the SP
            // block (pooler(∅)=sp0 is 32 noise soft-prompts that distract the model — fix 2026-06-12);
            // the Block-Recall `rec` ids ARE injected even with empty SP (recall is independent of SP).
            var parts = [MLXArray]()
            if !kept.isEmpty { parts.append(pooler.forward(embIds(kept).asType(.float32)).asType(embedDtype)) }
            if !recIds.isEmpty { parts.append(embIds(recIds)) }
            if R > 0 { parts.append(embIds(gen[(c0 - R) ..< c0])) }
            var last: MLXArray
            if parts.isEmpty {
                last = MLXArray.zeros([1, 1])   // first chunk, nothing to prefill: forced feed drives it
            } else {
                last = logitsFor(emb: parts.count == 1 ? parts[0] : concatenated(parts, axis: 1))
            }
            for _ in 0 ..< C {
                var t: Int
                if fi < feed.count { t = feed[fi]; fi += 1 }
                else {
                    if let sc = options.shouldContinue, !sc() { done = true; break }
                    let base = tempOverride ?? options.temp
                    let T = forceThink ? policy.temp(inThink, base) : base
                    t = sample(last, temp: T, topP: 0, greedy: false)
                    if t == eos { done = true; break }
                    new += 1
                    if new >= cap { done = true; break }
                }
                gen.append(t)
                if inThink, fi >= feed.count { thinkTok += 1 }
                if inThink, tokenizer.decode(tokens: Array(gen.suffix(8))).contains("</think>") { inThink = false }
                // hard CoT cap: think running past `thinkCap` tokens without closing → force-close
                // </think> and answer (no "Final answer:" priming, just close).
                if inThink, thinkCap > 0, thinkTok >= thinkCap {
                    feed += tokenizer.encode(text: "\n</think>\n\n", addSpecialTokens: false)
                    inThink = false
                } else if inThink, fi >= feed.count, policy.noteText(tokenizer.decode(tokens: Array(gen[start...]))) {
                    feed += tokenizer.encode(text: "\n</think>\n\nFinal answer: ", addSpecialTokens: false)
                    inThink = false
                }
                last = logitsFor(emb: embIds([t]))
            }
            if done { break }
        }

        func bodyOf() -> String {
            let full = tokenizer.decode(tokens: Array(gen[start...]))
            if let r = full.range(of: "<｜Assistant｜>") { return String(full[r.upperBound...]) }
            return full
        }
        var body = bodyOf()
        if !body.contains("</think>") || isEmptyAnswer(extractAnswer(body)) {
            // 2-pass salvage, INTENT-AWARE: compute primes "Final answer: " (a bare number) on a
            // small budget; non-compute closes the think and answers naturally on a larger budget.
            var last = MLXArray.zeros([1, 1])
            for t in tokenizer.encode(text: "\n</think>\n\n" + salvage, addSpecialTokens: false) {
                gen.append(t); last = logitsFor(emb: embIds([t]))
            }
            for _ in 0 ..< salvageBudget {
                let t = last[0].argMax().item(Int.self)
                if t == eos { break }
                gen.append(t); last = logitsFor(emb: embIds([t]))
            }
            body = bodyOf()
        }

        state.gen = gen; state.kept = kept; state.absorbed = absorbed
        var ans = extractAnswer(body)
        if ans.contains("Final answer:") {
            ans = String(ans.components(separatedBy: "Final answer:").last ?? ans).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return (String(ans.prefix(4000)), body)
    }

    /// RECALL_V2 §2: `genOnce` with the IN-GENERATION recall gate. Identical SP-evict decode, but at
    /// each chunk rebuild the gate (`RecallV2`) is scored from the q captured during that chunk's
    /// prefill — a per-position classifier, not a single pre-gen check. On the first fire it retrieves
    /// the top-2 query-relevant blocks (from `recallBlocks` = the prior conversation) via the bridge
    /// head, re-prefills the chunk with them injected ([SP|rec|window]), and keeps them for the rest of
    /// generation. `blockEmb` caches block BGE vectors across calls. Returns the recall telemetry too.
    public func genOnceRecall(_ aug: String, state: inout GenState, options: Options, firstTurn: Bool,
                              recall: RecallV2, recallBlocks: [[Int]], blockEmb: inout [[Float]],
                              policy: DecodePolicy = DecodePolicy(), forceThink: Bool = false,
                              salvage: String = "", salvageBudget: Int = 200, tempOverride: Float? = nil,
                              measureOnly: Bool = false)
        async -> (answer: String, body: String, fired: Bool, recallTokens: Int, maxScore: Float, recIds: [Int]) {
        let rw = options.rw, C = options.chunk, maxD = options.maxD
        let cap = options.genLen
        var gen = state.gen, kept = state.kept, absorbed = state.absorbed

        func evict(_ k: [Int]) -> [Int] {
            guard maxD > 0, k.count > maxD else { return k }
            let (_, mass) = pooler.forwardWithMass(embIds(k).asType(.float32))
            let m = mass[0].asArray(Float.self)
            let order = (0 ..< m.count).sorted { m[$0] > m[$1] }
            return order.prefix(maxD).sorted().map { k[$0] }
        }
        func logitsFor(emb: MLXArray) -> MLXArray {
            let l = model.callAsFunction(inputEmbeddings: emb, cache: cache)
            let last = l[0..., -1, 0...]; eval(last); return last
        }

        let prefix = (firstTurn ? "" : "<｜end▁of▁sentence｜>") + "<｜User｜>\(aug)<｜Assistant｜>"
        let thinkPart = forceThink ? thinkOpen
            : tokenizer.encode(text: "<think>\n\n</think>\n\n", addSpecialTokens: false)
        var feed = tokenizer.encode(text: prefix, addSpecialTokens: false) + thinkPart
        let start = gen.count
        var fi = 0, new = 0
        let cache = model.newCache(parameters: nil)
        let bos = tokenizer.bosTokenId ?? 151646
        _ = model.callAsFunction(MLXArray([Int32(bos)], [1, 1]), cache: cache)
        let MQ = cache[0].offset
        var inThink = forceThink, done = false

        var rec: [Int] = []
        var fired = false
        var maxScore: Float = -1e9
        let gLayers = recall.layersList

        while !done {
            let c0 = gen.count, R = min(c0, rw), ndEnd = c0 - R
            if ndEnd > absorbed { kept.append(contentsOf: gen[absorbed ..< ndEnd]); absorbed = ndEnd; kept = evict(kept) }
            for c in cache { c.trim(c.offset - MQ) }

            func buildParts() -> [MLXArray] {
                var parts = [MLXArray]()
                if !kept.isEmpty { parts.append(pooler.forward(embIds(kept).asType(.float32)).asType(embedDtype)) }
                if !rec.isEmpty { parts.append(embIds(rec)) }
                if R > 0 { parts.append(embIds(gen[(c0 - R) ..< c0])) }
                return parts
            }
            var parts = buildParts()
            var last: MLXArray
            if parts.isEmpty {
                last = MLXArray.zeros([1, 1])
            } else {
                let block = parts.count == 1 ? parts[0] : concatenated(parts, axis: 1)
                let blockLen = block.dim(1)
                let (logits, qd) = model.callCapturing(inputEmbeddings: block, cache: cache, layers: gLayers)
                last = logits[0..., -1, 0...]; eval(last)
                // Normal mode: score only until it fires (sticky). Calibration (measureOnly): score
                // every chunk to find the per-dialogue peak, but never fire/inject.
                if !recallBlocks.isEmpty, R > 0, measureOnly || !fired {
                    let qflat = recall.qFlat(from: qd, pos: blockLen - 1, seqLen: blockLen)
                    let score = recall.gateScore(qflat)
                    maxScore = Swift.max(maxScore, score)
                    if !measureOnly, !fired, score > recall.effectiveThresh {
                        fired = true
                        rec = await recall.retrieve(qflat: qflat, blocks: recallBlocks, blockEmb: &blockEmb)
                        for c in cache { c.trim(c.offset - MQ) }
                        parts = buildParts()
                        last = logitsFor(emb: parts.count == 1 ? parts[0] : concatenated(parts, axis: 1))
                    }
                }
            }
            for _ in 0 ..< C {
                var t: Int
                if fi < feed.count { t = feed[fi]; fi += 1 }
                else {
                    if let sc = options.shouldContinue, !sc() { done = true; break }
                    let base = tempOverride ?? options.temp
                    let T = forceThink ? policy.temp(inThink, base) : base
                    t = sample(last, temp: T, topP: 0, greedy: false)
                    if t == eos { done = true; break }
                    new += 1
                    if new >= cap { done = true; break }
                }
                gen.append(t)
                if inThink, tokenizer.decode(tokens: Array(gen.suffix(8))).contains("</think>") { inThink = false }
                if inThink, fi >= feed.count, policy.noteText(tokenizer.decode(tokens: Array(gen[start...]))) {
                    feed += tokenizer.encode(text: "\n</think>\n\nFinal answer: ", addSpecialTokens: false)
                    inThink = false
                }
                last = logitsFor(emb: embIds([t]))
            }
            if done { break }
        }

        func bodyOf() -> String {
            let full = tokenizer.decode(tokens: Array(gen[start...]))
            if let r = full.range(of: "<｜Assistant｜>") { return String(full[r.upperBound...]) }
            return full
        }
        var body = bodyOf()
        if !body.contains("</think>") || isEmptyAnswer(extractAnswer(body)) {
            var last = MLXArray.zeros([1, 1])
            for t in tokenizer.encode(text: "\n</think>\n\n" + salvage, addSpecialTokens: false) {
                gen.append(t); last = logitsFor(emb: embIds([t]))
            }
            for _ in 0 ..< salvageBudget {
                let t = last[0].argMax().item(Int.self)
                if t == eos { break }
                gen.append(t); last = logitsFor(emb: embIds([t]))
            }
            body = bodyOf()
        }

        state.gen = gen; state.kept = kept; state.absorbed = absorbed
        var ans = extractAnswer(body)
        if ans.contains("Final answer:") {
            ans = String(ans.components(separatedBy: "Final answer:").last ?? ans).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return (String(ans.prefix(4000)), body, fired, rec.count, maxScore, rec)
    }

    /// FAITHFUL port of `recall_gen.py reply()` (trigger_experiment/ondevice_recall) — the owner's
    /// reference decode for on-device recall. Differs from genOnce on purpose:
    ///   • NO `<think>` scaffolding — `feed` ends at `<｜Assistant｜>`; R1 opens/reasons/closes think
    ///     itself and commits to recalling the fact (pre-closing the think kills the recall state).
    ///   • PURE GREEDY (argmax), no temperature/top-p, NO salvage.
    ///   • Gate scored every C=64-token chunk; on fire, BGE bridge retrieves top-k and `rec_emb` is
    ///     injected into the prefix (`[SP | rec | window]`), updated on each subsequent fire.
    ///   • Fixed threshold (−3.5 for 4-bit); no mass-eviction (kept grows; SP compresses all of it).
    /// `feed` = full history + `<｜User｜>{q}<｜Assistant｜>` token ids. Returns the generated answer
    /// tokens, fire count, and the pulled block token-ids (for retrieval-hit scoring).
    public func replyRecall(feed: [Int], recall rv: RecallV2, thresh: Float,
                            rw: Int = 512, chunk: Int = 64, turnCap: Int = 110, recallK: Int = 2)
        async -> (genTokens: [Int], answer: String, nFire: Int, pulls: [[Int]], maxScore: Float) {
        let C = chunk
        var gen = [Int](); var kept = [Int](); var absorbed = 0
        var fi = 0, newtok = 0; var done = false; var nFire = 0
        var pulls = [[Int]](); var maxScore: Float = -1e9
        var recEmb: MLXArray? = nil
        var blockEmb = [[Float]]()
        let gLayers = rv.layersList

        let cache = model.newCache(parameters: nil)
        let bos = tokenizer.bosTokenId ?? eos
        let prime = model.callAsFunction(MLXArray([Int32(bos)], [1, 1]), cache: cache)
        let primeLast = prime[0..., -1, 0...]; eval(primeLast)
        let MQ = cache[0].offset

        func buildParts(_ R: Int, _ c0: Int) -> [MLXArray] {
            var parts = [MLXArray]()
            if !kept.isEmpty { parts.append(pooler.forward(embIds(kept).asType(.float32)).asType(embedDtype)) }
            if let re = recEmb { parts.append(re) }
            if R > 0 { parts.append(embIds(gen[(c0 - R) ..< c0])) }
            return parts
        }

        while !done {
            let c0 = gen.count, R = min(c0, rw), nd = c0 - R
            if nd > absorbed { kept.append(contentsOf: gen[absorbed ..< nd]); absorbed = nd }
            for c in cache { c.trim(c.offset - MQ) }
            var parts = buildParts(R, c0)
            var last: MLXArray
            if parts.isEmpty {
                last = primeLast
            } else {
                let block = parts.count == 1 ? parts[0] : concatenated(parts, axis: 1)
                let blockLen = block.dim(1)
                let (logits, qd) = model.callCapturing(inputEmbeddings: block, cache: cache, layers: gLayers)
                last = logits[0..., -1, 0...]; eval(last)
                // gate at the current decode position (every chunk), retrieve+inject on fire.
                let cands = rv.makeBlocksWithBuf(kept)
                if !cands.isEmpty {
                    let qflat = rv.qFlat(from: qd, pos: blockLen - 1, seqLen: blockLen)
                    let s = rv.gateScore(qflat)
                    maxScore = Swift.max(maxScore, s)
                    if s > thresh {
                        let rid = await rv.retrieve(qflat: qflat, blocks: cands, blockEmb: &blockEmb, topk: recallK)
                        if !rid.isEmpty {
                            recEmb = embIds(rid); nFire += 1; pulls.append(rid)
                            for c in cache { c.trim(c.offset - MQ) }
                            parts = buildParts(R, c0)
                            let b2 = parts.count == 1 ? parts[0] : concatenated(parts, axis: 1)
                            last = logitsFor(emb: b2, cache: cache)
                        }
                    }
                }
            }
            for _ in 0 ..< C {
                var t: Int
                if fi < feed.count { t = feed[fi]; fi += 1 }
                else {
                    t = last[0].argMax().item(Int.self)          // PURE GREEDY
                    if t == eos || newtok >= turnCap { done = true; break }
                    newtok += 1
                }
                gen.append(t)
                last = logitsFor(emb: embIds([t]), cache: cache)
            }
            if done { break }
        }
        // THINK-CLOSE + ANSWER EXTRACTION (production form): if the model never closed </think>,
        // force-close it and let it state the reply; then return ONLY the post-think text (not the raw
        // chain-of-thought). The recall already injected the fact during the think, so the closed
        // answer can state it.
        var full = tokenizer.decode(tokens: Array(gen.suffix(max(0, gen.count - feed.count))))
        if !full.contains("</think>") {
            var l2 = MLXArray.zeros([1, 1])
            for t in tokenizer.encode(text: "\n</think>\n\n", addSpecialTokens: false) {
                gen.append(t); l2 = logitsFor(emb: embIds([t]), cache: cache)
            }
            for _ in 0 ..< 80 {
                let t = l2[0].argMax().item(Int.self)
                if t == eos { break }
                gen.append(t); l2 = logitsFor(emb: embIds([t]), cache: cache)
            }
            full = tokenizer.decode(tokens: Array(gen.suffix(max(0, gen.count - feed.count))))
        }
        let ans: String
        if let r = full.range(of: "</think>", options: .backwards) {
            ans = String(full[r.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            ans = full.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return (Array(gen.suffix(max(0, gen.count - feed.count))), ans, nFire, pulls, maxScore)
    }

    private func logitsFor(emb: MLXArray, cache: [KVCache]) -> MLXArray {
        let l = model.callAsFunction(inputEmbeddings: emb, cache: cache)
        let last = l[0..., -1, 0...]; eval(last); return last
    }

    /// Isolated **clean-quote** answer (port of `tiered_rag_mlx._clean_quote`): generate from ONLY this
    /// prompt — no soft prompt, no conversation history — let it think briefly, then **force-close**
    /// `</think>` and generate the answer, so the result is never trapped inside `<think>`. If the forced
    /// answer is empty/degenerate, fall back to the last `\boxed{}` the reasoning reached. Used for
    /// recall/lookup (the retrieved context is short, so plain KV — no SP-evict needed).
    public func cleanQuote(_ prompt: String, temp: Float = 0.2, thinkBudget: Int = 200,
                           ansBudget: Int = 80, seed: UInt64 = 0,
                           shouldContinue: (@Sendable () -> Bool)? = nil) -> String {
        MLXRandom.seed(seed)
        let cache = model.newCache(parameters: nil)
        func prefill(_ ids: [Int]) -> MLXArray {
            let logits = model.callAsFunction(MLXArray(ids.map { Int32($0) }, [1, ids.count]), cache: cache)
            let last = logits[0..., -1, 0...]; eval(last); return last
        }
        func sampleRun(_ last0: MLXArray, _ budget: Int) -> [Int] {
            var last = last0; var out = [Int](); var prev = -1; var rep = 0
            for _ in 0 ..< budget {
                if let sc = shouldContinue, !sc() { break }
                let t = sample(last, temp: temp, topP: 0, greedy: false)
                if t == eos { break }
                rep = (t == prev) ? rep + 1 : 0
                if rep >= 5 { break }
                prev = t; out.append(t)
                let logits = model.callAsFunction(MLXArray([Int32(t)], [1, 1]), cache: cache)
                last = logits[0..., -1, 0...]; eval(last)
            }
            return out
        }
        let promptIds = tokenizer.encode(text: "<｜User｜>\(prompt)<｜Assistant｜>", addSpecialTokens: true)
            + thinkOpen
        var last = prefill(promptIds)
        let think = sampleRun(last, thinkBudget)
        last = prefill(thinkClose)                       // FORCE-close </think>
        var ans = tokenizer.decode(tokens: sampleRun(last, ansBudget))
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if isEmptyAnswer(ans) || looksDegenerate(ans) {
            if let box = lastBoxed(tokenizer.decode(tokens: think)) { ans = box }
        }
        return ans
    }

    /// Continue an arbitrary token stream (e.g. a whole multi-turn conversation) with bounded KV.
    ///
    /// Unlike `generate`, there is **no permanent prompt prefill** — the entire `stream` is part of
    /// the evictable history. Each chunk: tokens beyond the raw window are pooled into 32 soft prompts
    /// (mass-evicted to `maxD`), the KV cache is rebuilt from `[SP | raw window]`, and `chunk` tokens
    /// are decoded. So the conversation length can grow without bound while the KV stays O(rw + maxD-pool)
    /// — this is the "infinite rally" property: old turns are compressed, not kept verbatim in KV.
    public func continueStream(_ stream: [Int], maxNew: Int, options: Options = .default) -> StreamResult {
        MLXRandom.seed(options.seed)
        let rw = options.rw, C = options.chunk, maxD = options.maxD
        var evicts = 0, maxKept = 0

        func evict(_ kept: [Int]) -> [Int] {
            maxKept = max(maxKept, kept.count)
            guard maxD > 0, kept.count > maxD else { return kept }
            let (_, mass) = pooler.forwardWithMass(embIds(kept).asType(.float32))
            let m = mass[0].asArray(Float.self)
            let order = (0 ..< m.count).sorted { m[$0] > m[$1] }
            let top = order.prefix(maxD).sorted()
            evicts += 1
            return top.map { kept[$0] }
        }

        var gen = stream
        let startLen = stream.count
        var kept = [Int](); var absorbed = 0; var done = false
        let cache = model.newCache(parameters: nil)
        let t0 = Date()
        var produced = 0
        var prevTok = -1, rep = 0           // degeneration loop guard (port of rep>=6)

        outer: while produced < maxNew {
            if let sc = options.shouldContinue, !sc() { break outer }
            let c0 = gen.count
            let R = min(c0, rw)
            let ndEnd = c0 - R
            if ndEnd > absorbed {
                kept.append(contentsOf: gen[absorbed ..< ndEnd]); absorbed = ndEnd
                kept = evict(kept)
            }
            let sp = pooler.forward(embIds(kept).asType(.float32)).asType(embedDtype)
            var block = sp
            if R > 0 { block = concatenated([sp, embIds(gen[(c0 - R) ..< c0])], axis: 1) }
            for c in cache { c.trim(c.offset) }                        // rebuild [SP|window] each chunk
            var logits = model.callAsFunction(inputEmbeddings: block, cache: cache)
            var last = logits[0..., -1, 0...]
            eval(last)
            let step = min(C, maxNew - produced)
            for _ in 0 ..< step {
                let nxt = sample(last, temp: options.temp, topP: options.topP, greedy: options.greedy)
                gen.append(nxt); produced += 1
                if nxt == eos { done = true; break }
                rep = (nxt == prevTok) ? rep + 1 : 0; prevTok = nxt
                if rep >= 6 { done = true; break }                     // repetition loop → stop
                logits = model.callAsFunction(inputEmbeddings: embIds([nxt]), cache: cache)
                last = logits[0..., -1, 0...]
                eval(last)
            }
            if done { break }
        }

        let secs = Date().timeIntervalSince(t0)
        let newTokens = Array(gen[startLen...])
        return StreamResult(
            newTokens: newTokens, text: tokenizer.decode(tokens: newTokens), eosHit: done,
            evicts: evicts, maxKept: maxKept, keptFinal: kept.count, streamLen: gen.count, seconds: secs)
    }

    private func sample(_ last: MLXArray, temp: Float, topP: Float, greedy: Bool) -> Int {
        // Near-zero temperature == argmax. DecodePolicy's greedy-after-</think> phase passes
        // temp=1e-4; routing that through categorical(logits / 1e-4) scales logits by 1e4, and
        // MLX's categorical is not max-stabilised like PyTorch's softmax → it overflows to a NaN
        // distribution and emits garbage ()1!!!!). PyTorch's softmax(logits/1e-4) is shift-stable
        // and collapses to the argmax, so argmax here is the faithful, overflow-safe equivalent.
        if greedy || temp < 1e-2 {
            return last[0].argMax().item(Int.self)
        }
        if topP > 0, topP < 1.0 {
            return sampleTopP(last, temp: temp, topP: topP)
        }
        let idx = MLXRandom.categorical(last * (1.0 / temp))
        return idx.item(Int.self)
    }

    private func sampleTopP(_ last: MLXArray, temp: Float, topP: Float) -> Int {
        let l = last[0] / temp
        let probs = softmax(l, axis: -1)
        let sortedIdx = argSort(-probs)              // ascending of -probs == descending probs
        let sp = probs[sortedIdx]
        let cum = cumsum(sp, axis: -1) - sp
        let keep = cum .<= MLXArray(topP)
        let masked = MLX.where(keep, log(maximum(sp, MLXArray(Float(1e-12)))), MLXArray(Float(-1e9)))
        // sample within the kept set, then map back to original token id
        let pick = MLXRandom.categorical(masked)     // index into sorted order
        let tokenId = sortedIdx[pick].item(Int.self)
        return tokenId
    }
}

/// Stateful multi-turn chat with bounded KV ("infinite rally").
///
/// Holds the whole conversation as a growing token list and replies via `SPModel.continueStream`,
/// so the conversation length is unbounded while the live KV stays O(rw + maxD-pool): distant turns
/// (and injected RAG context) are compressed into 32 soft prompts instead of filling the KV cache.
public final class ChatSession {
    public let model: SPModel
    public private(set) var tokens: [Int]   // full conversation (grows unbounded)
    public private(set) var turns: Int = 0

    private var bos: Int { model.tokenizer.bosTokenId ?? 151646 }

    public init(model: SPModel) {
        self.model = model
        self.tokens = [model.tokenizer.bosTokenId ?? 151646]
    }

    /// Append a user turn and generate the assistant reply, continuing the bounded stream.
    @discardableResult
    public func reply(to user: String, maxNew: Int = 512, options: SPModel.Options = .default)
        -> SPModel.StreamResult
    {
        // force the think phase like the reference (`<｜User｜>…<｜Assistant｜>` + `<think>\n`)
        let turn = model.tokenizer.encode(
            text: "<｜User｜>\(user)<｜Assistant｜>", addSpecialTokens: false) + model.thinkOpen
        tokens.append(contentsOf: turn)
        let r = model.continueStream(tokens, maxNew: maxNew, options: options)
        tokens.append(contentsOf: r.newTokens)
        turns += 1
        return r
    }

    public func reset() { tokens = [bos]; turns = 0 }
}
