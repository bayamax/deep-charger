import Foundation
import MLX
import MLXLMCommon
import Tokenizers

/// Swift port of `recall_kit` (trigger_experiment/ondevice_recall, the chitchat-fixed handoff):
/// gate (`recall_kit.gate.RecallGate`) + BGE bridge-head retriever (`recall_kit.retriever.BGEHead` /
/// `BGERetriever`) wired exactly like `recall_kit.runtime.RecallRuntime.prepare`/`decide`.
///
/// Per turn, from the production-available context `[BOS][SP(evicted)][window]` (the window already
/// ends with the just-arrived turn), it:
///   1. block-aligned evicts everything past the 512-token window into 128-token blocks,
///   2. captures the pre-RoPE q at the LAST window position (layers 8/14/20 → 4608),
///   3. scores the gate; if it fires, BGE-base-encodes each block's TEXT and a two-tower bridge head
///      projects (q 4608→128, doc 768→128) to rank — top-2 block token-ids are returned verbatim.
///
/// The gate's q-feature IS the retrieval query (free — already computed). State-free: stores only
/// token ids. Gate thresh −2.82 (90%-recall point); chat-recommended −1.5…−2.0.
public final class RecallV2 {
    private static let layers = [8, 14, 20]
    public static let window = 512, block = 128
    private let Hq = 12, D = 128            // qdim 4608 = 3 × Hq × D

    private let model: SPQwen2Model
    private let pooler: Pooler
    private let tokenizer: Tokenizer
    private let bge: Embedder               // BGE-BASE (768-dim) — the bridge head's key tower
    private let embedDtype: DType
    private let bos: Int

    // gate
    private let coef: [Float], mean: [Float], scale: [Float]
    private let intercept: Float
    public let thresh: Float
    /// Threshold actually used at runtime (defaults to the file's −2.82; override for chat to −1.5…−2.0).
    public var threshOverride: Float? = nil
    public private(set) var lastScore: Float = 0

    // bridge head (two-tower MLP, Linear→GELU→Linear)
    private let wq0w: [Float], wq0b: [Float], wq2w: [Float], wq2b: [Float]   // 4608→128→128
    private let wk0w: [Float], wk0b: [Float], wk2w: [Float], wk2b: [Float]   // 768→128→128

    public init(model: SPQwen2Model, pooler: Pooler, tokenizer: Tokenizer, bge: Embedder,
                embedDtype: DType, bos: Int, gateURL: URL, headURL: URL) throws {
        self.model = model; self.pooler = pooler; self.tokenizer = tokenizer; self.bge = bge
        self.embedDtype = embedDtype; self.bos = bos
        let g = try loadArrays(url: gateURL)
        guard let c = g["coef"], let b = g["intercept"], let m = g["mean"],
              let s = g["scale"], let t = g["thresh"] else { throw Err.missing }
        coef = c.asType(.float32).asArray(Float.self)
        mean = m.asType(.float32).asArray(Float.self)
        scale = s.asType(.float32).asArray(Float.self)
        intercept = b.asType(.float32).asArray(Float.self).first ?? 0
        thresh = t.asType(.float32).asArray(Float.self).first ?? 0
        let h = try loadArrays(url: headURL)
        func a(_ k: String) throws -> [Float] {
            guard let v = h[k] else { throw Err.missing }; return v.asType(.float32).asArray(Float.self)
        }
        wq0w = try a("Wq.0.weight"); wq0b = try a("Wq.0.bias")
        wq2w = try a("Wq.2.weight"); wq2b = try a("Wq.2.bias")
        wk0w = try a("Wk.0.weight"); wk0b = try a("Wk.0.bias")
        wk2w = try a("Wk.2.weight"); wk2b = try a("Wk.2.bias")
    }
    enum Err: Error { case missing }

    private func embIds(_ ids: [Int]) -> MLXArray {
        model.embed(MLXArray(ids.map { Int32($0) }, [1, ids.count]))
    }

    /// Diagnostic: for `seq`, find the block whose decoded text contains `value`, and report its rank
    /// under (a) the learned bridge head and (b) raw BGE-base cosine of the query TEXT. Separates a
    /// broken BGE-base encoder (true block ranks badly even by raw cosine) from a bridge/q issue.
    public func debugRetrieve(seq: [Int], value: String, queryText: String) async
        -> (nBlocks: Int, trueIdx: Int, bridgeRank: Int, rawRank: Int, gateScore: Float) {
        let nEvict = ((seq.count - Self.window) / Self.block) * Self.block
        let nB = nEvict / Self.block
        guard nB >= 1 else { return (0, -1, -1, -1, 0) }
        let window = Array(seq[nEvict...])
        let kept = Array(seq[0 ..< nEvict])
        let blocks = (0 ..< nB).map { Array(seq[$0 * Self.block ..< ($0 + 1) * Self.block]) }
        let cache = model.newCache(parameters: nil)
        _ = model.callAsFunction(MLXArray([Int32(bos)], [1, 1]), cache: cache)
        let sp = pooler.forward(embIds(kept).asType(.float32)).asType(embedDtype)
        _ = model.callAsFunction(inputEmbeddings: sp, cache: cache)
        let qc = model.captureProj(inputEmbeddings: embIds(window), cache: cache, layers: Self.layers, wantQ: true)
        let last = window.count - 1
        var qflat = [Float]()
        for l in Self.layers { let q = qc[l]!.reshaped(window.count, Hq, D); let ql = q[last].asType(.float32); eval(ql); qflat.append(contentsOf: ql.asArray(Float.self)) }
        var s = intercept
        for i in 0 ..< min(qflat.count, coef.count) { s += ((qflat[i] - mean[i]) / scale[i]) * coef[i] }
        // diagnostic: compare my extracted q distribution to the reference fixture (norm≈104, std≈1.53,
        // range≈[-20,22]). A gross mismatch ⇒ structural extraction bug; a match ⇒ position issue.
        let qn = qflat.map { $0 * $0 }.reduce(0, +).squareRoot()
        let qmean = qflat.reduce(0, +) / Float(qflat.count)
        let qstd = (qflat.map { ($0 - qmean) * ($0 - qmean) }.reduce(0, +) / Float(qflat.count)).squareRoot()
        print(String(format: "[SP] V2DBG qflat norm=%.1f std=%.3f min=%.2f max=%.2f", qn, qstd, qflat.min() ?? 0, qflat.max() ?? 0))
        let pq = tower(qflat, wq0w, wq0b, wq2w, wq2b)
        let qbge = await bge.encode(queryText, isQuery: false)
        var bridge = [Float](repeating: 0, count: nB), raw = [Float](repeating: 0, count: nB)
        var trueIdx = -1
        for b in 0 ..< nB {
            let text = tokenizer.decode(tokens: blocks[b])
            if trueIdx < 0, text.lowercased().contains(value.lowercased()) { trueIdx = b }
            let ke = await bge.encode(text, isQuery: false)
            let pk = tower(ke, wk0w, wk0b, wk2w, wk2b)
            var d1: Float = 0; for j in 0 ..< 128 { d1 += pq[j] * pk[j] }; bridge[b] = d1
            var d2: Float = 0; for j in 0 ..< min(ke.count, qbge.count) { d2 += ke[j] * qbge[j] }; raw[b] = d2
        }
        func rank(_ arr: [Float], _ idx: Int) -> Int {
            guard idx >= 0 else { return -1 }
            return arr.indices.sorted { arr[$0] > arr[$1] }.firstIndex(of: idx) ?? -1
        }
        return (nB, trueIdx, rank(bridge, trueIdx), rank(raw, trueIdx), s)
    }

    /// nn.GELU() exact: 0.5·x·(1+erf(x/√2)).
    private static func gelu(_ x: Float) -> Float { 0.5 * x * (1 + Float(erf(Double(x) / 1.4142135623730951))) }

    /// Linear(out,in)→GELU→Linear(d,d). w0 row-major [128,in], w2 [128,128].
    private func tower(_ x: [Float], _ w0: [Float], _ b0: [Float], _ w2: [Float], _ b2: [Float]) -> [Float] {
        let inDim = x.count
        var g = [Float](repeating: 0, count: 128)
        for j in 0 ..< 128 {
            var s = b0[j]
            let base = j * inDim
            for i in 0 ..< inDim { s += x[i] * w0[base + i] }
            g[j] = Self.gelu(s)
        }
        var out = [Float](repeating: 0, count: 128)
        for j in 0 ..< 128 {
            var s = b2[j]
            let base = j * 128
            for i in 0 ..< 128 { s += g[i] * w2[base + i] }
            out[j] = s
        }
        return out
    }

    // ── In-generation API (RECALL_V2 §3) ───────────────────────────────────────────────────
    // The gate is a per-position, in-generation classifier; these let the decode loop score it from
    // the q already captured during each chunk's prefill, and retrieve on fire — no extra forward.

    public var layersList: [Int] { Self.layers }
    public var effectiveThresh: Float { threshOverride ?? thresh }

    /// Flatten the captured pre-RoPE q at `pos` (layers 8/14/20) into the 4608 gate feature, in the
    /// trainer's order: layer-major, then head, then dim.
    public func qFlat(from captured: [Int: MLXArray], pos: Int, seqLen: Int) -> [Float] {
        var qflat = [Float](); qflat.reserveCapacity(Self.layers.count * Hq * D)
        for l in Self.layers {
            let q = captured[l]!.reshaped(seqLen, Hq, D)
            let ql = q[pos].asType(.float32); eval(ql)
            qflat.append(contentsOf: ql.asArray(Float.self))
        }
        return qflat
    }

    /// Gate logit ((q−mean)/scale·coef+intercept) for a precomputed q-feature.
    public func gateScore(_ qflat: [Float]) -> Float {
        var s = intercept
        for i in 0 ..< min(qflat.count, coef.count) { s += ((qflat[i] - mean[i]) / scale[i]) * coef[i] }
        lastScore = s
        return s
    }

    /// 128-aligned contiguous blocks of a token stream (the recall candidates = prior conversation).
    public func makeBlocks(_ ids: [Int]) -> [[Int]] {
        let nB = ids.count / Self.block
        return (0 ..< nB).map { Array(ids[$0 * Self.block ..< ($0 + 1) * Self.block]) }
    }

    /// recall_gen.py candidate set: full 128-blocks of the evicted stream + a partial `buf` pseudo-block
    /// (the tail) when it has ≥16 tokens.
    public func makeBlocksWithBuf(_ ids: [Int]) -> [[Int]] {
        let nB = ids.count / Self.block
        var blocks = (0 ..< nB).map { Array(ids[$0 * Self.block ..< ($0 + 1) * Self.block]) }
        let rem = ids.count - nB * Self.block
        if rem >= 16 { blocks.append(Array(ids.suffix(rem))) }
        return blocks
    }

    /// Bridge-head retrieval from a precomputed q-feature over precomputed blocks → top-k block ids
    /// (chronological). `blockEmb` caches each block's BGE-base vector across calls (causal/stable).
    public func retrieve(qflat: [Float], blocks: [[Int]], blockEmb: inout [[Float]], topk: Int = 2) async -> [Int] {
        guard !blocks.isEmpty else { return [] }
        if blockEmb.count != blocks.count {
            blockEmb = []
            for b in blocks { blockEmb.append(await bge.encode(tokenizer.decode(tokens: b), isQuery: false)) }
        }
        let pq = tower(qflat, wq0w, wq0b, wq2w, wq2b)
        var scores = [Float](repeating: 0, count: blocks.count)
        for b in 0 ..< blocks.count {
            let pk = tower(blockEmb[b], wk0w, wk0b, wk2w, wk2b)
            var dot: Float = 0; for j in 0 ..< 128 { dot += pq[j] * pk[j] }
            scores[b] = dot
        }
        let top = scores.indices.sorted { scores[$0] > scores[$1] }.prefix(topk).sorted()
        return top.flatMap { blocks[$0] }
    }

    /// `RecallRuntime.prepare`+`decide` for one turn. `seq` = full running conversation token ids with
    /// the just-arrived turn already appended (the gate scores the final position = the decode point).
    /// Returns (gateFired, top-2 block token ids in chronological order).
    public func recall(seq: [Int], topk: Int = 2, ignoreGate: Bool = false) async -> (fired: Bool, recIds: [Int]) {
        let nEvict = ((seq.count - Self.window) / Self.block) * Self.block
        let nB = nEvict / Self.block
        guard nB >= 1 else { lastScore = 0; return (false, []) }
        let kept = Array(seq[0 ..< nEvict])
        let window = Array(seq[nEvict...])
        let blocks = (0 ..< nB).map { Array(seq[$0 * Self.block ..< ($0 + 1) * Self.block]) }

        // [BOS][SP(kept)][window] — capture pre-RoPE q over the window positions.
        let cache = model.newCache(parameters: nil)
        _ = model.callAsFunction(MLXArray([Int32(bos)], [1, 1]), cache: cache)
        let sp = pooler.forward(embIds(kept).asType(.float32)).asType(embedDtype)
        _ = model.callAsFunction(inputEmbeddings: sp, cache: cache)
        let qc = model.captureProj(inputEmbeddings: embIds(window), cache: cache,
                                   layers: Self.layers, wantQ: true)
        let last = window.count - 1
        var qflat = [Float](); qflat.reserveCapacity(Self.layers.count * Hq * D)
        for l in Self.layers {
            let q = qc[l]!.reshaped(window.count, Hq, D)
            let ql = q[last].asType(.float32)               // [Hq, D]
            eval(ql); qflat.append(contentsOf: ql.asArray(Float.self))
        }
        // gate: ((q − mean)/scale)·coef + intercept
        var s = intercept
        for i in 0 ..< min(qflat.count, coef.count) { s += ((qflat[i] - mean[i]) / scale[i]) * coef[i] }
        lastScore = s
        let fired = s > (threshOverride ?? thresh)
        guard fired || ignoreGate else { return (false, []) }

        // bridge-head retrieval: BGE-base-encode block TEXTS, project both towers, dot-product rank.
        let pq = tower(qflat, wq0w, wq0b, wq2w, wq2b)
        var scores = [Float](repeating: 0, count: nB)
        for b in 0 ..< nB {
            let text = tokenizer.decode(tokens: blocks[b])
            let ke = await bge.encode(text, isQuery: false)         // 768-d, normalized, no prefix
            let pk = tower(ke, wk0w, wk0b, wk2w, wk2b)
            var dot: Float = 0
            for j in 0 ..< 128 { dot += pq[j] * pk[j] }
            scores[b] = dot
        }
        let top = scores.indices.sorted { scores[$0] > scores[$1] }.prefix(topk).sorted()
        return (fired, top.flatMap { blocks[$0] })
    }
}
