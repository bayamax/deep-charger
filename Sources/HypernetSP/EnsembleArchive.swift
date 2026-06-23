import Foundation
import MLX
import MLXLMCommon
import Tokenizers

/// Swift port of `hypernet_sp/ensemble_archive.EnsembleArchive` (BLOCK_RECALL_SPEC §8, Phase B).
///
/// Same blocks/inject/wiring as `BlockArchive`; only the SCORING differs:
///   score_i = z(bge_i) + 1.5·z(idx_i)
/// where idx_i is a trained indexer over CONTEXTUAL pre-RoPE key/query projections at layers
/// (8,14,20). Block keys = mean+max of k_proj over the block (harvested with one cached forward over
/// the history); the query is forwarded WITH that history cache behind it (a bare forward drifts at
/// deep layers). Measured needle 10/10 (BGE-only 7/10).
public final class EnsembleArchive: RecallArchive {
    private static let layers = [8, 14, 20]
    private static let BLOCK = 128
    private let L = 3, Hq = 12, Hkv = 2, D = 128, dd = 32, group = 6

    private let model: SPQwen2Model
    private let tokenizer: Tokenizer
    private let bge: Embedder
    private let bos: Int

    // indexer weights, flattened row-major
    private let Aq: [Float]   // [L, Hq, D, dd]
    private let Bk: [Float]   // [L, Hkv, 2D, dd]
    private let wsm: [Float]  // softmax(w) [L, Hq]

    private var blocks: [[Int]] = []     // sealed block ids
    private var buf: [Int] = []
    private var archivedLen = 0
    // cached features (per current block set)
    private var kfeat: [Float]? = nil    // [nB, L, Hkv, 2D]
    private var emb: [[Float]] = []      // per-block BGE doc vectors
    private var histCache: [KVCache]? = nil
    private var clen = 0

    public init(model: SPQwen2Model, tokenizer: Tokenizer, bge: Embedder, bos: Int, indexerURL: URL) throws {
        self.model = model; self.tokenizer = tokenizer; self.bge = bge; self.bos = bos
        let ix = try loadArrays(url: indexerURL)
        guard let aq = ix["Aq"], let bk = ix["Bk"], let w = ix["w"] else { throw Err.missing }
        Aq = aq.asType(.float32).asArray(Float.self)
        Bk = bk.asType(.float32).asArray(Float.self)
        // softmax over the flattened w (matches np: e=exp(w-max); e/e.sum())
        let wv = w.asType(.float32).asArray(Float.self)
        let mx = wv.max() ?? 0
        let ex = wv.map { Foundation.exp($0 - mx) }
        let s = ex.reduce(0, +)
        wsm = ex.map { $0 / s }
    }
    enum Err: Error { case missing }

    public func reset() {
        blocks = []; buf = []; archivedLen = 0; kfeat = nil; emb = []; histCache = nil; clen = 0
    }

    public func sync(absorbed: [Int]) async {
        guard absorbed.count > archivedLen else { return }
        buf.append(contentsOf: absorbed[archivedLen...]); archivedLen = absorbed.count
        while buf.count >= Self.BLOCK {
            blocks.append(Array(buf.prefix(Self.BLOCK))); buf.removeFirst(Self.BLOCK)
            kfeat = nil                       // features stale
        }
    }

    private func embedIds(_ ids: [Int]) -> MLXArray {
        model.embed(MLXArray(ids.map { Int32($0) }, [1, ids.count]))
    }

    /// One cached forward over the whole history → contextual block keys + block BGE vectors.
    private func featurize() async {
        let nB = blocks.count
        let ids = blocks.flatMap { $0 }
        let cache = model.newCache(parameters: nil)
        let kc = model.captureProj(inputEmbeddings: embedIds(ids), cache: cache,
                                   layers: Self.layers, wantQ: false)
        clen = ids.count; histCache = cache
        // per layer: [hist, Hkv, D] → reshape [nB, BLOCK, Hkv, D] → concat(mean, max over BLOCK) → [nB, Hkv, 2D]
        var feat = [Float](repeating: 0, count: nB * L * Hkv * (2 * D))
        for (li, layer) in Self.layers.enumerated() {
            let k = kc[layer]!.reshaped(nB, Self.BLOCK, Hkv, D)
            let kmean = MLX.mean(k, axis: 1)              // [nB, Hkv, D]
            let kmax = MLX.max(k, axis: 1)                // [nB, Hkv, D]
            let kf = MLX.concatenated([kmean, kmax], axis: -1).asType(.float32)   // [nB, Hkv, 2D]
            eval(kf)
            let flat = kf.asArray(Float.self)             // nB*Hkv*2D row-major
            let twoD = 2 * D
            for b in 0 ..< nB {
                for kk in 0 ..< Hkv {
                    for f in 0 ..< twoD {
                        feat[((b * L + li) * Hkv + kk) * twoD + f] = flat[(b * Hkv + kk) * twoD + f]
                    }
                }
            }
        }
        kfeat = feat
        emb = []
        for b in blocks { emb.append(await bge.encode(tokenizer.decode(tokens: b), isQuery: false)) }
    }

    public func retrieve(query: String) async -> [Int] {
        guard !blocks.isEmpty else { return [] }
        if kfeat == nil { await featurize() }
        guard let feat = kfeat, let cache = histCache else { return [] }
        let nB = blocks.count
        // query features WITH the history cache behind (pre-closed empty think, like the trainer)
        let pre = tokenizer.encode(text:
            "<｜end▁of▁sentence｜><｜User｜>\(query)<｜Assistant｜><think>\n\n</think>\n\n",
            addSpecialTokens: false)
        for c in cache { _ = c.trim(c.offset - clen) }    // restore to the history prefix
        let qc = model.captureProj(inputEmbeddings: embedIds(pre), cache: cache,
                                   layers: Self.layers, wantQ: true)
        // qf[L,Hq,D] = mean of q_proj over the query tokens
        var qf = [Float](repeating: 0, count: L * Hq * D)
        for (li, layer) in Self.layers.enumerated() {
            let q = qc[layer]!.reshaped(pre.count, Hq, D)
            let qm = MLX.mean(q, axis: 0).asType(.float32)    // [Hq, D]
            eval(qm)
            let flat = qm.asArray(Float.self)
            for h in 0 ..< Hq { for dd2 in 0 ..< D { qf[(li * Hq + h) * D + dd2] = flat[h * D + dd2] } }
        }
        // qp[l,h,e] = Σ_d qf·Aq ;  kp[b,l,k,e] = Σ_f kfeat·Bk ; GQA replicate ; sim=ReLU(qp·kp_rep)
        let twoD = 2 * D
        var qp = [Float](repeating: 0, count: L * Hq * dd)
        for l in 0 ..< L { for h in 0 ..< Hq { for e in 0 ..< dd {
            var s: Float = 0
            for d in 0 ..< D { s += qf[(l * Hq + h) * D + d] * Aq[(((l * Hq + h) * D + d) * dd) + e] }
            qp[(l * Hq + h) * dd + e] = s
        }}}
        var sIdx = [Float](repeating: 0, count: nB)
        for b in 0 ..< nB {
            var acc: Float = 0
            for l in 0 ..< L {
                for h in 0 ..< Hq {
                    let kk = h / group                      // GQA: query head → kv head
                    var dot: Float = 0
                    for e in 0 ..< dd {
                        var kpe: Float = 0
                        for f in 0 ..< twoD {
                            kpe += feat[((b * L + l) * Hkv + kk) * twoD + f] * Bk[(((l * Hkv + kk) * twoD + f) * dd) + e]
                        }
                        dot += qp[(l * Hq + h) * dd + e] * kpe
                    }
                    acc += max(dot, 0) * wsm[l * Hq + h]
                }
            }
            sIdx[b] = acc
        }
        // BGE arm
        let qv = await bge.encode(query, isQuery: true)
        var sBge = [Float](repeating: 0, count: nB)
        for b in 0 ..< nB { sBge[b] = zip(emb[b], qv).reduce(0) { $0 + $1.0 * $1.1 } }
        // ensemble: z-normalise each, s = 1.5·z(idx) + z(bge)
        func z(_ a: [Float]) -> [Float] {
            let m = a.reduce(0, +) / Float(a.count)
            let v = a.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Float(a.count)
            let sd = v.squareRoot() + 1e-8
            return a.map { ($0 - m) / sd }
        }
        let zi = z(sIdx), zb = z(sBge)
        let score = (0 ..< nB).map { 1.5 * zi[$0] + zb[$0] }
        let top = score.indices.sorted { score[$0] > score[$1] }.prefix(2).sorted()
        return top.flatMap { blocks[$0] }
    }
}
