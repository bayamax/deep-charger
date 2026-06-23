import Foundation
import MLX
import MLXLMCommon
import Tokenizers

/// Swift port of `hypernet_sp/gate/trigger_gate.TriggerGate` (APP_SPEC_ADDENDUM, TRIGGER_V3).
///
/// A learned recall gate: scores the current question in the production layout
/// `[BOS][SP(kept)][raw window][question]` using the question-position pre-RoPE q_proj at layers
/// 8/14/20 (4608-dim) and a logistic head. `fires()==true` means "the answer needs something evicted
/// from this session's window" — only then does block recall pay for an archive retrieve + injection.
/// Threshold −2.6 (held-out TPR 1.000 / FPR 0.043): a false fire costs one redundant retrieve, a miss
/// costs the recall itself.
public final class TriggerGate {
    private static let layers = [8, 14, 20]
    private let model: SPQwen2Model
    private let pooler: Pooler
    private let tokenizer: Tokenizer
    private let embedDtype: DType
    private let bos: Int

    private let coef: [Float], mean: [Float], scale: [Float]
    private let intercept: Float
    public let thresh: Float
    public private(set) var lastScore: Float = 0

    public init(model: SPQwen2Model, pooler: Pooler, tokenizer: Tokenizer, embedDtype: DType,
                bos: Int, headURL: URL) throws {
        self.model = model; self.pooler = pooler; self.tokenizer = tokenizer
        self.embedDtype = embedDtype; self.bos = bos
        let z = try loadArrays(url: headURL)
        guard let c = z["coef"], let b = z["intercept"], let m = z["mean"],
              let s = z["scale"], let t = z["thresh"] else { throw Err.missing }
        coef = c.asType(.float32).asArray(Float.self)
        mean = m.asType(.float32).asArray(Float.self)
        scale = s.asType(.float32).asArray(Float.self)
        intercept = b.asType(.float32).asArray(Float.self).first ?? 0
        thresh = t.asType(.float32).asArray(Float.self).first ?? 0
    }
    enum Err: Error { case missing }

    private func embIds(_ ids: [Int]) -> MLXArray {
        model.embed(MLXArray(ids.map { Int32($0) }, [1, ids.count]))
    }

    /// Logistic score for "this turn needs an evicted fact". `kept` = evicted/distant buffer (SP
    /// source), `window` = recent exposure-window tokens, `query` = the (English-core) user message.
    public func score(kept: [Int], window: [Int], query: String) -> Float {
        let pre = tokenizer.encode(text:
            "<｜end▁of▁sentence｜><｜User｜>\(query)<｜Assistant｜><think>\n\n</think>\n\n",
            addSpecialTokens: false)
        let cache = model.newCache(parameters: nil)
        _ = model.callAsFunction(MLXArray([Int32(bos)], [1, 1]), cache: cache)   // BOS sink
        var parts = [MLXArray]()
        if !kept.isEmpty { parts.append(pooler.forward(embIds(kept).asType(.float32)).asType(embedDtype)) }
        if !window.isEmpty { parts.append(embIds(window)) }
        if !parts.isEmpty {
            _ = model.callAsFunction(inputEmbeddings: parts.count == 1 ? parts[0]
                                     : MLX.concatenated(parts, axis: 1), cache: cache)
        }
        let qc = model.captureProj(inputEmbeddings: embIds(pre), cache: cache,
                                   layers: Self.layers, wantQ: true)
        var f = [Float]()
        for l in Self.layers {
            let qm = MLX.mean(qc[l]![0], axis: 0).asType(.float32)   // mean over question tokens → [Hq*D]
            eval(qm); f.append(contentsOf: qm.asArray(Float.self))
        }
        var s = intercept
        for i in 0 ..< min(f.count, coef.count) { s += ((f[i] - mean[i]) / scale[i]) * coef[i] }
        lastScore = s
        return s
    }

    public func fires(kept: [Int], window: [Int], query: String) -> Bool {
        score(kept: kept, window: window, query: query) > thresh
    }
}
