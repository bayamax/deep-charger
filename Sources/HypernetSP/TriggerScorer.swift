import Foundation
import MLX

/// Swift port of `hypernet_sp/trigger_runtime.TriggerScorer` — the attention-mass retrieval trigger.
///
/// `score(sp:turnIds:)` returns P(this turn needs a value that now lives only in the compressed SP
/// history). One short eager forward over `[BOS | SP | turn-tokens]` reads how much the new turn's
/// tokens attend back onto the 32 SP positions, at 6 trained `(layer, head)` pairs; a logistic head
/// turns those 6 masses into a probability. Above `fireThreshold` → the app re-injects recent
/// memory so the model can answer referentially (multi-turn). Cost: ms-scale (layers ≤ 22 of 28).
public final class TriggerScorer {
    public struct Params: Decodable {
        public let heads: [[Int]]; public let mu: [Float]; public let sd: [Float]
        public let coef: [Float]; public let intercept: Float; public let threshold: Float
    }

    private let model: SPQwen2Model
    private let bos: Int
    private let nSP: Int
    private let pairs: [(layer: Int, head: Int)]
    private let mu, sd, coef: [Float]
    private let intercept: Float
    public let fireThreshold: Float

    public init(model: SPQwen2Model, bos: Int, nSP: Int, paramsURL: URL) throws {
        self.model = model; self.bos = bos; self.nSP = nSP
        let p = try JSONDecoder().decode(Params.self, from: Data(contentsOf: paramsURL))
        self.pairs = p.heads.map { (layer: $0[0], head: $0[1]) }
        self.mu = p.mu; self.sd = p.sd; self.coef = p.coef
        self.intercept = p.intercept; self.fireThreshold = p.threshold
    }

    /// `sp`: the conversation's soft prompts (1, nSP, H). `turnIds`: the new user-turn token ids.
    public func score(sp: MLXArray, turnIds: [Int]) -> Float {
        guard !turnIds.isEmpty, sp.dim(1) == nSP else { return 0 }
        let bosE = model.embed(MLXArray([Int32(bos)], [1, 1]))
        let turnE = model.embed(MLXArray(turnIds.map { Int32($0) }, [1, turnIds.count]))
        let input = concatenated([bosE, sp.asType(bosE.dtype), turnE.asType(bosE.dtype)], axis: 1)
        let masses = model.triggerMasses(inputEmbeddings: input, heads: pairs, nSP: nSP)
        var z = intercept
        for i in 0 ..< masses.count { z += coef[i] * (masses[i] - mu[i]) / sd[i] }
        return 1 / (1 + exp(-z))
    }
}
