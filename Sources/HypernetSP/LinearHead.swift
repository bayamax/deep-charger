import Foundation

/// A linear probe on the BGE embedding — the exported sklearn LogisticRegression heads
/// (intent router, specificity). Rows of `coef` follow `classes` order (sklearn `classes_`),
/// which is what `predict_proba` uses.
public struct LinearHead: Decodable {
    public let classes: [String]
    public let coef: [[Float]]       // (nClasses, dim) — for binary LR, 1 row
    public let intercept: [Float]
    public let dim: Int
    public let multiclass: Bool

    public static func load(_ url: URL) throws -> LinearHead {
        try JSONDecoder().decode(LinearHead.self, from: Data(contentsOf: url))
    }

    private func decision(_ x: [Float]) -> [Float] {
        coef.enumerated().map { (i, w) -> Float in
            var s = intercept[i]
            let n = min(w.count, x.count)
            var k = 0
            while k < n { s += w[k] * x[k]; k += 1 }
            return s
        }
    }

    /// Multiclass: softmax over the per-class decisions → (best label, its prob, all probs).
    public func classify(_ x: [Float]) -> (label: String, prob: Float, probs: [String: Float]) {
        let d = decision(x)
        let m = d.max() ?? 0
        let ex = d.map { Foundation.exp($0 - m) }
        let z = ex.reduce(0, +)
        let p = ex.map { $0 / z }
        var best = 0
        for i in p.indices where p[i] > p[best] { best = i }
        var dict = [String: Float]()
        for i in classes.indices { dict[classes[i]] = p[i] }
        return (classes[best], p[best], dict)
    }

    /// Binary LR: P(positive) = sigmoid(decision). `positive` is the class label (default "1").
    public func scorePositive(_ x: [Float], positive: String = "1") -> Float {
        // one-row coef → decision[0] is the log-odds of the positive class
        let d = decision(x)[0]
        return 1 / (1 + Foundation.exp(-d))
    }
}
