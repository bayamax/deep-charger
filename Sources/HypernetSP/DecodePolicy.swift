import Foundation

// Faithful Swift port of canonical `hypernet_sp/decode_policy.py` — two-phase temperature
// (diversity in <think>, greedy after) + streaming convergence/loop detector that force-closes
// <think> once the reasoning has converged (so the model can't talk itself out of a right answer).

private let assertNumRE =
    "(?:answer is|answer:|equals|=|total (?:is|of)|result is|gives us|so it'?s|"
    + "that(?:'s| is)|therefore,?|change (?:is|of|would be)|receives?|left with|"
    + "expecting)\\s*\\$?(-?\\d[\\d,]*(?:\\.\\d+)?)"
    + "|(?:gets?|got|give[sn]?|hand(?:ed)?)\\s+\\$?(-?\\d[\\d,]*(?:\\.\\d+)?)\\s+(?:back|in change)"
private let boxedRE = "\\\\boxed\\{([^}]*)\\}"

/// Canonical numeric form: strip $/commas/trailing zeros ('1,157.00' == '1157').
public func canonNum(_ s0: String) -> String {
    let s = s0.replacingOccurrences(of: ",", with: "").replacingOccurrences(of: "$", with: "")
        .trimmingCharacters(in: .whitespaces)
    guard let f = Double(s) else { return s }
    return f == f.rounded() ? String(Int(f)) : String(f)
}

/// Canonical values of all answer-shaped assertions, in order.
public func assertValues(_ text: String) -> [String] {
    guard let re = try? NSRegularExpression(pattern: assertNumRE, options: [.caseInsensitive]) else { return [] }
    let ns = text as NSString
    var out = [String]()
    for m in re.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
        for g in [1, 2] {
            let r = m.range(at: g)
            if r.location != NSNotFound { out.append(canonNum(ns.substring(with: r))); break }
        }
    }
    return out
}

private func boxedValues(_ text: String) -> [String] {
    guard let re = try? NSRegularExpression(pattern: boxedRE) else { return [] }
    let ns = text as NSString
    return re.matches(in: text, range: NSRange(location: 0, length: ns.length))
        .map { canonNum(ns.substring(with: $0.range(at: 1))) }
}

public final class DecodePolicy {
    private let k: Int
    private let greedyAfterThink: Bool
    private let minThinkChars: Int
    private(set) var counts: [String: Int] = [:]
    private(set) var fired = false

    public init(k: Int = 3, greedyAfterThink: Bool = true, minThinkChars: Int = 200) {
        self.k = k; self.greedyAfterThink = greedyAfterThink; self.minThinkChars = minThinkChars
    }

    /// Phase temperature: diversity inside <think>, argmax for the user-facing answer.
    public func temp(_ inThink: Bool, _ base: Float) -> Float {
        (inThink || !greedyAfterThink) ? base : 1e-4
    }

    private let loopNgram = 6, loopReps = 4
    private func thinkLoop(_ text: String) -> Bool {
        let w = text.split(whereSeparator: { $0 == " " || $0 == "\n" || $0 == "\t" }).map(String.init)
        if w.count < loopNgram * loopReps { return false }
        var counts = [String: Int](); var maxc = 0
        for i in 0 ... (w.count - loopNgram) {
            let g = w[i ..< i + loopNgram].joined(separator: " ")
            let c = (counts[g] ?? 0) + 1; counts[g] = c; if c > maxc { maxc = c }
        }
        return maxc >= loopReps
    }

    /// Feed the current think text. Returns true exactly once, when the same canonical asserted
    /// value has appeared ≥k times (and is the latest, with no live competitor) OR the stream loops.
    public func noteText(_ thinkText: String) -> Bool {
        if fired || thinkText.count < minThinkChars { return false }
        let vals = assertValues(thinkText) + boxedValues(thinkText)
        counts = [:]
        for c in vals { counts[c, default: 0] += 1 }
        if !counts.isEmpty {
            let (mode, nMode) = counts.max { $0.value < $1.value }!
            let runnerUp = counts.filter { $0.key != mode }.map { $0.value }.max() ?? 0
            let asserted = assertValues(thinkText)
            let last = asserted.last
            if nMode >= k, last == mode, runnerUp < 2 { fired = true; return true }
        }
        if thinkLoop(thinkText) { fired = true; return true }
        return false
    }

    public func convergedAnswer() -> String? {
        counts.max { $0.value < $1.value }?.key
    }
}
