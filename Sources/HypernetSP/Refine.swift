import Foundation

/// Stopwords from `tiered_rag_mlx._STOP` (groundedness salient-token filter).
let spStop: Set<String> = Set(
    ("the a an of in on at to is are was were am my your you i me we what who which when where "
        + "why how do does did can could will would should it its this that these those and or but for "
        + "with about please tell remember our us so let's let now and have has had get got make made "
        + "many much more most some any here there also like want need give given take just it's")
        .split(separator: " ").map(String.init))

func regexMatches(_ text: String, _ pattern: String, caseInsensitive: Bool = false) -> [String] {
    let opts: NSRegularExpression.Options = caseInsensitive ? [.caseInsensitive] : []
    guard let re = try? NSRegularExpression(pattern: pattern, options: opts) else { return [] }
    let ns = text as NSString
    return re.matches(in: text, range: NSRange(location: 0, length: ns.length))
        .map { ns.substring(with: $0.range) }
}

/// Port of `tiered_rag_mlx.is_grounded`: a faithful retrieval-grounded answer must echo a salient
/// *novel* token (code/number, capitalised, or ≥5 chars) from the retrieved context — not just the
/// question. Rejects out-of-context / empty / question-echo answers (necessary, not sufficient).
public func isGrounded(answer: String, chunks: [String], question: String) -> Bool {
    let ctx = chunks.joined(separator: " ").lowercased()
    let qw = Set(regexMatches(question.lowercased(), "[a-z0-9\\-]+").filter { $0.count >= 2 })
    let toks = regexMatches(answer, "[A-Za-z0-9][A-Za-z0-9\\-]{2,}")
    let salient = toks.filter { t in
        let lc = t.lowercased()
        let concrete = t.contains(where: { $0.isNumber }) || (t.first?.isUppercase ?? false) || t.count >= 5
        return concrete && !spStop.contains(lc) && !qw.contains(lc)
    }
    if salient.isEmpty { return false }
    return salient.contains { ctx.contains($0.lowercased()) }
}

/// Candidate pin-worthy spans (`tiered_rag_mlx._CAND`): money, times, measures, codes/ids,
/// proper-noun runs. Case-sensitive. Deduped, length ≥2, cap 24. Scoring (specificity head) is
/// applied by the caller.
public func specificCandidates(_ text: String) -> [String] {
    let pattern =
        "\\$\\d[\\d,]*(?:\\.\\d+)?"                                        // money
        + "|\\b\\d{1,2}(?::\\d{2})?\\s?(?:am|pm)\\b"                       // times
        + "|\\b\\d+(?:\\.\\d+)?\\s?(?:cm|mm|km|kg|%|percent|days?|years?|hours?|min)\\b"  // measures
        + "|\\b[A-Za-z]*\\d[A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\\b"             // codes/ids/numbers
        + "|\\b[A-Z][a-z]{2,}(?:\\s+[A-Z][a-z]{2,}){0,2}\\b"             // proper-noun runs
    var seen = Set<String>(); var out = [String]()
    for m in regexMatches(text, pattern) {
        let s = m.trimmingCharacters(in: .whitespaces)
        if s.count >= 2, seen.insert(s).inserted { out.append(s) }
        if out.count >= 24 { break }
    }
    return out
}

/// Port of `tiered_rag_mlx._looks_degenerate`: a 3-gram repeated ≥4 times = a repetition loop.
public func looksDegenerate(_ text: String) -> Bool {
    let w = text.split(whereSeparator: { $0 == " " || $0 == "\n" || $0 == "\t" }).map(String.init)
    if w.count < 8 { return false }
    var counts = [String: Int]()
    var maxc = 0
    for i in 0 ... (w.count - 3) {
        let tri = w[i] + " " + w[i + 1] + " " + w[i + 2]
        let c = (counts[tri] ?? 0) + 1
        counts[tri] = c
        if c > maxc { maxc = c }
    }
    return maxc >= 4
}

/// The last `\boxed{...}` value in `text`, if any (the value the reasoning reached).
public func lastBoxed(_ text: String) -> String? {
    let matches = regexMatches(text, "\\\\boxed\\{([^}]*)\\}")
    guard let last = matches.last else { return nil }
    // strip the \boxed{ } wrapper to the inner value
    if let open = last.range(of: "{"), let close = last.range(of: "}", options: .backwards) {
        let inner = String(last[open.upperBound ..< close.lowerBound]).trimmingCharacters(in: .whitespaces)
        return inner.isEmpty ? nil : inner
    }
    return nil
}

private let emptyAnswers: Set<String> = ["", "(still thinking — timed out)", "(no answer)"]
public func isEmptyAnswer(_ s: String) -> Bool { emptyAnswers.contains(s) || s.count < 2 }

/// A garbage answer: a character repeated ≥6× (e.g. "!!!!!!"), or mostly non-alphanumeric.
public func looksGarbage(_ text: String) -> Bool {
    let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
    if t.isEmpty { return true }
    if !regexMatches(t, "(.)\\1{5,}").isEmpty { return true }
    let alnum = t.unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) }.count
    return Double(alnum) / Double(t.count) < 0.3
}

/// The user-facing answer: the text after the final `</think>`, else the whole thing (stripped of a
/// leading unterminated `<think>`). Used for display + the groundedness check.
public func extractAnswer(_ text: String) -> String {
    if let r = text.range(of: "</think>", options: .backwards) {
        return String(text[r.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
    }
    var t = text
    if let r = t.range(of: "<think>") { t = String(t[r.upperBound...]) }
    return t.trimmingCharacters(in: .whitespacesAndNewlines)
}
