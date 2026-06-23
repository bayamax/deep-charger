import Foundation

// Faithful Swift port of canonical `hypernet_sp/anaphora.py` — append recent entities to a web
// query that has an unresolved referring expression ("temples to visit there?" → "… (Kyoto)").

// Referring expressions needing an antecedent: place/object/person pronouns + bare deictic NPs,
// PLUS value-anaphora NPs (the number/total/price/…) added per the model-team finding-B fix
// (2026-06-12): "Can you verify the number?" → "Can you verify the number? (Brent, $82.40)".
private let anaphorRE =
    "\\b(there|it|that one|that place|this place|them|they|he|she|him|her|its|"
    + "the (place|city|town|area|spot|venue|hotel|restaurant|museum|station|one|"
    + "number|total|price|amount|figure|rate|value|cost|result|sum|count|score|percentage|quote|reading))\\b"
private let codeishRE = "\\b[A-Za-z]*\\d[A-Za-z0-9\\-]*\\b"
private let properRE = "\\b[A-Z][a-z]+(?:[\\s\\-][A-Z][a-z]+){0,2}\\b"
// money / decimal values, so the antecedent of a value-anaphor ("the number") can be carried:
// "$82.40", "82.40", "50%", "450円".
private let valueRE = "\\$\\d[\\d,]*(?:\\.\\d+)?|\\b\\d[\\d,]*\\.\\d+\\b|\\b\\d[\\d,]*\\s?(?:%|円)"

private let genericCap: Set<String> = Set(
    ("the this that what who when where which how why can could should would do does did is are was "
        + "i my our we you your please also and but yes no okay monday tuesday wednesday thursday friday "
        + "saturday sunday january february march april may june july august september october november "
        + "december am pm ok hey hi hello thanks").split(separator: " ").map(String.init))

private func isGeneric(_ w: String) -> Bool {
    let lc = w.lowercased()
    return genericCap.contains(lc) || w.split(separator: " ").allSatisfy { genericCap.contains($0.lowercased()) }
}

/// Proper-noun runs + codes + money/decimal values, in order, minus generic capitalised words.
func anaphoraEntities(_ text: String) -> [String] {
    var ents = [String]()
    for w in regexMatches(text, properRE) where !isGeneric(w) { ents.append(w) }
    for w in regexMatches(text, codeishRE) {
        let isDigits = w.allSatisfy { $0.isNumber }
        if (w.count >= 2 && !isDigits) || (isDigits && w.count >= 3) { ents.append(w) }
    }
    for w in regexMatches(text, valueRE) { ents.append(w.trimmingCharacters(in: .whitespaces)) }
    var seen = Set<String>(); return ents.filter { seen.insert($0).inserted }
}

/// True if the query already names a concrete entity (so expansion must not run).
func hasOwnAnchor(_ query: String) -> Bool {
    if !regexMatches(query, codeishRE).isEmpty { return true }
    // proper noun not at position 0, not generic
    let ns = query as NSString
    guard let re = try? NSRegularExpression(pattern: properRE) else { return false }
    for m in re.matches(in: query, range: NSRange(location: 0, length: ns.length)) {
        if m.range.location == 0 { continue }
        let w = ns.substring(with: m.range)
        if isGeneric(w) { continue }
        return true
    }
    return false
}

func needsResolution(_ query: String) -> Bool {
    !regexMatches(query, anaphorRE, caseInsensitive: true).isEmpty && !hasOwnAnchor(query)
}

/// `expand_web_query`: most-recent-LAST `pins`/`session`; appends up to `cap` recent entities.
public func expandWebQuery(_ query: String, pins: [String], session: [String] = [], cap: Int = 2) -> String {
    guard needsResolution(query) else { return query }
    var ents = [String]()
    let lcq = query.lowercased()
    for turn in Array(pins.reversed()) + Array(session.reversed()) {
        for e in anaphoraEntities(turn) where !lcq.contains(e.lowercased()) && !ents.contains(e) {
            ents.append(e)
        }
    }
    if ents.isEmpty { return query }
    return "\(query) (\(ents.prefix(cap).joined(separator: ", ")))"
}
