import Foundation

// Faithful Swift port of canonical `hypernet_sp/intent_route.py` — three-band intent routing.
// Intent classes: math, command, recall, lookup, fact, chitchat.

private let commandRE = "\\b(recompute|recalculate|recalc|redo|new total)\\b|the total"
private let recallCueRE =
    "\\b(remind me|recall|do you remember|did (i|you) (say|mention|tell|leave|park)|what (was|did) (i|you)\\b|again\\b)|"
    + "\\b(what|whats|what's|which|when|where|how)\\b[^,.!?]{0,24}\\b(my|mine)\\b"
private let adviceRE =
    "\\b(tips?|suggest(ions?)?|recommend(ations?)?|ideas?|should (i|we)|what do you think|brainstorm|help me (plan|write|pick|decide)|good)\\b"
private let numWordRE =
    "\\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|hundred|thousand|dozen|half|quarter|double)\\b"
private let opWordRE =
    "\\b(plus|minus|times|divided|multiply|multiplied|subtract(ed)?|add(ed|s)?|percent|sum of|total of|product of|difference|average of|squared|cubed|double|half of)\\b"

private func has(_ text: String, _ pattern: String) -> Bool {
    !regexMatches(text, pattern, caseInsensitive: true).isEmpty
}

/// `regex_intent`: the no-information fallback.
func regexIntent(_ text: String) -> String {
    let t = text.lowercased()
    if mcLooksMathy(text) { return "math" }
    if has(t, commandRE) { return "command" }
    if mcIsQuestion(text) {
        return has(text, "\\b(my|mine|i|me|we|our)\\b") ? "recall" : "lookup"
    }
    return mcIsFactlike(text) ? "fact" : "chitchat"
}

/// `_evidence`: weighted syntactic support for `label` (arbitrates the classifier's top-2 only).
private func evidence(_ text: String, _ label: String) -> Int {
    let t = text.lowercased()
    let q = mcIsQuestion(text)
    switch label {
    case "math":
        return 2 * ((mcLooksMathy(text) || (has(text, numWordRE) && (has(t, mathRE_local) || has(t, opWordRE)))) ? 1 : 0)
    case "command":
        return 2 * (has(t, commandRE) ? 1 : 0)
    case "recall":
        return 2 * ((q && has(text, recallCueRE)) ? 1 : 0)
    case "lookup":
        return 0
    case "fact":
        return mcIsFactlike(text) ? 1 : 0
    case "chitchat":
        if q && has(text, adviceRE) { return 2 }
        return (!q && !mcIsFactlike(text)) ? 1 : 0
    default:
        return 0
    }
}
// `_MATH` is private in MemoryCore; re-declare the same pattern here for `_evidence`.
private let mathRE_local = "how (much|many)|calculat|comput|percent|%|\\bplus\\b|\\bminus\\b|times|divided|average|\\btotal\\b|\\bsum\\b|sqrt|\\d\\s*[-+*/=]\\s*\\d"

private func fixInterrogativeFact(_ lab: String, _ text: String) -> String {
    guard lab == "fact", mcIsQuestion(text) else { return lab }
    if mcLooksMathy(text) { return "math" }
    if has(text, recallCueRE) { return "recall" }
    if has(text, adviceRE) { return "chitchat" }
    return "lookup"
}

/// Three-band `route_intent`. `docEmbedding` is the BGE doc-mode embedding of `text`.
public func routeIntent(_ text: String, intentHead: LinearHead, docEmbedding: [Float],
                 hi: Float = 0.5, lo: Float = 0.30) -> String {
    guard !docEmbedding.isEmpty else { return regexIntent(text) }
    let (_, _, probs) = intentHead.classify(docEmbedding)
    // order classes by probability desc
    let ordered = intentHead.classes.sorted { (probs[$0] ?? 0) > (probs[$1] ?? 0) }
    let top1 = ordered[0]
    let top2 = ordered.count > 1 ? ordered[1] : top1
    let p1 = probs[top1] ?? 0

    if p1 >= hi { return fixInterrogativeFact(top1, text) }
    if p1 >= lo {
        let e1 = evidence(text, top1), e2 = evidence(text, top2)
        if e1 == e2, e1 == 0, mcIsQuestion(text), has(text, adviceRE),
            !has(text, recallCueRE), !mcLooksMathy(text) {
            return "chitchat"
        }
        let lab = e2 > e1 ? top2 : top1
        return fixInterrogativeFact(lab, text)
    }
    return regexIntent(text)
}
