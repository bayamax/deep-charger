import Foundation

// Faithful Swift port of canonical `hypernet_sp/memory_core.py` — lexical/semantic matching
// helpers + TieredMemory (L1 session / L2 disk / pins). Behaviour mirrors the reference.

private let jpStop: Set<String> = Set(
    "です ます した して いる ある する それ これ あれ どこ なに 何で ですか ました".split(separator: " ").map(String.init))

/// True if the scalar is in the kana/CJK ranges used by `_CJK`.
private func isCJK(_ u: Unicode.Scalar) -> Bool {
    let v = u.value
    return (0x3040...0x30FF).contains(v)       // hiragana + katakana (぀-ヿ)
        || (0x3400...0x9FFF).contains(v)       // CJK unified (㐀-鿿)
        || (0xF900...0xFAFF).contains(v)       // CJK compat (豈-﫿)
}

private func cjkRuns(_ s: String) -> [String] {
    var runs = [String](); var cur = ""
    for ch in s.unicodeScalars {
        if isCJK(ch) { cur.unicodeScalars.append(ch) }
        else if !cur.isEmpty { runs.append(cur); cur = "" }
    }
    if !cur.isEmpty { runs.append(cur) }
    return runs
}

/// `_words`: content tokens for lexical overlap; CJK runs contribute character bigrams.
func mcWords(_ s: String) -> Set<String> {
    var out = Set<String>()
    for w in regexMatches(s.lowercased(), "[a-z0-9][a-z0-9\\-]*") where w.count >= 2 && !spStop.contains(w) {
        out.insert(w)
    }
    for run in cjkRuns(s) where run.count >= 2 {
        let chars = Array(run)
        for i in 0 ..< (chars.count - 1) {
            let bg = String(chars[i ... i + 1])
            if !jpStop.contains(bg) { out.insert(bg) }
        }
    }
    return out
}

/// `_overlap`: count of query words matching a target word (exact or ≥4-char shared prefix).
func mcOverlap(_ qw: Set<String>, _ tw: Set<String>) -> Int {
    var n = 0
    for a in qw {
        if tw.contains(where: { b in
            a == b || (a.count >= 4 && b.count >= 4 && (a.hasPrefix(b) || b.hasPrefix(a)))
        }) { n += 1 }
    }
    return n
}

/// `_matches`: lexical top-`cap` chunks (within 1 of best overlap), in chronological order.
func mcMatches(_ query: String, _ store: [String], cap: Int = 4) -> [String] {
    let qw = mcWords(query)
    var scored = store.enumerated().map { (i, t) in (s: mcOverlap(qw, mcWords(t)), i: i, t: t) }
        .filter { $0.s > 0 }
    guard let hi = scored.map({ $0.s }).max() else { return [] }
    let near = scored.filter { $0.s >= hi - 1 }
    let top = near.sorted { ($0.s, -$0.i) > ($1.s, -$1.i) }.prefix(cap)
    return top.sorted { $0.i < $1.i }.map { $0.t }
}

private let correctionRE = "\\b(actually|instead|scratch that|correction|rather|no wait|wait no|make it|changed?( it| to)?|should be|moved to|now it'?s|\\bnot\\b)"

/// `_with_amendments`: any correction line that follows and shares a word with a selected line rides along.
func withAmendments(_ selected: [String], _ store: [String]) -> [String] {
    var sel = Set(selected)
    var changed = true
    while changed {
        changed = false
        for (i, t) in store.enumerated() {
            if sel.contains(t) || regexMatches(t, correctionRE, caseInsensitive: true).isEmpty { continue }
            let tw = mcWords(t)
            for (j, s) in store.enumerated() where sel.contains(s) && j < i && mcOverlap(mcWords(s), tw) >= 1 {
                sel.insert(t); changed = true; break
            }
        }
    }
    return store.filter { sel.contains($0) }
}

/// `mark_superseded`: tag corrected lines `(outdated)` and corrections `(current)`.
func markSuperseded(_ chunks: [String]) -> [String] {
    let n = chunks.count
    var superseded = [Bool](repeating: false, count: n)
    var corrects = [Bool](repeating: false, count: n)
    for i in 0 ..< n {
        if regexMatches(chunks[i], correctionRE, caseInsensitive: true).isEmpty { continue }
        let cw = mcWords(chunks[i])
        for j in 0 ..< i where mcOverlap(mcWords(chunks[j]), cw) >= 1 {
            superseded[j] = true; corrects[i] = true
        }
    }
    return chunks.enumerated().map { (i, c) in
        superseded[i] ? "(outdated) \(c)" : (corrects[i] ? "(current) \(c)" : c)
    }
}

private let qLead = ["what", "what's", "whats", "who", "who's", "whose", "when", "where",
                     "which", "how", "why", "recall", "remind", "find", "do you know",
                     "tell me", "is my", "are my"]

func mcIsQuestion(_ text: String) -> Bool {
    let t = text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    return t.hasSuffix("?") || qLead.contains { t.hasPrefix($0) }
}

private let mathRE = "how (much|many)|calculat|comput|percent|%|\\bplus\\b|\\bminus\\b|times|divided|average|\\btotal\\b|\\bsum\\b|sqrt|\\d\\s*[-+*/=]\\s*\\d"
func mcLooksMathy(_ q: String) -> Bool {
    return !regexMatches(q, "\\d").isEmpty && !regexMatches(q.lowercased(), mathRE).isEmpty
}

func mcWantsPersist(_ text: String) -> Bool {
    !regexMatches(text.lowercased(), "remember|save|profile|note|for later|always").isEmpty
}

private let factlikeRE = "\\d|\\b(is|are|am|was|were|will|'ll|named|called|theme|colou?r|name|make it|change|instead|order|paint|costs?|each)\\b"
func mcFactlike(_ text: String) -> Bool {
    !regexMatches(text, factlikeRE, caseInsensitive: true).isEmpty
}
func mcIsFactlike(_ text: String) -> Bool { !mcIsQuestion(text) && mcFactlike(text) }

/// `_answer_ok`: empty/degenerate → bad; if chunks present, must be grounded.
func answerOk(_ answer: String, _ chunks: [String], _ question: String) -> Bool {
    let a = answer.trimmingCharacters(in: .whitespacesAndNewlines)
    if isEmptyAnswer(a) { return false }
    if looksDegenerate(a) || looksGarbage(a) { return false }
    if !chunks.isEmpty && !isGrounded(answer: answer, chunks: chunks, question: question) { return false }
    return true
}

/// Tiered memory (L1 session / L2 disk / pins). Strings are stored; embeddings are cached per
/// string and matched with BGE — mirroring the reference `_sem_matches` / `_match`.
public final class TieredMemory {
    public static let PINCAP = 12
    public static let LOGCAP = 16

    public private(set) var session: [String] = []
    public private(set) var persistent: [String] = []
    public private(set) var pins: [String] = []

    private let bge: Embedder?
    private let path: URL?
    private var embCache: [String: [Float]] = [:]

    public init(bge: Embedder?, l2URL: URL? = nil) {
        self.bge = bge
        self.path = l2URL
        if let url = l2URL, let data = try? String(contentsOf: url, encoding: .utf8) {
            for line in data.split(separator: "\n") where !line.isEmpty {
                if let d = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
                    let t = d["text"] as? String { persistent.append(t) }
            }
        }
    }

    public func remember_session(_ t: String) { session.append(t) }
    /// A recompute supersedes the previous result line (working state, not history).
    public func dropComputedResults() { session.removeAll { $0.hasPrefix("Earlier computed result:") } }
    public func persist(_ t: String) {
        session.append(t); persistent.append(t)
        if let url = path {
            let line = String(data: try! JSONSerialization.data(withJSONObject: ["text": t]), encoding: .utf8)! + "\n"
            if let h = try? FileHandle(forWritingTo: url) {
                h.seekToEndOfFile(); h.write(Data(line.utf8)); try? h.close()
            } else { try? line.write(to: url, atomically: true, encoding: .utf8) }
        }
    }
    public func pin(_ t: String) {
        pins.removeAll { $0 == t }
        pins.append(t)
        if pins.count > Self.PINCAP { pins = Array(pins.suffix(Self.PINCAP)) }
    }

    public var counts: (l1: Int, pins: Int, l2: Int) { (session.count, pins.count, persistent.count) }

    private func embed(_ t: String, isQuery: Bool) async -> [Float] {
        if !isQuery, let c = embCache[t] { return c }
        guard let bge else { return [] }
        let e = await bge.encode(t, isQuery: isQuery)
        if !isQuery { embCache[t] = e }
        return e
    }

    /// `_sem_matches`: cosine ≥ min_sim, within top_delta of best, recency tie-break, cap.
    func semMatches(_ query: String, _ store: [String], minSim: Float = 0.46,
                    topDelta: Float = 0.12, recEps: Float = 0.05, cap: Int = 4) async -> [String]? {
        guard bge != nil, !store.isEmpty else { return nil }
        let qv = await embed(query, isQuery: true)
        var sims = [Float](); sims.reserveCapacity(store.count)
        for t in store { sims.append(cosine(await embed(t, isQuery: false), qv)) }
        var keep = (0 ..< store.count).filter { sims[$0] >= minSim }
        if keep.isEmpty { return [] }
        let hi = keep.map { sims[$0] }.max()!
        keep = keep.filter { sims[$0] >= hi - topDelta }
        keep.sort { (a, b) in
            let ra = -(sims[a] / recEps).rounded(), rb = -(sims[b] / recEps).rounded()
            return ra != rb ? ra < rb : a > b
        }
        return Array(keep.prefix(cap)).sorted().map { store[$0] }
    }

    /// `_match`: semantic-first for English; lexical-first (raised bar) for CJK; then amendments.
    func match(_ query: String, _ store: [String], cap: Int = 4) async -> [String] {
        if !cjkRuns(query).isEmpty {
            let lex = mcMatches(query, store, cap: cap)
            if !lex.isEmpty { return withAmendments(lex, store) }
            let r = await semMatches(query, store, minSim: 0.72, topDelta: 0.06, cap: cap)
            return (r?.isEmpty == false) ? withAmendments(r!, store) : []
        }
        var r = await semMatches(query, store, cap: cap)
        if r == nil { r = mcMatches(query, store, cap: cap) }
        return (r?.isEmpty == false) ? withAmendments(r!, store) : []
    }

    /// `retrieve_known`: STRICT personal probe for lookup turns (min_sim 0.6 + shared content word).
    public func retrieveKnown(_ query: String, minSim: Float = 0.6) async -> (src: String?, chunks: [String]) {
        let pool = persistent.filter { !session.contains($0) } + session
        var hits = (await semMatches(query, pool, minSim: minSim)) ?? []
        let qw = mcWords(query)
        hits = hits.filter { mcOverlap(qw, mcWords($0)) >= 1 }
        if hits.isEmpty { return (nil, []) }
        let tiers = hits.map { session.contains($0) ? "L1·same-session" : "L2·prior-session" }
        return (orderedJoin(tiers), withAmendments(hits, pool))
    }

    /// `retrieve_personal`: L1+L2 ranked as one pool via `_match`, then pins fallback.
    public func retrievePersonal(_ query: String) async -> (src: String?, chunks: [String]) {
        let pool = persistent.filter { !session.contains($0) } + session
        let hits = await match(query, pool)
        if !hits.isEmpty {
            let tiers = hits.map { session.contains($0) ? "L1·same-session" : "L2·prior-session" }
            return (orderedJoin(tiers), hits)
        }
        if !pins.isEmpty, let mp = await semMatches(query, pins, minSim: 0.5), !mp.isEmpty {
            return ("WM·pins", mp)
        }
        return (nil, [])
    }

    private func orderedJoin(_ tiers: [String]) -> String {
        var seen = Set<String>(); var out = [String]()
        for t in tiers where seen.insert(t).inserted { out.append(t) }
        return out.joined(separator: "+")
    }
}
