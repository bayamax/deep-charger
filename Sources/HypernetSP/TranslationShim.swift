import Foundation

/// Faithful Swift port of canonical `hypernet_sp/translation_shim.py` — the "JP→EN-only-inside"
/// architecture: the user types their language, the whole reasoning pipeline runs in English, and
/// only the final answer is rendered back. The CoT is never translated.
///
/// The invariant a translator must NOT break is **verbatim values** (quoted strings, codes/IDs,
/// money/numbers): we extract those spans BEFORE translation, swap in `[[i]]` placeholders the NMT
/// copies through, and restore the ORIGINAL surface after — so a passphrase `「さくら」` round-trips
/// exactly, not as さくら/サクラ/桜 by luck.
public enum TranslationShim {

    // Only GENUINE verbatim values that a translator would otherwise alter. Deliberately tighter
    // than the opus-mt-era canonical floor: Apple's on-device NMT mangles dense `[[N]]` placeholders
    // (nests/duplicates them), so masking ordinary capitalized words and bare numbers backfired —
    // those should translate (words) or are preserved natively by the NMT (numbers). We protect:
    //   • quoted strings (JP + latin) — passphrases like 「さくら」
    //   • mixed alphanumeric codes (must contain BOTH a letter and a digit) — EMP-4471, QX7-2291
    //   • money ($650) and numbers with an explicit unit (450円, 50%)
    // Protect ONLY genuine verbatim tokens a translator would corrupt: quoted passphrases and mixed
    // alphanumeric codes (EMP-4471). Bare quantities ("120円", "$650", "50%") are NOT masked — Apple's
    // on-device NMT preserves digits natively, and masking them BROKE arithmetic: the model never saw
    // the numbers (a word problem like "120円×5, おつり?" became "[[1]]×5" placeholder soup that the NMT
    // mangled, e.g. into a stray "ter"). Numbers must reach the model intact.
    private static let protectPattern =
        "「[^」]{1,40}」|『[^』]{1,40}』|\"[^\"]{1,40}\"|'[^']{1,40}'"
        + "|(?<![A-Za-z0-9])(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*[0-9])[A-Za-z0-9][A-Za-z0-9-]*(?![A-Za-z0-9])"

    private static let protectRE = try! NSRegularExpression(pattern: protectPattern)
    private static let phFindRE = try! NSRegularExpression(pattern: "\\[\\[\\s?(\\d+)\\s?\\]\\]")

    /// Extract protected spans and replace them with `[[i]]` placeholders.
    /// Returns the masked text and the index→original-surface mapping.
    public static func protect(_ text: String, extra: [String] = []) -> (masked: String, mapping: [Int: String]) {
        let ns = text as NSString
        var found = protectRE.matches(in: text, range: NSRange(location: 0, length: ns.length))
            .map { ns.substring(with: $0.range) }
        found.append(contentsOf: extra)
        // dedup preserving first-seen order, then longest-first so substrings don't clobber
        var seen = Set<String>()
        var spans = found.filter { !$0.isEmpty && seen.insert($0).inserted }
        spans.sort { $0.count > $1.count }

        var mapping = [Int: String]()
        var out = text
        for (i, s) in spans.enumerated() where out.contains(s) {
            mapping[i] = s
            out = out.replacingOccurrences(of: s, with: "[[\(i)]]")
        }
        return (out, mapping)
    }

    /// Swap `[[i]]` placeholders back to their original surfaces. Any placeholder the translator
    /// dropped has its VALUE appended in parentheses so it is never lost.
    public static func restore(_ text: String, mapping: [Int: String]) -> String {
        let ns = text as NSString
        var seen = Set<Int>()
        var out = ""
        var last = 0
        for m in phFindRE.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            out += ns.substring(with: NSRange(location: last, length: m.range.location - last))
            let idx = Int(ns.substring(with: m.range(at: 1))) ?? -1
            if let orig = mapping[idx] { seen.insert(idx); out += orig }
            else { out += ns.substring(with: m.range) }
            last = m.range.location + m.range.length
        }
        out += ns.substring(from: last)
        let missing = mapping.keys.filter { !seen.contains($0) }.sorted().map { mapping[$0]! }
        if !missing.isEmpty {
            out = out.trimmingCharacters(in: .whitespacesAndNewlines) + " (" + missing.joined(separator: ", ") + ")"
        }
        return out
    }

    /// A translator: `(text, sourceLang, targetLang) -> translatedText`. Injected so the engine
    /// (Apple Translation on-device, a mock in tests) is swappable.
    public typealias Translate = (String, String, String) async -> String

    /// User language → core EN, verbatim spans preserved byte-for-byte.
    public static func inbound(_ userText: String, userLang: String = "ja", coreLang: String = "en",
                               extra: [String] = [], translate: Translate) async -> String {
        let (masked, mapping) = protect(userText, extra: extra)
        let en = await translate(masked, userLang, coreLang)
        return restore(en, mapping: mapping)
    }

    /// Core EN answer → user language for display. Run AFTER the groundedness gate.
    public static func outbound(_ enAnswer: String, userLang: String = "ja", coreLang: String = "en",
                                extra: [String] = [], translate: Translate) async -> String {
        let (masked, mapping) = protect(enAnswer, extra: extra)
        let jp = await translate(masked, coreLang, userLang)
        return restore(jp, mapping: mapping)
    }

    /// True if the text contains Japanese (hiragana/katakana/CJK) — used to decide whether to
    /// route a turn through the shim at all.
    public static func isJapanese(_ text: String) -> Bool {
        text.unicodeScalars.contains { s in
            (0x3040...0x309F).contains(s.value) ||   // hiragana
            (0x30A0...0x30FF).contains(s.value) ||   // katakana
            (0x4E00...0x9FFF).contains(s.value)      // CJK unified
        }
    }
}
