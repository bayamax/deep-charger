import Foundation

// Faithful Swift port of canonical `hypernet_sp/calculator.py` — post-hoc arithmetic verification
// & repair. The 1.5B writes correct expressions then mis-evaluates them; this recomputes claims
// (`<expr> = <value>`) with a safe arithmetic evaluator and substitutes wrong final values.

/// Normalise an arithmetic expression (strip $ , %, unify operators, implicit mult).
private func normalise(_ s0: String) -> String {
    var s = s0.replacingOccurrences(of: ",", with: "").replacingOccurrences(of: "$", with: "")
        .replacingOccurrences(of: "%", with: "")
    for (a, b) in [("×", "*"), ("·", "*"), ("÷", "/"), ("−", "-"),
                   ("\\times", "*"), ("\\cdot", "*"), ("\\div", "/")] {
        s = s.replacingOccurrences(of: a, with: b)
    }
    // implicit mult: 2(3+4) -> 2*(3+4)
    if let re = try? NSRegularExpression(pattern: "(\\d)\\s*\\(") {
        let ns = s as NSString
        s = re.stringByReplacingMatches(in: s, range: NSRange(location: 0, length: ns.length),
                                        withTemplate: "$1*(")
    }
    return s.trimmingCharacters(in: .whitespaces)
}

/// Recursive-descent evaluator for + - * / ^ ( ) and unary signs. Returns nil on anything else.
func safeEval(_ expr: String) -> Double? {
    let s = Array(normalise(expr))
    var i = 0
    func skip() { while i < s.count, s[i] == " " { i += 1 } }
    func parseExpr() -> Double? {
        guard var v = parseTerm() else { return nil }
        while true {
            skip()
            guard i < s.count, s[i] == "+" || s[i] == "-" else { break }
            let op = s[i]; i += 1
            guard let r = parseTerm() else { return nil }
            v = op == "+" ? v + r : v - r
        }
        return v
    }
    func parseTerm() -> Double? {
        guard var v = parsePower() else { return nil }
        while true {
            skip()
            guard i < s.count, s[i] == "*" || s[i] == "/" else { break }
            let op = s[i]; i += 1
            guard let r = parsePower() else { return nil }
            if op == "/" { if r == 0 { return nil }; v /= r } else { v *= r }
        }
        return v
    }
    func parsePower() -> Double? {
        guard let b = parseUnary() else { return nil }
        skip()
        if i < s.count, s[i] == "^" {
            i += 1
            guard let e = parsePower() else { return nil }   // right-assoc
            return pow(b, e)
        }
        return b
    }
    func parseUnary() -> Double? {
        skip()
        if i < s.count, s[i] == "+" || s[i] == "-" {
            let op = s[i]; i += 1
            guard let v = parseUnary() else { return nil }
            return op == "-" ? -v : v
        }
        return parseAtom()
    }
    func parseAtom() -> Double? {
        skip()
        if i < s.count, s[i] == "(" {
            i += 1
            guard let v = parseExpr() else { return nil }
            skip()
            guard i < s.count, s[i] == ")" else { return nil }
            i += 1
            return v
        }
        var num = ""
        while i < s.count, s[i].isNumber || s[i] == "." { num.append(s[i]); i += 1 }
        return Double(num)
    }
    guard let v = parseExpr() else { return nil }
    skip()
    return i == s.count ? v : nil   // trailing garbage → invalid
}

private let numPat = "\\$?\\d[\\d,]*(?:\\.\\d+)?"
private let claimPat =
    "((?:\\$?\\d[\\d,]*(?:\\.\\d+)?|[()+\\-*/×÷·^\\s]|\\\\times|\\\\cdot|\\\\div){3,})=\\s*(\\$?\\d[\\d,]*(?:\\.\\d+)?(?:\\s*/\\s*\\$?\\d[\\d,]*(?:\\.\\d+)?)?)"

private func toValue(_ s: String) -> Double? {
    let n = normalise(s)
    if n.contains("/") { return safeEval(n) }
    return Double(n)
}

struct Claim { let expr: String; let claimedStr: String; let claimed: Double; let actual: Double; let ok: Bool }

func findClaims(_ text: String) -> [Claim] {
    guard let re = try? NSRegularExpression(pattern: claimPat) else { return [] }
    let ns = text as NSString
    var out = [Claim]()
    for m in re.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
        let expr = ns.substring(with: m.range(at: 1))
        let claimedStr = ns.substring(with: m.range(at: 2))
        if regexMatches(expr, "[+\\-*/^×÷]|\\\\times|\\\\div").isEmpty { continue }
        if regexMatches(expr, numPat).count < 2 { continue }
        guard let claimed = toValue(claimedStr), let actual = safeEval(expr) else { continue }
        let claimedTrim = claimedStr.trimmingCharacters(in: .whitespaces)
        let dec = claimedTrim.contains(".") ? (claimedTrim.split(separator: ".").last?.count ?? 0) : 0
        let ok: Bool
        if dec > 0 {
            let factor = pow(10.0, Double(dec))
            ok = abs((actual * factor).rounded() / factor - claimed) <= max(pow(10.0, -Double(dec)), abs(actual) * 1e-9)
        } else {
            ok = abs(actual - claimed) < max(0.5, abs(actual) * 1e-9) + 1e-9
        }
        out.append(Claim(expr: expr.trimmingCharacters(in: .whitespaces), claimedStr: claimedTrim,
                         claimed: claimed, actual: actual, ok: ok))
    }
    return out
}

private func fmt(_ v: Double) -> String {
    if abs(v - v.rounded()) < 1e-9 { return String(Int(v.rounded())) }
    return String(format: "%g", (v * 1e6).rounded() / 1e6)
}

/// `repair_answer`: if the answer's value equals a WRONG claim's RHS, substitute the recomputed
/// value. Returns (fixed, corrections).
public func repairAnswer(_ answer: String, fullBody: String? = nil) -> (fixed: String, corrections: [(String, Double, Double)]) {
    let scope = fullBody ?? answer
    let wrong = findClaims(scope).filter { !$0.ok }
    if wrong.isEmpty { return (answer, []) }
    var fixed = answer
    var corrections = [(String, Double, Double)]()
    for w in wrong {
        // candidate strings of the claimed value to find in the answer
        var cands = [w.claimedStr.replacingOccurrences(of: ",", with: "").replacingOccurrences(of: "$", with: ""),
                     fmt(w.claimed), String(format: "%g", w.claimed)]
        cands = cands.filter { !$0.isEmpty }
        let norm = fixed.replacingOccurrences(of: ",", with: "")
        for c in cands where norm.contains(c) {
            if let r = fixed.range(of: c) ?? norm.range(of: c) {
                fixed.replaceSubrange(r, with: fmt(w.actual))
                corrections.append((w.expr, w.claimed, w.actual))
                break
            }
        }
    }
    return (fixed, corrections)
}
