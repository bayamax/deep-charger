import Foundation
import SwiftUI
#if canImport(Translation)
import Translation
#endif

/// On-device translation bridge for the JP↔EN shim. Wraps Apple's Translation framework
/// (macOS 15 / iOS 18+), which translates **on-device/offline** once the language pack has been
/// downloaded once (the framework prompts for that one-time download). If the framework or a pack
/// is unavailable, `translate` returns the input unchanged so the pipeline degrades to English
/// rather than breaking.
///
/// The Translation API is SwiftUI-coupled: a `.translationTask` in the view vends a
/// `TranslationSession` whenever `config` changes. We box `config` as `Any?` so this type still
/// compiles on the 14/16 deployment floor (a stored `TranslationSession.Configuration` would force
/// an `@available` the language allows only on the whole type).
@MainActor
final class TranslationService: ObservableObject {
    /// Bumped to re-drive the `.translationTask`; holds a `TranslationSession.Configuration` when available.
    @Published var config: Any?

    /// Whether on-device translation is usable on this OS build.
    static var isAvailable: Bool {
        if #available(macOS 15.0, iOS 18.0, *) { return true } else { return false }
    }

    /// True once both `lang`↔EN packs are installed (so no system prompt will appear on a turn).
    func packsInstalled(userLang lang: String) async -> Bool {
        let a = await availability(from: lang, to: "en")
        let b = await availability(from: "en", to: lang)
        return a == "installed" && b == "installed"
    }

    /// ONBOARDING: proactively trigger the one-time `lang`↔EN (core) pack download when the user picks
    /// their language, so the single system "Download" sheet appears up front. After the user taps
    /// Download once, every later turn is instant and fully offline. No-op once installed / for English.
    func prewarm(userLang lang: String) async {
        guard Self.isAvailable, lang != "en", !lang.isEmpty else { return }
        if await packsInstalled(userLang: lang) { return }
        _ = await translate("Hello, this downloads the language pack.", from: lang, to: "en")  // lang→en
        _ = await translate("Hello, this downloads the language pack.", from: "en", to: lang)  // en→lang
    }

    /// Pack status for a language pair: "installed" / "supported (needs download)" / "unsupported".
    func availability(from: String, to: String) async -> String {
        #if canImport(Translation)
        if #available(macOS 15.0, iOS 18.0, *) {
            let s = await LanguageAvailability().status(
                from: Locale.Language(identifier: from), to: Locale.Language(identifier: to))
            switch s {
            case .installed: return "installed"
            case .supported: return "supported (needs one-time download)"
            case .unsupported: return "unsupported"
            @unknown default: return "unknown"
            }
        }
        #endif
        return "framework-unavailable"
    }

    // class (not struct) so the timeout path and the session path share one resume guard
    private final class Req {
        let text: String
        let cont: CheckedContinuation<String, Never>
        var done = false
        init(_ text: String, _ cont: CheckedContinuation<String, Never>) { self.text = text; self.cont = cont }
    }
    private var queue: [Req] = []
    private var current: (from: String, to: String)?
    /// Per-turn ceiling: never hang the turn. Generous enough to cover a first-time language-pack
    /// download (the user taps "Download" on the system prompt), then falls back to passthrough.
    private let timeout: TimeInterval = 25

    private func resume(_ r: Req, _ s: String) {
        guard !r.done else { return }
        r.done = true
        r.cont.resume(returning: s)
    }

    /// Translate `text` from→to (BCP-47 codes, e.g. "ja"/"en"). Returns the input unchanged if
    /// translation is unavailable or fails.
    func translate(_ text: String, from: String, to: String) async -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard Self.isAvailable, !trimmed.isEmpty else { return text }
        #if canImport(Translation)
        if #available(macOS 15.0, iOS 18.0, *) {
            return await withCheckedContinuation { (cont: CheckedContinuation<String, Never>) in
                let req = Req(text, cont)
                queue.append(req)
                // hard timeout → fall back to the original text rather than hang the turn
                Task { [weak self] in
                    try? await Task.sleep(nanoseconds: UInt64((self?.timeout ?? 6) * 1_000_000_000))
                    self?.resume(req, text)
                }
                let src = Locale.Language(identifier: from)
                let tgt = Locale.Language(identifier: to)
                if current?.from == from, current?.to == to,
                   var existing = config as? TranslationSession.Configuration {
                    existing.invalidate()              // same pair → force the task to re-run
                    config = existing
                } else {
                    current = (from, to)
                    config = TranslationSession.Configuration(source: src, target: tgt)
                }
            }
        }
        #endif
        return text
    }

    #if canImport(Translation)
    /// Called by the view's `.translationTask` with a live session; drains the pending queue.
    @available(macOS 15.0, iOS 18.0, *)
    func run(_ session: TranslationSession) async {
        let batch = queue; queue = []
        // Explicitly prepare (downloads the pack with a system prompt if not installed).
        try? await session.prepareTranslation()
        for req in batch where !req.done {
            do {
                let resp = try await session.translate(req.text)
                resume(req, resp.targetText)
            } catch {
                resume(req, req.text)   // fallback: original text, never lost
            }
        }
    }
    #endif
}

extension View {
    /// Attach the translation driver. No-op (passthrough) on OS builds without the framework.
    @ViewBuilder
    func translationDriver(_ service: TranslationService) -> some View {
        #if canImport(Translation)
        if #available(macOS 15.0, iOS 18.0, *) {
            self.translationTask(service.config as? TranslationSession.Configuration) { session in
                await service.run(session)
            }
        } else {
            self
        }
        #else
        self
        #endif
    }
}
