import SwiftUI
import LaTeXSwiftUI
import UniformTypeIdentifiers
#if canImport(UIKit)
import UIKit
#endif

/// The bundled MathJax doesn't render `\boxed{}` (AMS), so the model's boxed final answers would show
/// as raw LaTeX. Rewrite `\boxed{X}` → `\mathbf{X}` so the answer renders (bold) instead of raw.
private func renderMath(_ s: String) -> String {
    if s.isEmpty { return "—" }
    return s.replacingOccurrences(of: #"\\boxed\s*\{([^{}]*)\}"#,
                                  with: #"\\mathbf{$1}"#, options: .regularExpression)
}

struct ContentView: View {
    @StateObject private var app = AppModel()
    @State private var showLang = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            transcript
            Divider()
            inputBar
        }
        .onAppear {
            #if canImport(UIKit)
            UIApplication.shared.isIdleTimerDisabled = true  // keep screen on (GPU needs foreground)
            #endif
            app.load()
        }
        // First-launch (or re-pick) language onboarding: "What language do you speak?"
        .overlay {
            if app.needsLanguagePick || showLang {
                LanguagePicker { code in app.chooseLanguage(code); showLang = false }
            }
        }
        // Driver must live in a view that DIRECTLY observes the service — a nested ObservableObject
        // doesn't republish through `app`, so config changes wouldn't otherwise re-run the task.
        .background(TranslationDriverHost(service: app.translation))
    }

    // MARK: header

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Deep Charger — on-device").font(.headline)
                Text(statusLine).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if case .loading = app.phase { ProgressView().controlSize(.small) }
            // Current language → tap to re-pick (fully offline, no web in this build).
            if app.translationSupported {
                Button { showLang = true } label: {
                    Label(app.userLang.isEmpty ? "Lang" : app.userLang, systemImage: "globe")
                        .font(.caption)
                }
                .buttonStyle(.bordered).controlSize(.small)
                .help("Your language (model runs in English)")
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
    }

    private var statusLine: String {
        switch app.phase {
        case .idle: return "starting…"
        case .loading: return "loading model on \(app.deviceName)…"
        case .ready: return "ready · \(app.deviceName)"
        case .error(let e): return "error: \(e)"
        }
    }

    // MARK: transcript

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    ForEach(app.messages) { MessageRow(msg: $0) }
                    Color.clear.frame(height: 1).id("bottom")
                }
                .padding(12)
            }
            .onChange(of: app.messages.count) { _ in withAnimation { proxy.scrollTo("bottom", anchor: .bottom) } }
            .onChange(of: app.messages.last?.text) { _ in proxy.scrollTo("bottom", anchor: .bottom) }
        }
    }

    // MARK: input

    private var inputBar: some View {
        HStack(spacing: 8) {
            TextField("Message…", text: $app.input, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...5)
                .disabled(!isReady)
                .onSubmit { if !app.busy { app.generate() } }
            if app.busy {
                Button(role: .destructive) { app.cancel() } label: {
                    Image(systemName: "stop.fill").imageScale(.large)
                }
                .help("Stop")
            } else {
                Button { app.generate() } label: {
                    Image(systemName: "paperplane.fill").imageScale(.large)
                }
                .disabled(!isReady || app.input.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(12)
    }

    private var isReady: Bool { if case .ready = app.phase { return true } else { return false } }
}

/// One chat bubble (user right / assistant left / system centered), with intent + source + meta.
private struct MessageRow: View {
    let msg: ChatMessage

    var body: some View {
        switch msg.role {
        case .system:
            Text(msg.text)
                .font(.caption).foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .center)
                .padding(.vertical, 2)
        case .user:
            HStack {
                Spacer(minLength: 40)
                Text(msg.text)
                    .padding(.horizontal, 12).padding(.vertical, 8)
                    .background(Color.accentColor.opacity(0.85))
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                    .textSelection(.enabled)
            }
        case .assistant:
            VStack(alignment: .leading, spacing: 4) {
                if let label = topLabel {
                    Text(label).font(.caption2).foregroundStyle(.secondary)
                }
                Group {
                    if msg.pending {
                        ThinkingDots()
                    } else {
                        // Render the model's LaTeX (\(…\), \[…\], \boxed{…}) as real math; plain prose
                        // and code stay as text. .original error mode falls back to raw text on any
                        // unparseable fragment (the 1.5B sometimes emits partial LaTeX).
                        LaTeX(renderMath(msg.text))
                            .errorMode(.original)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .padding(.horizontal, 12).padding(.vertical, 8)
                .background(secondaryBubble)
                .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                if let meta = msg.meta, !msg.pending {
                    Text(meta).font(.caption2).foregroundStyle(.tertiary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.trailing, 40)
        }
    }

    private var topLabel: String? {
        var parts = [String]()
        if let i = msg.intent { parts.append(i) }
        if let s = msg.source { parts.append(s) }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    private var secondaryBubble: Color {
        #if canImport(UIKit)
        return Color(UIColor.secondarySystemBackground)
        #else
        return Color.gray.opacity(0.15)
        #endif
    }
}

/// First-launch onboarding: "What language do you speak?" The model always runs in English; a
/// non-English choice downloads that language's pack (+ English) once and translates each turn.
private struct LanguagePicker: View {
    let onPick: (String) -> Void
    private let langs: [(code: String, name: String)] = [
        ("en", "English"), ("ja", "日本語"), ("zh-Hans", "中文"), ("ko", "한국어"),
        ("es", "Español"), ("fr", "Français"), ("de", "Deutsch"), ("pt-BR", "Português"),
        ("it", "Italiano"), ("ru", "Русский"),
    ]
    var body: some View {
        ZStack {
            Color.black.opacity(0.001).ignoresSafeArea()   // capture taps below
            Rectangle().fill(.ultraThinMaterial).ignoresSafeArea()
            VStack(spacing: 16) {
                Text("What language do you speak?").font(.title3).bold()
                Text("The assistant runs in English. We'll download your language pack (and English) once, then translate for you.")
                    .font(.caption).foregroundStyle(.secondary)
                    .multilineTextAlignment(.center).padding(.horizontal, 24)
                ScrollView {
                    VStack(spacing: 8) {
                        ForEach(langs, id: \.code) { l in
                            Button { onPick(l.code) } label: {
                                Text(l.name).frame(maxWidth: .infinity).padding(.vertical, 12)
                            }
                            .buttonStyle(.borderedProminent)
                        }
                    }
                    .padding(.horizontal, 32)
                }
            }
            .padding(.vertical, 28)
            .frame(maxWidth: 420)
        }
    }
}

/// Hosts the `.translationTask` in a view that directly observes the service, so setting
/// `service.config` actually re-drives the task (and fires the system download prompt).
private struct TranslationDriverHost: View {
    @ObservedObject var service: TranslationService
    var body: some View {
        Color.clear.frame(width: 0, height: 0).translationDriver(service)
    }
}

/// Animated "typing" indicator shown while the assistant generates.
private struct ThinkingDots: View {
    @State private var phase = 0
    private let timer = Timer.publish(every: 0.4, on: .main, in: .common).autoconnect()
    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<3) { i in
                Circle().frame(width: 6, height: 6)
                    .opacity(phase == i ? 1 : 0.3)
            }
        }
        .foregroundStyle(.secondary)
        .onReceive(timer) { _ in phase = (phase + 1) % 3 }
    }
}
