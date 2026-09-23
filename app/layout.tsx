import type { Metadata } from "next";
import Link from "next/link";
import { Inconsolata } from "next/font/google";
import "katex/dist/katex.min.css";
import "./globals.css";
import { profile } from "@/content/profile";

const inconsolata = Inconsolata({ subsets: ["latin"], weight: ["400", "700"], variable: "--font-inconsolata" });

export const metadata: Metadata = {
  metadataBase: new URL("https://sudhirpol522.github.io"),
  title: { default: `${profile.name} · ${profile.title}`, template: `%s · ${profile.name}` },
  description: `${profile.name}, ${profile.title} working on LLM inference, agentic AI, and model evaluation.`,
  openGraph: { images: ["/assets/img/og.png"] },
};

const nav = [
  { href: "/", label: "about" },
  { href: "/projects/", label: "projects" },
  { href: "/writing/", label: "writing" },
];

// Runs before paint so a saved dark preference never flashes light. Light is the default.
const themeScript = `
try { if (localStorage.getItem("theme") === "dark") document.documentElement.classList.add("dark"); } catch (e) {}
document.addEventListener("click", function (e) {
  if (!e.target.closest || !e.target.closest("#theme-toggle")) return;
  var dark = document.documentElement.classList.toggle("dark");
  try { localStorage.setItem("theme", dark ? "dark" : "light"); } catch (e) {}
});`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning className={inconsolata.variable}>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>
        <div className="container">
          <header className="site-header">
            <Link href="/" className="site-name">
              {profile.name}
            </Link>
            <nav>
              {nav.map((item) => (
                <Link key={item.href} href={item.href}>
                  {item.label}
                </Link>
              ))}
              <button id="theme-toggle" type="button" aria-label="Toggle dark mode">
                <span className="when-light">dark</span>
                <span className="when-dark">light</span>
              </button>
            </nav>
          </header>
          <main className="content">{children}</main>
          <footer className="site-footer muted">
            © {new Date().getFullYear()} {profile.name} ·{" "}
            {profile.links.map((l, i) => (
              <span key={l.label}>
                {i > 0 && " · "}
                <a href={l.href}>
                  {l.label}
                </a>
              </span>
            ))}
          </footer>
        </div>
      </body>
    </html>
  );
}
