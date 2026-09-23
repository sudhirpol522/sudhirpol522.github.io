import Link from "next/link";
import { Fragment } from "react";
import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import remarkRehype from "remark-rehype";
import rehypeKatex from "rehype-katex";
import rehypeStringify from "rehype-stringify";

export type Project = {
  slug: string;
  title: string;
  description: string;
  area: string;
  stack: string[];
  highlight: string;
  importance: number;
  html: string;
};

export type Post = {
  slug: string;
  title: string;
  description: string;
  date: Date;
  html: string;
};

const root = path.join(process.cwd(), "content");

const markdown = unified()
  .use(remarkParse)
  .use(remarkGfm)
  .use(remarkMath)
  .use(remarkRehype, { allowDangerousHtml: true })
  .use(rehypeKatex, { strict: false })
  .use(rehypeStringify, { allowDangerousHtml: true });

function readDir(dir: string) {
  const full = path.join(root, dir);
  return fs
    .readdirSync(full)
    .filter((f) => f.endsWith(".md"))
    .map((file) => {
      const { data, content } = matter(fs.readFileSync(path.join(full, file), "utf8"));
      // Kramdown treats a line that is only $$...$$ as a display equation; remark-math needs the fences on their own lines.
      const source = content.replace(/^[ \t]*\$\$([^\n]+?)\$\$[ \t]*$/gm, "$$$$\n$1\n$$$$");
      return { slug: file.replace(/\.md$/, ""), data, html: String(markdown.processSync(source)) };
    });
}

export function getProjects(): Project[] {
  return readDir("projects")
    .map(({ slug, data, html }) => ({
      slug,
      html,
      title: data.title,
      description: data.description,
      area: data.area,
      stack: String(data.stack ?? "").split(",").map((s) => s.trim()).filter(Boolean),
      highlight: data.highlight ?? "",
      importance: data.importance ?? 99,
    }))
    .sort((a, b) => a.importance - b.importance);
}

export function getPosts(): Post[] {
  return readDir("posts")
    .map(({ slug, data, html }) => ({
      slug,
      html,
      title: data.title,
      description: data.description,
      date: new Date(data.date),
    }))
    .sort((a, b) => b.date.getTime() - a.date.getTime());
}

export function formatDate(date: Date, style: "long" | "short" = "long") {
  return date.toLocaleDateString("en-US", {
    timeZone: "UTC",
    year: "numeric",
    month: style === "long" ? "long" : "short",
    day: style === "long" ? "numeric" : undefined,
  });
}

/** Renders **bold** keywords and `code` spans from plain content strings. */
export function Rich({ text }: { text: string }) {
  return (
    <>
      {text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, i) =>
        part.startsWith("**") ? (
          <strong key={i}>{part.slice(2, -2)}</strong>
        ) : part.startsWith("`") ? (
          <code key={i}>{part.slice(1, -1)}</code>
        ) : (
          <Fragment key={i}>{part}</Fragment>
        )
      )}
    </>
  );
}


export function Article(props: { back: { href: string; label: string }; meta: string; title: string; lead: string; html: string }) {
  return (
    <article>
      <Link href={props.back.href}>← {props.back.label}</Link>
      <h1>{props.title}</h1>
      <p className="muted">{props.meta}</p>
      <p className="lead">{props.lead}</p>
      <hr/>
      <div className="prose" dangerouslySetInnerHTML={{ __html: props.html }} />
    </article>
  );
}
