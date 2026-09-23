import type { Metadata } from "next";
import Link from "next/link";
import { Rich, getProjects } from "@/lib/site";
import { earlierProjects } from "@/content/profile";

export const metadata: Metadata = {
  title: "Projects",
  description: "Projects in agent systems, LLM inference, model serving, retrieval, and applied machine learning.",
};

export default function ProjectsPage() {
  return (
    <>
      <h1>Projects</h1>
      <ul className="plain">
        {getProjects().map((p) => (
          <li key={p.slug} className="spaced">
            <Link href={`/projects/${p.slug}/`}>{p.title}</Link> <span className="muted">({p.area})</span>
            <br />
            {p.description} <strong>{p.highlight}.</strong>
            <br />
            <span className="muted">{p.stack.join(", ")}</span>
          </li>
        ))}
      </ul>

      <h2>Earlier work</h2>
      <ul className="plain">
        {earlierProjects.map((p) => (
          <li key={p.title} className="spaced">
            <strong>{p.title}</strong> <span className="muted">({p.year})</span>. <Rich text={p.detail} />
          </li>
        ))}
      </ul>
    </>
  );
}
