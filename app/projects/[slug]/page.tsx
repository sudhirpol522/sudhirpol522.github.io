import type { Metadata } from "next";
import { Article, getProjects } from "@/lib/site";
import { notFound } from "next/navigation";

type Props = { params: Promise<{ slug: string }> };

export const dynamicParams = false;

export function generateStaticParams() {
  return getProjects().map((p) => ({ slug: p.slug }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const project = getProjects().find((p) => p.slug === slug);
  return { title: project?.title, description: project?.description };
}

export default async function ProjectPage({ params }: Props) {
  const { slug } = await params;
  const project = getProjects().find((p) => p.slug === slug);
  if (!project) notFound();
  return (
    <Article
      back={{ href: "/projects/", label: "All projects" }}
      meta={`${project.area} · ${project.stack.join(", ")}`}
      title={project.title}
      lead={project.description}
      html={project.html}
    />
  );
}
