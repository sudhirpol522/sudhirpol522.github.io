import type { Metadata } from "next";
import { Article, formatDate, getPosts } from "@/lib/site";
import { notFound } from "next/navigation";

type Props = { params: Promise<{ slug: string }> };

export const dynamicParams = false;

export function generateStaticParams() {
  return getPosts().map((p) => ({ slug: p.slug }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const post = getPosts().find((p) => p.slug === slug);
  return { title: post?.title, description: post?.description };
}

export default async function PostPage({ params }: Props) {
  const { slug } = await params;
  const post = getPosts().find((p) => p.slug === slug);
  if (!post) notFound();
  return (
    <Article
      back={{ href: "/writing/", label: "All writing" }}
      meta={formatDate(post.date)}
      title={post.title}
      lead={post.description}
      html={post.html}
    />
  );
}
