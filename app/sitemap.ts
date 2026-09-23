import type { MetadataRoute } from "next";
import { getPosts, getProjects } from "@/lib/site";

export const dynamic = "force-static";

const base = "https://sudhirpol522.github.io";

export default function sitemap(): MetadataRoute.Sitemap {
  return [
    { url: `${base}/` },
    { url: `${base}/writing/` },
    { url: `${base}/projects/` },
    ...getProjects().map((p) => ({ url: `${base}/projects/${p.slug}/` })),
    ...getPosts().map((p) => ({ url: `${base}/blog/${p.slug}/`, lastModified: p.date })),
  ];
}
