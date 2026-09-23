import type { Metadata } from "next";
import Link from "next/link";
import { profile } from "@/content/profile";
import { formatDate, getPosts } from "@/lib/site";

export const metadata: Metadata = {
  title: "Writing",
  description: "Technical articles on quantization, speculative decoding, and LLM serving.",
};

export default function WritingPage() {
  return (
    <>
      <h1>Writing</h1>
      <p>
        Technical notes connecting mathematical derivations to working implementations. More on{" "}
        <a href="https://sudhirpol522.medium.com">Medium</a>.
      </p>
      <ol className="citations">
        {getPosts().map((post) => (
          <li key={post.slug}>
            {profile.name}. <Link href={`/blog/${post.slug}/`}>&ldquo;{post.title}.&rdquo;</Link>{" "}
            <em>Technical blog</em>, {formatDate(post.date)}.
            <br />
            <span className="muted">{post.description}</span>
          </li>
        ))}
      </ol>
    </>
  );
}
