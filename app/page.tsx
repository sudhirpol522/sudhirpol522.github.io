import Link from "next/link";
import { Rich, formatDate, getPosts, getProjects } from "@/lib/site";
import { education, experience, interests, openSource, profile } from "@/content/profile";

export default function Home() {
  const projects = getProjects().slice(0, 4);
  const posts = getPosts().slice(0, 3);

  return (
    <>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={profile.image} alt={profile.name} className="avatar" />
      <h1 className="role">{profile.title}</h1>
      <p className="muted">{profile.focus}</p>
      {profile.summary.map((p) => (
        <p key={p}>
          <Rich text={p} />
        </p>
      ))}
      <p>
        {profile.links.map((l, i) => (
          <span key={l.label}>
            {i > 0 && " · "}
            <a href={l.href}>{l.text ?? l.label}</a>
          </span>
        ))}
      </p>

      <h2>Experience</h2>
      <ul>
        {experience.map((job) => (
          <li key={job.company}>
            <strong>{job.company}</strong>, <em>{job.role}</em> <span className="muted">({job.period})</span>
            <br />
            <Rich text={job.summary} />
          </li>
        ))}
      </ul>

      <h2>Open source</h2>
      <ul>
        {openSource.map((c) => (
          <li key={c.project}>
            <strong>{c.project}</strong>: {c.title} ({c.state.toLowerCase()},{" "}
            {c.prs.map((pr, i) => (
              <span key={pr.href}>
                {i > 0 && ", "}
                <a href={pr.href}>{pr.ref}</a>
              </span>
            ))}
            ).
            <br />
            <Rich text={c.summary} />
          </li>
        ))}
      </ul>

      <h2>Research interests</h2>
      <ul>
        {interests.map((i) => (
          <li key={i}>{i}</li>
        ))}
      </ul>

      <h2>Selected projects</h2>
      <ul>
        {projects.map((p) => (
          <li key={p.slug}>
            <Link href={`/projects/${p.slug}/`}>{p.title}</Link>: {p.highlight}.
          </li>
        ))}
      </ul>
      <p>
        <Link href="/projects/">All projects →</Link>
      </p>

      <h2>Recent writing</h2>
      <ol className="citations">
        {posts.map((post) => (
          <li key={post.slug}>
            {profile.name}. <Link href={`/blog/${post.slug}/`}>&ldquo;{post.title}.&rdquo;</Link>{" "}
            <em>Technical blog</em>, {formatDate(post.date, "short")}.
          </li>
        ))}
      </ol>

      <h2>Education</h2>
      <ul>
        {education.map((e) => (
          <li key={e.institution}>
            <strong>{e.institution}</strong>, {e.degree} <span className="muted">({e.period})</span>
          </li>
        ))}
      </ul>
    </>
  );
}
