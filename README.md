# Sudhir Pol Portfolio

Personal site for Sudhir Pol, built with Next.js and plain CSS and exported as static HTML for GitHub Pages.

## Local development

Requires Node.js 20.9 or newer.

```bash
npm install
npm run dev      # http://localhost:3000
npm run build    # static site in out/
```

## Content

- `content/profile.ts`: summary, experience summaries, open source, interests, earlier projects, education (wrap text in `**...**` to highlight it)
- `app/`: one folder per page (home, projects, writing, blog); `lib/site.tsx` loads markdown and holds shared components; `app/globals.css` is the only stylesheet
- `content/projects/*.md`: project pages (`/projects/<file-name>/`)
- `content/posts/*.md`: articles (`/blog/<file-name>/`), with `$$...$$` math rendered by KaTeX
- `public/assets`: images

## Deployment

Pushing to `main` runs `.github/workflows/deploy.yml`, which builds the site and publishes `out/` to the `gh-pages` branch.
