---
layout: page
title: Projects
permalink: /projects/
---

<h2 class="section-label">Some of my work</h2>

{% for project in site.data.projects %}
<article class="card">
  <h3>
    {% if project.github %}<a href="{{ project.github }}" target="_blank">{{ project.title }}</a>{% else %}{{ project.title }}{% endif %}
  </h3>
  <div class="meta">
    {% for tag in project.tags %}<span class="tag">{{ tag }}</span>{% endfor %}
    {% if project.date %}<span>{{ project.date }}</span>{% endif %}
  </div>
  <p>{{ project.description }}</p>
  <div class="card-links">
    {% if project.github %}<a href="{{ project.github }}" target="_blank">→ GitHub</a>{% endif %}
    {% if project.blog_part1 %}<a href="{{ project.blog_part1 }}" target="_blank">→ Blog Part 1</a>{% endif %}
    {% if project.blog_part2 %}<a href="{{ project.blog_part2 }}" target="_blank">→ Blog Part 2</a>{% endif %}
  </div>
</article>
{% endfor %}
