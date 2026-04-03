---
layout: page
title: Experience
permalink: /experience/
---

<h2 class="section-label">Experience</h2>

{% for job in site.data.experience %}
<article class="card">
  <h3>
    {{ job.title }} &mdash;
    {% if job.url %}<a href="{{ job.url }}" target="_blank">{{ job.company }}</a>{% else %}{{ job.company }}{% endif %}
  </h3>
  <div class="meta">
    {{ job.period }}
    {% if job.duration %}&nbsp;&middot;&nbsp; {{ job.duration }}{% endif %}
    &nbsp;&middot;&nbsp; {{ job.location }}
    {% if job.type %}&nbsp;&middot;&nbsp; {{ job.type }}{% endif %}
  </div>
  <p>{{ job.description }}</p>
</article>
{% endfor %}
