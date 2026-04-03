---
layout: page
title: Education
permalink: /education/
---

<h2 class="section-label">Education</h2>

{% for edu in site.data.education %}
<article class="card">
  <h3>
    {{ edu.degree }} &mdash;
    {% if edu.url %}<a href="{{ edu.url }}" target="_blank">{{ edu.institution }}</a>{% else %}{{ edu.institution }}{% endif %}
  </h3>
  <div class="meta">
    {{ edu.period }} &nbsp;&middot;&nbsp; {{ edu.location }}
    {% if edu.gpa %}&nbsp;&middot;&nbsp; GPA: {{ edu.gpa }}{% endif %}
  </div>
  {% if edu.description %}<p>{{ edu.description }}</p>{% endif %}
</article>
{% endfor %}
