---
layout: page
title: experience
permalink: /experience/
description: Production machine learning experience across evaluation, document intelligence, and applied NLP.
nav: true
nav_order: 2
---

<div class="section-intro">
  Production ML roles at Adobe, S&P Global, and American Express — from evaluation design and document AI to deployment and monitoring.
</div>

{% for job in site.data.experience %}

<article class="experience-entry">
  <header class="experience-header">
    <div>
      <h2>{{ job.company }}</h2>
      <p class="experience-role">{{ job.position }}</p>
    </div>
    <div class="experience-meta">
      <span>{{ job.period }}</span>
      <span>{{ job.location }}</span>
    </div>
  </header>
  <ul>
    {% for highlight in job.highlights %}
    <li>{{ highlight }}</li>
    {% endfor %}
  </ul>
</article>
{% endfor %}
