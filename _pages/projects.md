---
layout: page
title: projects
permalink: /projects/
description: Selected work in agent systems, LLM inference, model serving, retrieval, and applied machine learning.
nav: true
nav_order: 3
---

<div class="section-intro">
  Projects and professional case studies spanning human-in-the-loop agents, MCP-integrated workflows, inference systems, quantization, retrieval, and applied machine learning — from architecture through evaluation and deployment.
</div>

<div class="portfolio-grid">
  {% assign sorted_projects = site.projects | sort: "importance" %}
  {% for project in sorted_projects %}
  <article class="portfolio-card">
    <p class="portfolio-card-label">{{ project.area }}</p>
    <h2><a href="{{ project.url | relative_url }}">{{ project.title }}</a></h2>
    <p>{{ project.description }}</p>
    <p class="portfolio-card-stack">{{ project.stack }}</p>
    <a class="text-link" href="{{ project.url | relative_url }}">View details</a>
  </article>
  {% endfor %}
</div>
