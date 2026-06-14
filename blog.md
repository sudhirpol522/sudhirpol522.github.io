---
layout: page
title: Blog
permalink: /blog/
---

<p style="font-size:0.92rem;color:#4a5568;margin-bottom:4px">
  Deep-dive technical articles on <a href="https://sudhirpol522.medium.com" target="_blank">Medium</a>,
  implementing LLM inference primitives from scratch with full math derivations.
</p>

<div class="blog-group-label">LLM Inference Series</div>

{% for post in site.data.blog.llm_inference_series %}
<div class="blog-item"{% if forloop.first %} style="border-top:1px solid #eee"{% endif %}>
  <h3><a href="{{ post.url }}" target="_blank">{{ post.title }}</a></h3>
  <p>{{ post.description }}</p>
</div>
{% endfor %}

<div class="blog-group-label">Kaggle Competition Writeup &mdash; Bristol-Myers Squibb</div>

{% for post in site.data.blog.kaggle_writeups %}
<div class="blog-item"{% if forloop.first %} style="border-top:1px solid #eee"{% endif %}>
  <h3><a href="{{ post.url }}" target="_blank">{{ post.title }}</a></h3>
  <p>{{ post.description }}</p>
</div>
{% endfor %}
