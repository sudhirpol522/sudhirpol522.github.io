---
layout: page
title: writing
permalink: /writing/
description: Technical articles on quantization, speculative decoding, and LLM serving.
nav: true
nav_order: 4
---

<div class="section-intro">
  I write technical explanations that connect mathematical derivations to working implementations.
  Additional articles are available on <a href="https://sudhirpol522.medium.com">Medium</a>.
</div>

<div class="writing-list">
  {% assign sorted_posts = site.posts | sort: "date" | reverse %}
  {% for post in sorted_posts %}
  <article class="writing-entry">
    <p class="writing-date">{{ post.date | date: "%B %Y" }}</p>
    <h2><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
    <p>{{ post.description }}</p>
    <a class="text-link" href="{{ post.url | relative_url }}">Read article</a>
  </article>
  {% endfor %}
</div>
