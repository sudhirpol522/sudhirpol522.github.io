---
layout: page
title: resume
permalink: /resume/
description: Resume and professional background for Sudhir Pol.
nav: true
nav_order: 5
---

<div class="resume-downloads">
  <h2>Resume versions</h2>
  <p>Select the version most relevant to the role.</p>
  <div class="resume-links">
    <a href="{{ '/assets/pdf/Sudhir_Pol_MLE.pdf' | relative_url }}">Machine Learning Engineer resume</a>
    <a href="{{ '/assets/pdf/Sudhir_Pol_LLM_Inference.pdf' | relative_url }}">LLM Inference resume</a>
    <a href="{{ '/assets/pdf/Sudhir_Pol_Data_Scientist.pdf' | relative_url }}">Data Scientist resume</a>
  </div>
</div>

## Professional summary

Machine Learning Engineer with 3 years of experience taking models from prototype to production at Adobe,
S and P Global, and American Express. Experienced across training, evaluation, containerization, deployment,
and monitoring on AWS and Kubernetes, with latency, cost, quality, and reliability treated as design constraints.

## Experience

{% for job in site.data.experience %}

### {{ job.company }}

**{{ job.position }}**

{{ job.period }}, {{ job.location }}

{% for highlight in job.highlights %}

- {{ highlight }}
  {% endfor %}

{% endfor %}

## Education

{% for item in site.data.education %}

### {{ item.institution }}

**{{ item.degree }}**

{{ item.period }}, {{ item.location }}

{% endfor %}

## Skills

{% for group in site.data.skills %}
**{{ group.category }}:** {{ group.items }}

{% endfor %}
