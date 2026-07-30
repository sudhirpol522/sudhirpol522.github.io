---
layout: page
title: resume
permalink: /resume/
description: Resume and professional background for Sudhir Pol.
nav: true
nav_order: 5
---

<div class="resume-downloads">
  <h2>Download</h2>
  <p>Role-specific resumes drawn from the same experience and project history.</p>
  <div class="resume-links">
    <a href="{{ '/assets/pdf/Sudhir_Pol_MLE.pdf' | relative_url }}">Machine Learning Engineer</a>
    <a href="{{ '/assets/pdf/Sudhir_Pol_LLM_Inference.pdf' | relative_url }}">LLM Inference</a>
    <a href="{{ '/assets/pdf/Sudhir_Pol_Data_Scientist.pdf' | relative_url }}">Data Scientist</a>
    <a href="{{ '/assets/pdf/Sudhir_Pol.pdf' | relative_url }}">General / AI Engineer</a>
  </div>
</div>

## Professional summary

Machine Learning Engineer with 3 years of experience designing, building, and deploying models in production for cross-functional teams. Covers the full path from data preprocessing and feature engineering to evaluation, fine-tuning, and continuous delivery on AWS with Docker, Kubernetes, and automated CI/CD. Currently completing an M.S. in Data Science at Indiana University (May 2026).

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
