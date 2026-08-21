---
permalink: /
title: "Home"
author_profile: true
classes: wide
redirect_from: 
  - /about/
  - /about.html
---
I am Huang Yuqi, a Ph.D. student at The Hong Kong Polytechnic University, supervised by [Prof. Jing Li](https://www4.comp.polyu.edu.hk/~jing1li/). Before that, I worked as a Research Assistant in [Prof. Kam-Fai Wong](https://www.se.cuhk.edu.hk/people/academic-staff/prof-wong-kam-fai/)'s group at The Chinese University of Hong Kong. I received my bachelor's degree in Computer Science from Harbin Institute of Technology, Shenzhen, where I was supervised by [Prof. Ruifeng Xu](https://homepage.hit.edu.cn/xuruifeng) in the HLT Group.

## Education

- Ph.D. student, Department of Computing, The Hong Kong Polytechnic University (2026 - expected 2030)
- B.Eng. in Computer Science, Harbin Institute of Technology, Shenzhen (2021 - 2025)

## Experience

- Research Assistant, The Chinese University of Hong Kong (Sep. 2025 - Aug. 2026)  
  Supervisor: [Prof. Kam-Fai Wong](https://www.se.cuhk.edu.hk/people/academic-staff/prof-wong-kam-fai/)

## Publications

{% if site.author.googlescholar %}
  <div class="wordwrap">You can also find my articles on <a href="{{site.author.googlescholar}}">my Google Scholar profile</a>.</div>
{% endif %}

{% include base_path %}

<div class="publications">
{% for post in site.publications reversed %}
  {% include archive-single-cv.html %}
{% endfor %}
</div>
