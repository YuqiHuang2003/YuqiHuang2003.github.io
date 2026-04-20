---
permalink: /
title: "Home"
author_profile: false
classes: wide
redirect_from: 
  - /about/
  - /about.html
---
I am Huang Yuqi, currently a Research Assistant in [Prof. Kam-Fai Wong](https://kfwong.cintec.cuhk.edu.hk/)'s group at CUHK. I received my bachelor's degree from Harbin Institute of Technology, Shenzhen, where I was supervised by [Prof. Ruifeng Xu](https://homepage.hit.edu.cn/xuruifeng) in the HLT Group.

## Education

- B.Eng. in Computer Science, Harbin Institute of Technology, Shenzhen (2021-2025)

## Experience

- Research Assistant, The Chinese University of Hong Kong (Fall 2025-present)  
  Supervisor: [Prof. Kam-Fai Wong](https://kfwong.cintec.cuhk.edu.hk/)

## Publications

{% if site.author.googlescholar %}
  <div class="wordwrap">You can also find my articles on <a href="{{site.author.googlescholar}}">my Google Scholar profile</a>.</div>
{% endif %}

{% include base_path %}

<ul>
{% for post in site.publications reversed %}
  {% include archive-single-cv.html %}
{% endfor %}
</ul>
