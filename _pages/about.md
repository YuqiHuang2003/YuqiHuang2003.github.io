---
permalink: /
title: "Bio"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---
I'm Huang Yuqi, now a Research Assistant in [Prof. Kam-Fai Wong](https://kfwong.cintec.cuhk.edu.hk/)'s group, CUHK. I obtained my bachlor's degree under the supervision of [Prof. Ruifeng Xu]( https://homepage.hit.edu.cn/xuruifeng ) in the HLT Group at Harbin Institute of Technology, Shenzhen.

## Publications

{% if site.author.googlescholar %}
  <div class="wordwrap">You can also find my articles on <a href="{{site.author.googlescholar}}">my Google Scholar profile</a>.</div>
{% endif %}

{% include base_path %}

<!-- New style rendering if publication categories are defined -->
{% if site.publication_category %}
  {% for category in site.publication_category  %}
    {% assign title_shown = false %}
    {% for post in site.publications reversed %}
      {% if post.category != category[0] %}
        {% continue %}
      {% endif %}
      {% unless title_shown %}
        <h3>{{ category[1].title }}</h3>
        {% assign title_shown = true %}
      {% endunless %}
      {% include archive-single.html type="list" %}
    {% endfor %}
  {% endfor %}
{% else %}
  {% for post in site.publications reversed %}
    {% include archive-single.html type="list" %}
  {% endfor %}
{% endif %}

