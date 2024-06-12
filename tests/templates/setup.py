from setuptools import setup

{% for line in setup_py_lines %}
{{ line }}
{% endfor %}
