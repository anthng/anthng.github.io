---
layout: post
title:  Virtual Environment Inside Jupyter Notebook
excerpt: ""
categories:
    - Python
tags:
    - python
comments: true
permalink: blogs/venv-inside-nb

---


## 1. Introduction

In a previous post about [Python virtual environment](https://anthng.github.io/blogs/venv-python), I introduced to the virtual environment (venv) and how to create it. Today, I will show you how to set up venv inside your Jupyter notebook or Jupyter Lab. Make sure that you have read [Python virtual environment](https://anthng.github.io/blogs/venv-python) before keeping on this post.

<h2>Contents</h2>

* TOC
{:toc}

## 2. Installation

- The first step, the venv must be installed and activated.

- Next, you install *jupyter* package via *pip* in order to execute notebook.

{% highlight cmd %}
(.env) PS D:\demo> pip install jupyter
{% endhighlight %}

- You add your venv to Jupyter:

{% highlight cmd %}
python -m ipykernel install --user --name=.envNameAny
Installed kernelspec .envNameAny in C:\Users\AnNg\AppData\Roaming\jupyter\kernels\.envnameany
{% endhighlight %}

Where ```--name``` is an argument for the name of venv in the kernel Jupyter. For the above example, ```.envNameAny``` is the name of venv in kernel Jupyter used for this demo (the name in *ipykernel* is not necessarily the same name as the venv is generated via ```python -m venv <envName>```)

- Finally, launch Jupyter with your venv by typing:

{% highlight cmd %}
jupyter-notebook
{% endhighlight %}

## 3. The Results

To clarify the results, I installed **numpy** package before running **jupyter-notebook**. Fig 1 shows UI of the notebook when starting. You choose ***New > .envNameAny*** to create your notebook with venv.

<a href="../images/posts/venv-jup/ui-jup.png" target="_blank">
<img src="../images/posts/venv-jup/ui-jup.png" alt="UI when opening jupyter-notebook" style="max-width:80%;"/>
</a>
<caption style="caption-side:bottom;"><b>Figure 1.</b> UI when opening notebook</caption>
<br>

Let's write our first magic code now.

<a href="../images/posts/venv-jup/final-result.png" target="_blank">
<img src="../images/posts/venv-jup/final-result.png" alt="first magic code" style="max-width:80%;"/>
</a>
<caption style="caption-side:bottom;"><b>Figure 2.</b> Final result</caption>
<br>
