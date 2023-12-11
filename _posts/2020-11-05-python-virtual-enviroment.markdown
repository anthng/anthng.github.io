---
layout: post
title:  Python Virtual Environment
excerpt: "It's too bad every package is installed in the default system of Python."
categories:
    - Python
tags:
    - python
comments: true
permalink: blogs/venv-python

---

## 1. Introduction

A Python virtual environment *(venv)* supports for managing your dependencies without affecting with default environment. This *venv* is based in your own site directories. The *venv* is a copy of an existing version Python. It is really useful when you develop a project. Helping you to manage your own site-packages, avoiding the changes which might cause nightmares, between the current version of the library (working on your project) and the latest version of the library in the future.

It's too bad every package is installed in the default system of Python. The venv is part of the most essential work most Python developers use.

<h2>Contents</h2>

* TOC
{:toc}

## 2. Requirements

- An installation of Python

- Opening the terminal and changing direction to your project.
    - **Windows:** Open the *cmd* or *powershell*
    - **MacOS/Linux:** Open the terminal

## 3. Creating Virutal Environment

There are many ways to create a virtual environment, which include using **virtualenv** or **venv** or **conda**. If you're using Python 2, you have to install the *virtualenv* with *pip* or use *conda*. In this article, I just mention to create venv via **virtualenv** and **venv**.

### 3.1 Virtualenv package

For Python 2, you have to install virtualenv to create vitural environment. Python 3 can also use this way. 

In the terminal of Windows (I use powershell):
{% highlight cmd %}
PS D:\my-project> pip install virtualenv
{% endhighlight %}
In the terminal of MacOS/ Linux:

For Python 3:

{% highlight bash %}
anthng@anthng:~/my-project$ pip3 install virtualenv
{% endhighlight %}

For Python 2:

{% highlight bash %}
anthng@anthng:~/my-project$ pip install virtualenv
{% endhighlight %}

Next, type the following:
{% highlight bash %}
virtualenv .envName
{% endhighlight %}
where ```.envName``` is the name of your virtual environment (you can change it).

### 3.2 Venv module

Since version 3.3, Python 3 has come with a built-in ```venv``` module. To use the module, in your terminal:

```bash
python3 -m venv .envName
```

where ```.envName``` is the name of your virtual environment (you can change it).

Type 'python' instead of 'python3' if you use Windows.

### 3.3 Active the virtual environment

Activate the environment by running the following command:

MacOS/Linux:

{% highlight bash %}
anthng@anthng:~/my-project$ source .envName/bin/activate
(.envName)anthng@anthng:~/my-project$ 
{% endhighlight %}

Windows:

{% highlight cmd %}
PS D:\my-project> .envName/Scripts/activate
(.envName) PS D:\my-project>
{% endhighlight %}

The name of your venv will appear on the left side of your terminal when it is active (```(.envName)```in this case)

## 4. Deactive the virtual environment

To deactivate the venv, type ```deactivate```

```bash
deactivate
```

## 5. Other Notes

In order to automatically save all packages to the text file that have been installed in the venv.

For Python 2 or Windows use below:

```bash
pip freeze > requirements.txt
```

For Python 3:
```bash
pip3 freeze > requirements.txt
```

In order to install packages from ```requirements.txt```:

```bash
pip install -r requirements.txt
```

You should also add the venv to ```.gitignore``` file to exclude this folder if you use Git.

## 6. Conclusion

That’s all being needed to start a project with Python Virtual Environment. Lastly, I highly recommend creating a venv before starting any project, which will help you become more professional.