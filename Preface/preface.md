# Preface

```{epigraph}
> "We demand rigidly defined areas of doubt and uncertainty!"

-- Douglas Adams, *The Hitchhiker's Guide to the Galaxy*
```

These are the lecture notes for an advanced-level course "Learning from data". 

## Course aim

The course introduces a variety of central algorithms and methods essential for performing scientific data analysis using statistical inference and machine learning. Much emphasis is put on practical applications of Bayesian inference in the natural and engineering sciences, i.e. the ability to quantify the strength of inductive inference from facts (such as experimental data) to propositions such as scientific hypotheses and models.

As a teacher-led university course, the learning and examination is project-based. Students will be exposed to contemporary research problems as the specific aim of course projects is to reproduce state-of-the-art scientific results from published papers. Students will use the Python programming language, with relevant open-source libraries, and will learn to develop and structure both workflows and computer codes for scientific modeling and data analysis projects.
<!-- !split -->
## About these lecture notes

These lecture notes have been authored by [Christian Forssén](https://www.chalmers.se/en/persons/f2bcf/) with input from several colleagues (see below), and are released under a [Creative Commons BY-NC license](https://creativecommons.org/licenses/by-nc/4.0/). The book format is powered by [Jupyter Book](https://jupyterbook.org/).

```{admonition} Accompanying git repository
  All source files, including the jupyter notebooks used for exercises and demonstrations, can be found in the accompanying git repository (see the gitlab icon ![gitlab download icon](./figs/gitlab-1-32.png) at the top-middle-right and click on Source repository). 
  
  Note that the repository contains git submodules. It is recommended to use the following git command to clone the repository and automatically initialize and update each submodule (including nested submodules).
  
  `git clone --recurse-submodules https://gitlab.com/cforssen/tif285-book.git`
  
  Here we have used the HTTPS address of the gitlab repository (this URL is found via the Source repository by clicking the "Code" button in the top menu).
  ```

```{admonition} Open an issue
  The author appreciates feedback if you find typos, inconsistent notation, or have a suggestion (on physics, statistics, coding, or formatting). From any page, click on the gitlab icon ![gitlab download icon](./figs/gitlab-1-32.png) at the top-right and go to the Source repository. Click on Issues in the left hand margin and select New issue in the top. You might need to create a gitlab account before being allowed to proceed. This will take you to the Issues section of the gitlab repository for the book. Enter a descriptive title and then write your feedback in the text box.
  ```
  
## Brief guide to online Jupyter Book features

* A clickable high-level table of contents (TOC) is available in the panel at the left of each page. (You can close this panel with the left arrow at the top-left-middle of the page or open it with the contents icon at the upper left.) 

```{admonition} Icons and menus
  The Jupyter book has several useful features:
- For each section that has subsections, a clickable table of contents appears in the rightmost panel.
- The icons at the top-right can be used to take you to the source repository for the book; download the source code for the page (in different formats); view the page in full-screen mode; switch between light and dark mode; or search the book.
```

# Acknowledgements

These notes have evolved over several years with the experience from teaching different courses that included different subsets of the material. The origin was an intensive three-week summer school course taught at the [University of York](https://www.york.ac.uk/) in 2019 by Christian Forssén, Dick Furnstahl, and Daniel Phillips as part of the [TALENT](https://fribtheoryalliance.org/TALENT/) initiative. New material has subsequently been added for course developments at Chalmers and for lecture series at other universities. Significant contributions from Andreas Ekström are specifically acknowledged. In general, both the original notes and subsequent revisions have been informed by interactions with many colleagues. I am particularly grateful to:

* Andreas Ekström, Chalmers University of Technology
* Richard Furnstahl, Ohio State University
* Morten Hjorth-Jensen, Oslo University and Michigan State University
* Daniel Phillips, Ohio University
* Ian Vernon, Durham University
* Sarah Wesolowski, University of Pennsylvania

Many of the advanced Bayesian methods that might be included in these notes have been used in scientific studies with different collaborators. In particular, several postdocs, PhD students and master students have had leading roles in the development and application of these methods to address various scientific questions. In alphabetical order I would like to highlight the contributions of: Boris Carlsson, Tor Djärv, Weiguang Jiang, Eleanor May, Isak Svensson, and Oliver Thim.

The full list of people that have contributed with ideas, discussions, or by generously sharing their knowledge is very long. Rather than inadvertently omitting someone, I simply say thank you to all. More generally, I am truly thankful for being part of an academic environment in which ideas and efforts are shared rather than kept isolated.

The last statement extends to the open-source communities through which great computing tools are made publicly available. In this course we take great advantage of open-source python libraries.  

The development of this course would not have been possible without the knowledge gained through the study of several excellent textbooks, most of which are listed as recommended course literature. Here is a short list of those references that I have found particularly useful as a physicist learning Bayesian statistics and the fundamentals of machine learning:

1. Phil Gregory, *"Bayesian Logical Data Analysis for the Physical Sciences"*, Cambridge University Press (2005) {cite}`Gregory2005`.
2. E. T. Jaynes, *"Probability Theory: The Logic of Science"*, Cambridge University Press (2003) {cite}`Jaynes2003`.
3. David J.C. MacKay, *"Information Theory, Inference, and Learning Algorithms"*, Cambridge University Press (2005) {cite}`Mackay2003`.
4. D.S. Sivia, *"Data Analysis : A Bayesian Tutorial"*, Oxford University Press (2006) {cite}`Sivia2006`.

