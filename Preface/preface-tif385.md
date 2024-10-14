# Preface

```{epigraph}
> "We demand rigidly defined areas of doubt and uncertainty!"

-- Douglas Adams, *The Hitchhiker's Guide to the Galaxy*
```

These are the lecture notes for the bachelor-level course "Bayesian Inference and Machine Learning" (TIF385) that is taught at Chalmers University of Technology. The accompanying jupyter notebooks can be found in [this git repository](https://gitlab.com/cforssen/tif385-book).

```{admonition} Accompanying git repository
  All source files, including the jupyter notebooks used for exercises and demonstrations, can be found in the accompanying git repository (see the github icon ![github download icon](./figs/GitHub-Mark-32px.png) at the top-middle-right. Depending on the platform you might have to click on Source repository).
  ```

## Course aim
Building on a first course in mathematical statistics this course aims to provide knowledge on Bayesian inference, both in general and in the context of modeling physical systems, and a deeper understanding of modern machine learning. In combination, this knowledge should provide a firm basis for practical applications of statistical models in science.

The course is partly project-based, and the students will learn to develop and structure computer codes for statistical inference and machine learning with scientific applications. Specifically, students will perform physics projects using the Python programming language with relevant open-source libraries.

<!-- !split -->
## About this book

These lecture notes have been authored by [Christian Forssén](https://www.chalmers.se/en/persons/f2bcf/) and are released under a [Creative Commons BY-NC license](https://creativecommons.org/licenses/by-nc/4.0/). The book format is powered by [Jupyter Book](https://jupyterbook.org/).
  
A clickable high-level table of contents (TOC) is available in the panel at the left of each page. (You can open and close this panel with the contents icon at the upper left.) 

```{admonition} Jupyter book features
  The Jupyter book has several useful features:
- For each section that has subsections, a clickable table of contents appears in the rightmost panel.
- The "Search this book..." box is a useful tool. Depending on the platform it is either found just below the title in the TOC panel, or via the magnifying glass in the top right.
- The icons at the top-right can be used to take you to the source repository for the book; download the source code for the page (in different formats); view the page in full-screen mode.
- Depending on the platform, you migt also get a direct link to the Issues section of the git repository for the book. This is a good place to post comments and suggestions.
```

<!-- ======= Acknowledgements ======= -->
## Acknowledgements

These notes have evolved over several years with the experience from teaching courses at various levels that included different subsets of the material. The origin was an intensive three-week summer school course for young researchers taught at the [University of York](https://www.york.ac.uk/) in 2019 by Christian Forssén, Dick Furnstahl, and Daniel Phillips as part of the [TALENT](https://fribtheoryalliance.org/TALENT/) initiative. New material has subsequently been added for course developments at Chalmers and for lecture series at other universities. Significant contributions from Andreas Ekström are specifically acknowledged. In general, both the original notes and subsequent revisions have been informed by interactions with many colleagues. I am particularly grateful to:

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

