[![LinkedIn][linkedin-shield]][linkedin-url]
[![MIT License][license-shield]][license-url]




<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/eth_logo_kurz_pos.png" alt="Logo" width="150" height="80">
  </a>

  <h3 align="center">A minimal machine learning mass balance model:</h3>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Glacier retreat presents significant environmental and social challenges. Understanding the local impacts of climatic drivers on glacier evolution is crucial, with mass balance being a central concept. This study introduces miniML-MB, a new minimal machine learning model designed to estimate annual point mass balance (PMB) for very small datasets. Based on an XGBoost architecture, miniML-MB is applied to model PMB at individual sites in the Swiss Alps, emphasizing the need for an appropriate training framework and dimensionality reduction techniques. A substantial added value of miniML-MB is its data-driven identification of key climatic drivers of local mass balance. The best PMB prediction performance was achieved with two predictors: mean air temperature (May-August) and total precipitation (October-February). miniML-MB models PMB accurately from 1961 to 2021, with a mean absolute error (MAE) of 0.417 m w.e. across all sites. Notably, miniML-MB demonstrates similar and, in most cases, superior predictive capabilities compared to a simple positive degree-day (PDD) model (MAE of 0.541 m w.e.). Compared to the PDD model, miniML-MB is less effective at reproducing extreme mass balance values (e.g., 2022) that fall outside its training range. As such, miniML-MB shows promise as a gap-filling tool for sites with incomplete PMB measurements, as long as the missing year's climate conditions are within the training range. This study underscores potential ways for further refinement and broader applications of data-driven approaches in glaciology.



### Built With

* [![PyTorch][pytorch.py]][pytorch-url]
* [![Python][python.py]][python-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

All notebooks to run the model and explore its pre-processing are available [here](https://github.com/marvande/miniML-MB/tree/main/src). 



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.



<!-- CONTACT -->
## Contact

Marijn van der Meer - vmarijn@ethz.ch

Project Link: [https://github.com/marvande](https://github.com/marvande)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/marvande/master-thesis/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/marvande/master-thesis/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/marvande/master-thesis/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/marvande/master-thesis/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/marijn-van-der-meer/
[product-screenshot]: images/screenshot.png
[pytorch-url]: [https://pytorch.org/](https://scikit-learn.org/stable/)
[pytorch.py]: https://img.shields.io/badge/scikit-learn?style=for-the-badge&logo=scikit-learn&logoColor=white
[python-url]: https://www.python.org/
[python.py]: https://img.shields.io/badge/Python-563D7C?style=for-the-badge&logo=python&logoColor=white
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
