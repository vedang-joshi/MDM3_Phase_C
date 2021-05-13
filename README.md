# Anti Drowning Detection Algorithm

Recently, the frequency of drowning incidences has been increasing. A study revealed children aged 1-4 and adolescents to be most at risk of death in swimming pools and freshwater respectively in the UK. Current technologies like Poseidon detect, using cameras, when a person has been motionless at the bottom of a pool for 10 seconds, and sends an alarm. Other technologies rely on manual deployment of flotation devices which is unsuitable for children who will panic in a crisis. This repository contains a suite of trained convolutional neural networks to differentiate between swimming strokes and drowning (flailing) along with our thresholding algorithm to detect sharp instantaneous changes in heart rate. We obtainean accuracy of 97%.


### Prerequisites

Install the following dependencies to your virtul environment:

```
pip install scikit-learn
pip install pywt
pip install tensorflow
pip install keras
```

### Results
For the CNN models, we get the following results:
![BERT results](confusion_bert.jpg)


## Authors

* **Victoria Adcock** - [GitHub Page](https://github.com/victoriaea97)
* **Vedang Joshi** - [GitHub Page](https://github.com/vedang-joshi)
* **Stefan Johansen** - [GitHub Page](https://github.com/stefanjohansen)
* **Neave Wray** - [GitHub Page](https://github.com/Neavewray1)

## License

This project is licensed under the GNU General Public License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* A special thanks to [Ryan McConville](https://github.com/rymc) for guiding us through this project.
