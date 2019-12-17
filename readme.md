# Recommender Systems

The repository deals with recommender systems and various approaches to it. The problem is to predict ratings for items for each user who has not yet rated the item we wish to get the rating for. This repository uses collaborative filtering based approaches to tackle the problem.
CF=> Collaborative Filtering

## Structure of repository

--------
	|
	-Code
		|
		-All the code files
--------
	|
	-ml-latest-small
		|
		-Dataset Files


All the code is present as jupyter notebooks, with my findings in findings.pdf file. The notebooks are well documented. A notebook named UnderstandingDataset is also present, to get some insights into the dataset. Train-Test split used is 70-30.
The **dataset** used is MovieLens Small, with 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users, available on https://grouplens.org/datasets/movielens/ (Last accessed Dec 17, 2019). The files are also available in the folder "ml-latest-small".


### Prerequisites

Python 3 has been used in the code.
Other libraries used include:
jupyter
numpy
pandas
matplotlib
Installation:

```
pip3 install -r requirements.txt
```



## Approaches

This section highlights the different approaches tested.

### User Based and Item Based CF

Prediction function is based on top 'k' similar users or items. It is weighed on Similarity between the users/items. Similarity functions used are pearson coefficient and cosine similarity. 
Example for user based prediction:

```
numerator=0
denominator=0
for k similar users:
	numerator+=similar user's ratings*similar user's similarity
	denominator+=abs(similar user's similarity)
predicted_rating=numerator/denominator
```

Later, mean centring is also used, to account for biases.
After that, Z score prediction is also used.

```
numerator=0
denominator=0
for k similar users:
	numerator+=similar user's ratings*similar user's similarity
	denominator+=abs(similar user's similarity)
predicted_rating=(numerator/denominator)*(standard deviations of ratings by the concerned user) + (mean rating of concerned user)
```

For the long tail issue, inverse frequency is also introduced while calculating similarity.




### Clustering Approaches

Basic clustering approach, as described in [1](https://github.com/abhineet99/cs529/#references) is implemented.

Other approach using data smoothening, as described in [2](https://github.com/abhineet99/cs529/#references) is also implemented.

## Latent Factor Models

In these approaches, we factorize the ratings matrix, and multiply the factors to get the predicted ratings.
Different Approaches for factorization in code:

1) Batch Gradient Descent
2) Stochastic Gradient Descent
3) Gradient Descent with Regularization
4) Accounting for bias terms in GD + Regularization
5) Non Negative Factorization (NMF)
6) Singular Value Decomposition

A detail on on NMF is provided in a reference text in the repository.

## Results


| Method | RMSE Error on test set* |
| --- | --- |
| User Based (Raw) | 1.3844 |
| User Based (Mean Centred) | 0.9088 |
| User Based (Mean Centred+IUF) | 0.9083 |
| Item Based(Mean Centred) | 1.15935 |
| Z-Score User Based | 1.138 |
| KMeans | 1.0093 |
| Clustering Approach [2] | 0.9622 |
|--**Latent Factor Models**--| |	
| BGD | 0.96311 |
| With Regularization | 0.93846 |
| With Regularization+Bias Accounting | 0.93794 |
| SVD | 0.89248 |
| NMF | 0.9468 |

\* Pearson Coefficient has been used as similarity metric in this table wherever required.
## Authors

* **Abhineet Pandey**  - [abhineet99](https://github.com/abhineet99) - [Homepage](https://abhineet99.github.io/)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* [PurpleBooth](https://github.com/PurpleBooth): for the readme template !
* [Dr. Shashi Shekhar Jha](https://sites.google.com/view/shashi-iitrpr/): for making us work on this in Applied AI Course.

## References
1. Charu C. Aggarwal (2016) *Recommender Systems* , Springer . [Available here](https://www.amazon.in/Recommender-Systems-Textbook-Charu-Aggarwal/dp/3319296574/ref=tmm_hrd_swatch_0?_encoding=UTF8&qid=&sr=)

2.  G.-R. Xue, C. Lin, Q. Yang, et al., “Scalable collaborative filtering using cluster-based smoothing,” in *Proceedings of the ACM SIGIR Conference*, pp. 114–121, Salvador, Brazil, 2005.
