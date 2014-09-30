#include "Neuron.hpp"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <ctime>
#include <boost/random.hpp>

namespace ESP {

template<typename T>
Population<T>::Population(int size, T& ex) : individuals(size),
											 exemplar(ex),
											 evolvable(size),
											 created(false),
											 maxID(0) {
	numBreed = (unsigned int) individuals.size() / 4;
}

template<typename T>
Population<T>::~Population() {
	destroyIndividuals();
}

/*!
 * Creates the Population
 * Creates the random Population and set the member created to true.
 */
template<typename T>
void Population<T>::create() {
	if (!created) {
		if (evolvable) {
			for (unsigned int i = 0; i < individuals.size(); ++i) {
				individuals[i] = exemplar.clone();
				individuals[i]->create();
			}
			created = true;
		}
	}
	maxID = individuals.back()->getID();
	bestIndividual = individuals.front();
}

/*!
 * Destroys the Neurons in the Population
 * The Neurons are destroyed without deleting the
 * Population. If create is called after calling
 * this method new Neurons will be created and placed
 * in the Population
 */
template<typename T>
void Population<T>::destroyIndividuals() {
	std::cout << "Destroying individuals" << std::endl;
	for (int i = 0; i < individuals.size(); ++i) {
		delete individuals[i];
	}
	created = false;
}

template<typename T>
T* Population<T>::operator[](int i) {
	if ((i >= 0) && (i < individuals.size())) {
		return individuals[i];
	} else {
		std::cerr << "Index out of bounds" << std::endl;
		return 0;
	}
}

/*!
 * Reset fitness and test values of all Neurons
 */
template <typename T>
void Population<T>::evalReset() {
	mapv(typename &T::resetFitness);
}

/*!
 * Select an individual at random
 */
template <typename T>
T* Population<T>::selectRndIndividual(int i) {
	boost::mt19937 rng(time(0));
	boost::uniform_int<> dist(0, (i > 0 && i < individuals.size()) ? i : individuals.size());
	return individuals[dist(rng)];
}

/*!
 * Sort the neurons by fitness in each NeuronIndividuals
 */
template <typename T>
void Population<T>::qsortIndividuals() {
	std::sort(individuals.begin(), individuals.end(), max_fit());
	bestIndividual = individuals.front();
}

/*!
 * Mutate half of the Neurons with Cauchy noise
 */
template <typename T>
void Population<T>::mutate(double mutrate) {
	boost::mt19937 rng(time(0));
	boost::uniform_real<> dist(0.0, 1.0);
	for (int i = numBreed * 2; i < individuals.size(); ++i) {
		if (dist(rng) < mutrate) {
			individuals[i]->mutate();
		}
	}
}

/*! 
 * Used to perform "delta coding" like burst mutation
 * Make each Neuron a perturbation of the Neuron in
 * the best Network that corresponds to that subIndividuals
 */
template <typename T>
void Population<T>::deltify(T* best) {
	for (int i = 0; i < individuals.size(); ++i) {
		individuals[i]->perturb(best);
	}
}

/*!
 * Removes an individual from the Population
 */
template <typename T>
void Population<T>::popIndividual() {
	if (individuals.size() > 0) {
		delete individuals.back();
		individuals.pop_back();
	}
}

/*!
 * Adds an individual to the Population
 * The fitness is added to the Population by pushing the pointer to in
 * onto the back of the Vector individuals
 */
template <typename T>
void Population<T>::pushIndividual(T* n) {
	if (n->getID() > maxID) {
		maxID = n->getID();
	}
	individuals.push_back(n);
}

template <typename T>
double Population<T>::getAverageFitness() {
	double sum = 0;
	for (int i = 0; i < individuals.size(); ++i) {
		sum += individuals[i]->getFitness();
	}
	return sum / individuals.size();
}

}

template <typename T>
ostream& operator<<(ostream& os, ESP::Population<T>& p) {
	for (int i = 0; i < p.getNumIndividuals(); ++i) {
		os << *p.getNumIndividual(i) << std::endl;
	}
	return os;
}