#ifndef _POPULATION_HPP_
#define _POPULATION_HPP_

#include "Network.hpp"
#include <typeinfo>
#include <cstdio>
#include <vector>
#include <functional>

namespace ESP {

class Neuron;
class Network;

template <typename T>
class Population {
public:
	Population(int, T&);
	~Population();
	void create();
	struct max_fit : public std::binary_function<T*, T*, bool> {
		bool operator()(T* x, T* y) { return x->getFitness() > y->getFitness(); }
	};
	void destroyIndividuals();
	void map(double (*map_fn)(T*)) {
		for (typename std::vector<T*>::iterator i = individuals.begin(); i != individuals.end(); ++i) {
			map_fn(i);
		}
	}
	void mapv(void (typename T::*map_fn)()) {
		for (typename std::vector<T*>::iterator i = individuals.begin(); i != individuals.end(); ++i) {
			(*i->*map_fn)();
		}
	}
	T* operator[](int i);
	void evalReset();
	T* selectRndIndividual(int i = -1);
	void average();
	void qsortIndividuals();
	void mutate(double);
	void deltify(T*);
	void popIndividual();
	void pushIndividual();
	double getAverageFitness(T*);
	inline unsigned int getNumIndividuals() { return individuals.size(); };
	inline T* getIndividual(int i) { return individuals[i]; };
	inline unsigned int getNumBreed() { return numBreed; };
	inline void setNumBreed(int n) { if (n > 0) numBreed = n; };
	inline int getMaxID() { return maxID; }
	std::vector<T*> individuals;
protected:
	T& exemplar;
	bool evolvable;
	T* bestIndividual;
	bool created;
	unsigned int numBreed;
	int maxID;
};

typedef Population<Neuron> NeuronPop;
typedef Population<Network> NetworkPop;

}

#include "Population.cpp"
#endif