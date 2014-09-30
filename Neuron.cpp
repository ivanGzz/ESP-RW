#include "Neuron.hpp"
#include <cmath>
#include <ctime>
#include <boost/random.hpp>

#define PI 3.1415926535897931 

std::ostream& operator<<(std::ostream& os, ESP::Neuron& n) {
	os.precision(20);
	os << n.getName() << " " << n.getID() << ": " << std::endl;
	for (int i = 0; i < n.getSize(); ++i) {
		os << n.getWeight(i) << " ";
	}
	os << std::endl;
	return os;
}

/*!
 * Generates a random number from a Cauchy distribution centered in zero
 */
double rndCauchy(double wtrange) {
	double u = 0.5, Cauchy_cut = 10.0;
	boost::mt19937 rng(time(0));
	boost::uniform_real<> dist(0.0, 1.0);
	while (u == 0.5) {
		u = dist(rng);
	}
	u = wtrange * tan(u * PI);
	if (fabs(u) > Cauchy_cut) {
		return rndCauchy(wtrange);
	} else {
		return u;
	}
}

namespace ESP {

Neuron::Neuron(int size) : lesioned(false),
						   parent1(-1),
						   parent2(-1),
						   tag(false),
				    	   weight(size),
						   trials(0),
						   fitness(0.0) {
	name = "basic neuron";
	static int counter = 0;
	id = ++counter;
}

inline bool Neuron::checkBounds(int i) {
	if (i >= 0 && i < (int)weight.size()) {
		return true;
	} else {
		std::cerr << "Error: weight index out of bounds" << std::endl;
		abort();
	}
}

/*!
 * Assign fitness to a Neuron
 */
void Neuron::addFitness(double fit) {
	fitness += fit;
	++trials;
}

/*!
 * Set a Neuron's fitness to zero
 */
void Neuron::resetFitness() {
	fitness = 0.0;
	trials = 0;
}

inline double Neuron::getFitness() {
	if (trials) {
		return fitness / (double)trials;
	} else {
		return fitness;
	}
}

inline void Neuron::setWeight(int i, double w) {
	if (checkBounds(i)) {
		weight[i] = w;
		newID();
	}
}

/*!
 * Perturb the weights of a Neuron
 * Used to search in a neighbourhood around some Neuron (best)
 */
void Neuron::perturb(const Neuron* n, double (*randFn)(double), double coeff) {
	for (int i = 0; i < weight.size(); ++i) {
		setWeight(i, n->weight[i] + (randFn)(coeff));
	}
	resetFitness();
}

void Neuron::perturb(const Neuron* n) {
	perturb(n, rndCauchy, 0.3);
}

/*!
 * Same as above but called on self and returns new Neuron
 */
Neuron* Neuron::perturb(double coeff) {
	Neuron* n = new Neuron(weight.size());
	for (int i = 0; i < weight.size(); ++i) {
		n->setWeight(i, weight[i] + rndCauchy(coeff));
	}
	return n;
}

/*!
 * Neuron assignment operator
 */
Neuron& Neuron::operator=(const Neuron &n) {
	id = n.id;
	parent1 = n.parent1;
	parent2 = n.parent2;
	fitness = n.fitness;
	trials = n.trials;
	weight = n.weight;
	return *this;
}

/*!
 * Check if two Neurons are equal
 * Two Neurons are considered equal if they have equal weight vectors
 */
bool Neuron::operator==(Neuron& n) {
 	if (weight == n.weight) {
 		return true;
 	} else {
 		return false;
 	}
}

/*!
 * Check if two Neurons are not equal
 * Two Neurons are considered not equal if they have weight vectors that are not equal
 */
bool Neuron::operator!=(Neuron& n) {
	if (*this == n) {
		return false;
	} else {
		return true;
	}
}

/*!
 * Add a connection to a Neuron
 */
inline void Neuron::addConnection(int n) {
	weight.insert(weight.begin() + n, 1.0);
}

inline void Neuron::removeConnection(int n) {
	weight.erase(weight.begin() + n);
}

/*!
 * Creates a new set of random weights
 */
void Neuron::create() {
	boost::mt19937 rng(time(0));
	boost::uniform_real<> dist(0.0, 12.0);
	for (int i = 0; i < weight.size(); ++i) {
		weight[i] = dist(rng) - 6.0; //change to Boost Random
	}
}

void Neuron::mutate() {
	boost::mt19937 rng(time(0));
	boost::uniform_int<> dist(0, weight.size() - 1);
	weight[dist(rng)] += rndCauchy(0.3);
}

Neuron* Neuron::crossoverOnePoint(Neuron& n) {
	int s1 = weight.size();
	std::vector<double>::iterator i, j, k;
	boost::mt19937 rng(time(0));
	boost::uniform_int<> dist(1, s1 - 1); // int cross1 = lrand48() % (s1 - 1) + 1
	int cross1 = dist(rng);
	Neuron* child = new Neuron(s1);
	k = n.weight.begin() + cross1;
	j = weight.begin();
	for (i = child->weight.begin(); i != child->weight.begin() + cross1; ++i, ++j) {
		*i = *j;
	}
	for (; i != child->weight.end(); ++i, ++k) {
		*i = *k;
	}
	return child;
}

}