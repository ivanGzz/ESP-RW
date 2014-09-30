#include "Network.hpp"
#include "Neuron.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <boost/random.hpp>
#include <ctime>

std::ostream& operator<<(std::ostream& os, ESP::Network &net) {
	os << net.getName() << " " << net.getID() << ": " << std::endl;
	for (int i = 0; i < net.getNumNeurons(); ++i) {
		os << *net.getNeuron(i) << std::endl;
	}
	return os;
}

namespace ESP {

Network::Network(int n, int hid, int out) : activation(hid),
 											hiddenUnits(hid),
 											trials(0),
 											fitness(0.0),
 											parent1(-1),
 											parent2(-1),
 											created(false),
 											numInputs(in),
 											numOutputs(out),
 											bias(0.0) {
 	static int counter = 0;
 	id = ++counter;
}

Network::~Network() {
	deleteNeurons();
}

inline double Network::sigmoid(double x, double slope) {
	return (1.0 / (1.0 + exp(-(slope * x))));
}

/*!
 * Delete the Neurons
 */
void Network::deleteNeurons() {
	if (created) {
		for (int i = 0; i < hiddenUnits.size(); ++i) {
			delete hiddenUnits[i];
		}
		created = false;
	}
}

/*!
 * Delete a network without deleting its Neurons
 */
void Network::releaseNeurons() {
	for (int i = 0; i < hiddenUnits.size(); ++i) {
		hiddenUnits[i] = 0;
	}
	delete this;
}

/*!
 * Set the fitness
 */
void Network::setFitness(double fit) {
	++trials;
	fitness += fit;
}

double Network::getFitness() {
	if (trials) {
		return fitness / (double)trials;
	} else {
		return fitness;
	}
}

void Network::create() {
	for (int i = 0; i < hiddenUnits.size(); ++i) {
		hiddenUnits[i] = new Neuron(geneSize);
		hiddenUnits[i]->create();
	}
	created = true;
}

bool Network::sizeEqual(Network& n) {
	bool equal = true;
	if (!created || !n.created) {
		equal = false;
	} else if (hiddenUnits.size() != n.hiddenUnits.size()) {
		equal = false;
	} else {
		for (int i = 0; i < hiddenUnits.size(); ++i) {
			if (hiddenUnits[i]->getSize() != n.hiddenUnits[i]->getSize()) {
				equal = false;
				break;
			}
		}
	}
	return equal;
}

void Network::operator=(Network& n) {
	if (n.created) {
		std::cerr << "Assigning uncreated Network; Network::operator=" << std::endl;
		abort();
	}
	// Check if Networks are of the same type
	if (type != n.type) {
		std::cerr << "Assigning Networks of type " << n.getName() << " to type " << getName() << "; Network::operator=" << std::endl;
		abort();
	}
	activation = n.activation;
	trials = n.trials;
	fitness = n.fitness;
	parent1 = n.parent1;
	parent2 = n.parent2;
	geneSize = n.geneSize;
	numInputs = n.numInputs;
	numOutputs = n.numOutputs;
	bias = n.bias;
	deleteNeurons();
	hiddenUnits.clear();
	for (int i = 0; i < n.getNumNeurons(); ++i) {
		hiddenUnits.push_back(new Neuron(geneSize));
		*hiddenUnits[i] = *n.hiddenUnits[i];
	}
	created = true;
}

bool Network::operator==(Network& n) {
	bool equal = true;
	if (hiddenUnits.size() != n.hiddenUnits.size()) {
		equal = false;
	} else {
		for (int i = 0; i < hiddenUnits.size(); ++i) {
			if (*hiddenUnits[i] != *n.hiddenUnits[i]) {
				equal = false;
				break;
			}
		}
	}
	return equal;
}

bool Network::operator!=(Network& n) {
	if (*this == n) {
		return false;
	} else {
		return true;
	}
}

void Network::setNeuron(Neuron* n, int position) {
	hiddenUnits[position] = n;
}

inline Neuron* Network::getNeuron(int i) {
	if (i >= 0 && i < (int)hiddenUnits.size()) {
		return hiddenUnits[i];
	} else {
		std::cerr << "Index out of bounds; Network::getNeuron" << std::endl;
	}
}

inline int Network::getParent(int p) {
	if (p == 1) {
		return parent1;
	} else if (p == 2) {
		return parent2;
	} else {
		std::cerr << "Parent must be 1 or 2; Network::getParent" << std::endl;
	}
}

inline Network::setParent(int p, int id) {
	if (p == 1) {
		parent1 = id;
	} else if (p == 2) {
		parent2 = id;
	} else {
		std::cerr << "Parent must be 1 or 2; Network::getParent" << std::endl;
	}
}

void Network::setNetwork(Network* n) {
	parent1 = n->parent1;
	parent2 = n->parent2;
	fitness = n->fitness;
	trials = n->trials;
	for (int i = 0; i < hiddenUnits.size(); ++i) {
		hiddenUnits[i] = n->hiddenUnits[i];
	}
}

void Network::addFitness() {
	for (int i = 0; i < hiddenUnits.size() ++i) {
		hiddenUnits[i]->addFitness(fitness);
	}
}

void Network::resetActivation() {
	for (int i = 0; i < hiddenUnits.size(); ++i) {
		activation[i] = 0.0;
	}
}

/*!
 * Used by complete
 */
void Network::perturb(Network* net) {
	for (int i = 0; i < hiddenUnits.size(); ++i) {
		hiddenUnits[i]->perturb(net->hiddenUnits[i], rndCauchy, 0.01);
	}
}

/*!
 * Same as above but called on self and returns new Network
 */
Network* Network::perturb(double coeff) {
	Network* n = this->clone();
	for (int i = 0; i < hiddenUnits.size(); ++i) {
		n->hiddenUnits[i] = hiddenUnits[i]->perturb(0.05);
	}
	n->created = true;
	return n;
}

void Network::mutate(double mutRate) {
	boost::mt19937 rng(time(0));
	boost::uniform_real<> rdist(0.0, 1.0);
	if (rdist(rng) < mutRate) {
		boost::uniform_int<> idist(0, hiddenUnits.size() - 1);
		hiddenUnits[idist(rng)]->mutate();
	} 
}

void Network::printActivation(FILE* file) {
	for (int i = 0; i < hiddenUnits.size(); ++i) {
		fprintf(file, "%f ", activation[i]);
	}
	fprintf(file, "\n");
}

void Network::addConnection(int locus) {
	for (int i = 0; i < hiddenUnits.size(); ++i) {
		hiddenUnits[i]->addConnection(locus);
	}
}

void Network::removeConnection(int locus) {
	if (locus < geneSize) {
		for (int i = 0; i < hiddenUnits.size(); ++i) {
			hiddenUnits[i]->removeConnection(locus);
		}
	}
}

void Network::saveText(std::string fname) {
	std::string newname = fname + name;
	std::ofstream file(newname.c_str(), std::ofstream::out);
	if (file) {
		file << type << std::endl;
		file << numInputs << std::endl;
		file << hiddenUnits.size() << std::endl;
		file << numOutputs << std::endl;
		for (int i = 0; i < hiddenUnits.size(); ++i) {
			for (int j = 0; j < geneSize; ++j) {
				file << hiddenUnits[i]->getWeight(j) << " ";
			}
			file << std::endl;
		}
	} else {
		std::cerr << "Error - cannot open " << newname << "; Network::saveText" << std::endl;
	}
	file.close();
}

}