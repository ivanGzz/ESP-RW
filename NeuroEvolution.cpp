#include "NeuroEvolution.hpp"
#include "Environment.hpp"
#include "Neuron.hpp"
#include "Network.hpp"
#include <algorithm>
#include <boost/random.hpp>
#include <ctime>

namespace ESP {

NeuroEvolution::NeuroEvolution(Environment &e) : inputDimension(e.getInputDimention()),
												 outputDimension(e.getOutputDimension()),
												 evaluations(0),
												 envt(e) {
	envt.setNePtr(this);
}

/*!
 * Arithmetic crossover
 */
void NeuroEvolution::crossoverArithmetic(Neuron* parent1, Neuron* parent2, Neuron* child1, Neuron* child2) {
	child1->parent1 = parent1->getID();
	child1->parent2 = parent2->getID();
	child2->parent1 = parent1->getID();
	child2->parent2 = parent2->getID();
	child1->resetFitness();
	child2->resetFitness();
	double a = 0.25, b = 0.75;
	for (int i = 0; i < parent1->getSize(); ++i) {
		child1->setWeight(i, a * parent1->getWeight(i) + (b * parent2->getWeight(i)));
		child2->setWeight(i, a * parent2->getWeight(i) + (b * parent1->getWeight(i)));
	}
}

/*!
 * Another linear combination crossover
 */
void NeuroEvolution::crossoverEir(Neuron* parent1, Neuron* parent2, Neuron* child1, Neuron* child2) {
	child1->parent1 = parent1->getID();
	child1->parent2 = parent2->getID();
	child2->parent1 = parent1->getID();
	child2->parent2 = parent2->getID();
	double d = 0.4;
	double d2 = 2.0 * d + 1;
	boost::mt19937 rng(time(0));
	boost::uniform_real<> dist(0.0, 1.0);
	for (int i = 0; i < parent1->getSize(); ++i) {
		child1->setWeight(i, parent1->getWeight(i) + (d2 * dist(rng) - d) * (parent2->getWeight(i) - parent1.getWeight(i)));
		child2->setWeight(i, parent2->getWeight(i) + (d2 * dist(rng) - d) * (parent1->getWeight(i) - parent2.getWeight(i)));
	}
}

/*!
 * One-point crossover for Neurons
 * Two parent neurons are mated to produce two offspring
 * by exchanging chromosomal sub-strings at a random crossover point
 */
void NeuroEvolution::crossoverOnePoint(Neuron* parent1, Neuron* parent2, Neuron* child1, Neuron* child2) {
	boost::mt19937 rng(time(0));
	boost::uniform_int<> dist1(0, parent1->getSize() - 1);
	boost::uniform_int<> dist2(0, parent2->getSize() - 1);
	int cross1 = dist1(rng);
	if (parent1->getSize() > parent2->getSize()) {
		cross1 = dist2(rng);
	}
	*child1 = *parent2;
	*child2 = *parent1;
	child1->parent1 = parent1->getID();
	child1->parent2 = parent2->getID();
	child2->parent1 = parent1->getID();
	child2->parent2 = parent2->getID();
	child1->resetFitness();
	child2->resetFitness();
	double tmp;
	for (int i = 0; i < cross1; ++i) {
		tmp = child2->getWeight(i);
		child2->setWeight(i, child1->getWeight(i));
		child1->setWeight(i, tmp);
	}
}

/*!
 * One-point crossover for Networks
 * Two parent Networks are mated to produce two offspring
 * by exchanging chromosomal sub-strings at a random crossover point
 */
void NeuroEvolution::crossoverOnePoint(Network* parent1, Network* parent2, Network* child1, Network* child2) {
	boost::mt19937 rng(time(0));
	boost::uniform_int<> dist1(0, parent1->getNumNeurons() - 1);
	boost::uniform_int<> dist2(0, parent2->getNumNeurons() - 1);
	int crossNeuron = dist1(rng);
	if (parent1->getNumNeurons() > parent2->getNumNeurons()) {
		crossNeuron = dist2(rng);
	}
	child1->resetFitness();
	child2->resetFitness();
	crossoverOnePoint(parent1->getNeuron(crossNeuron), 
					  parent2->getNeuron(crossNeuron),
					  child1->getNeuron(crossNeuron),
					  child2->getNeuron(crossNeuron));
	child1->setParent(1, parent1->getID());
	child1->setParent(2, parent2->getID());
	child2->setParent(1, parent1->getID());
	child2->setParent(2, parent2->getID());
}

/*!
 * N-point crossover for Networks
 * Two parent Networks are mated to produce two offspring
 * by exchanging chromosomal sub-strings at N random crossover points
 */
void NeuroEvolution::crossoverNPoint(Network* parent1, Network* parent2, Network* child1, Network* child2) {
	*child1 = *parent1;
	*child2 = *parent2;
	child1->resetFitness();
	child2->resetFitness();
	for (int i = 0; i < parent1->getNumNeurons(); ++i) {
		crossoverOnePoint(parent1->getNeuron(i),
			   			  parent2->getNeuron(i),
			   			  child1->getNeuron(i),
			   			  child2->getNeuron(i));
	}
	child1->setParent(1, parent1->getID());
	child1->setParent(2, parent2->getID());
	child2->setParent(1, parent1->getID());
	child2->setParent(2, parent2->getID());
}

void NeuroEvolution::crossoverArithmetic(Network* parent1, Network* parent2, Network* child1, Network* child2) {
	child1->resetFitness();
	child2->resetFitness();
	for (int i = 0; i < parent1->getNumNeurons(); ++i) {
		crossoverArithmetic(parent1->getNeuron(i),
						    parent2->getNeuron(i),
						    child1->getNeuron(i),
						    child2->getNeuron(i));
	}
	child1->setParent(1, parent1->getID());
	child1->setParent(2, parent2->getID());
	child2->setParent(1, parent1->getID());
	child2->setParent(2, parent2->getID());
}

}