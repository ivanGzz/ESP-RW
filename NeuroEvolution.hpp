#ifndef _NEUROEVOLUTION_HPP_
#define _NEUROEVOLUTION_HPP_

namespace ESP {

class Neuron;
class Network;
class Environment;

/*!
 * Base class for other NeuroEvolution algorithms
 * Class NeuroEvolution is a virtual class that
 * describes the basic structure that derived NeuroEvolutionary
 * algorithms must take, as well as implementing the genetic operators
 * for recombining Networks and Neurons
 */
class NeuroEvolution {
protected:
	int inputDimension; 		///< The number of variables that the nets receive as inputs
	int outputDimension;		///< The number of variables in the action space
	int evaluations;			///< The number of Network evaluations
public:
	bool minimize;				///< Whether or not fitness is maximized or minimized
	Environment& envt;			///< The task environment
	NeuroEvolution(Environment& e);
	int getInDim() { return inputDimension; };
	int getOutDim() { return outputDimension; };
	// Genetic operators
	void crossoverOnePoint(Neuron*, Neuron*, Neuron*, Neuron*);
	void crossoverArithmetic(Neuron*, Neuron*, Neuron*, Neuron*);
	void crossoverEir(Neuron*, Neuron*, Neuron*, Neuron*);
	void crossoverOnePoint(Network*, Network*, Network*, Network*);
	void crossoverArithmetic(Network*, Network*, Network*, Network*);
	void crossoverNPoint(Network*, Network*, Network*, Network*);
	void incEvals() { ++evaluations; };
};

}

#endif