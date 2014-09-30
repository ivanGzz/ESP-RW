#include "Environment.hpp"
#include "NeuroEvolution.hpp"
#include "Network.hpp"
#include <iostream>

namespace ESP {

/*!
 * Evaluate a Network in the Environment
 * Takes a Network and evaluates it on a task.  First
 * checks to see if it is connected to a NeuroEvolution
 * algorithm, then increments the algorithm's
 * evaluate the network, assigns it a fitness and return the
 * fitness of the network.  This function is a friend of
 * Network and is the only function outside of the Network
 * class that can set the value of a Network.  This ensures
 * that Networks are only assigned fitness when they are
 * evaluated.
 */
double Environment::evaluateNetwork(Network* net) {
	if (netPtr) {
		nePtr->incEvals();
	}
	net->resetActivation();
	double fit = evalNet(net);
	if (nePtr && nePtr->minimize) {
		net->setFitness(1.0 / (fit + 1.0));
	} else {
		net->setFitness(fit);
	}
	return fit;
}

}