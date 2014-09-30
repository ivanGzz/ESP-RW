#ifndef _NETWORK_HPP_
#define _NETWORK_HPP_

#include "Environment.hpp"
#include <vector>
#include <string>
#include <ostream>

namespace ESP {

class Neuron;

/*!
 * Neural network base class
 * Virtual class for neural networks consisting of a vector of Neurons that are
 * connected through the implementation is an activation function in the derived
 * classes 
 */
class Network {
protected:
	std::vector<double> activation;
	std::vector<Neuron*> hiddenUnits;
	int trials;
	double fitness;
	int id;
	int parent1;
	int parent2;
	int geneSize;
	std::string name;
	int type;
	void setFitness(double);
	void addConnection(int);
	void removeConnection(int);
	double sigmoid(double x, double slope = 1.0);
public:
	bool created;
	int numInputs;
	int numOutputs;
	double bias;
	Network(int, int, int);
	Network(const Network &n) {};
	virtual ~Network();
	virtual Network* newNetwork(int, int, int) = 0;
	virtual Network* clone() = 0;
	virtual void growNeuron(Neuron*) = 0;
	virtual void shrinkNeuron(Neuron*, int) = 0;
	virtual void addNeuron() = 0;
	virtual void removeNeuron(int) = 0;
	virtual void activate(std::vector<double>&, std::vector<double>&) = 0;
	inline virtual int getMinUnits() { return 1; };
	void releaseNeurons();
	void deleteNeurons();
	void operator=(Network& n);
	void operator==(Network& n);
	void operator!=(Network& n);
	void create();
	void resetActivation();
	void setNeuron(Neuron*, int);
	void setNetwork(Network*);
	void addFitness();
	void perturb(Network*);
	Network* perturb(double coeff = 0.3);
	void mutate(double);
	void printActivation(FILE*);
	void saveText(std::string);
	void resetFitness() { fitness = 0.0; trials = 0; };
	friend double Environment::evaluateNetwork(Network*);
	inline int getNumNeurons() { return (int)hiddenUnits.size(); };
	double getFitness();
	Neuron* getNeuron(int);
	inline int getID() { return id; };
	int getParent(int);
	inline int getGeneSize() { return geneSize; };
	void setParent(int, int);
	std::string getName() { return name; };
	int getType() { return type; };
private:
	bool sizeEqual(Network& n);
};

}

std::ostream& operator<<(std::ostream &, ESP::Network &);

#endif