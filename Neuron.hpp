#ifndef _NEURON_HPP_
#define _NEURON_HPP_

#include <string>
#include <ostream>
#include <vector>

namespace ESP {

class Neuron {
public:
	bool lesioned;
	Neuron(int);
	virtual Neuron* clone() { return new Neuron( weight.size() );};
	virtual Neuron& operator=(const Neuron&);
	bool operator==(Neuron &);
	bool operator!=(Neuron &);
	virtual void create();
	virtual void addFitness(double);
	virtual void resetFitness();
	virtual void addConnection(int);
	virtual void removeConnection(int);
	void perturb(const Neuron *);
	void perturb(const Neuron *, double (*randFn)(double), double);
	Neuron* perturb(double coeff = 0.3);
	virtual void mutate();
	double getFitness();
	bool checkBounds(int);
	inline unsigned int getSize() { return weight.size(); };
	inline double getWeight(int i) { if( checkBounds(i) ) return weight[i]; else return -1.0; };
	void setWeight(int, double);
	inline int getID() { return id; };
	inline std::string getName() { return name; };
	Neuron* crossoverOnePoint(Neuron &);
	int parent1;
	int parent2;
	bool tag;
protected:
	inline int newID() { Neuron n(0); id = n.getID(); return id; };
	std::vector<double> weight;
	int trials;
	double fitness;
	int id;
	std::string name;
};

}

double rndCauchy(double);

std::ostream& operator<<(std::ostream &, ESP::Neuron &);

#endif