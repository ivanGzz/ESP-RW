#ifndef _ENVIRONMENT_HPP_
#define _ENVIRONMENT_HPP_

#include <string>
#include <vector>
#include <cstdio>

namespace ESP {

class Network;
class NeuroEvolution;

/*!
 * Virtual class that describes the interface
 * for all task environments used in NeuroEvolution objects
 */
class Environment {
public:
	Environment() : nePtr(0), tolerance(0), incremental(false) {};
	virtual ~Environment() {};
	double evaluateNetwork(Network*);
	virtual void nextTask() {};
	virtual void simplifyTask() {};
	virtual double evalNetDump(Network *net, FILE*) { return 0.0; };
	virtual double generalizationTest(Network*) { return 0.0; };
	void setNetPtr(NeuroEvolution* e) { nePtr = e; };
	inline int getInputDimension() { return inputDimension; };
	inline int getOutputDimension() { return outputDimension; };
	inline double getTolerance() { return tolerance; };
	inline bool getIncremental() { return incremental; };
	inline std::string getName() { return name; };
protected:
	NeuroEvolution* nePtr; 		///< Pointer to the NeuroEvolution algorithm
	std::string name;
	double tolerance;
	bool incremental;
	int inputDimension;			///< Dimension of input space
	int outputDimension;		///< Dimension of output space
	virtual void setupInput(std::vector<double>& input) = 0;
	virtual double evalNet(Network* net) = 0;
};

}

#endif