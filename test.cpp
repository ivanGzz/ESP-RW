#include "Neuron.hpp"
#include <iostream>

int main() {
	ESP::Neuron n(10);
	n.create();
	std::cout << n << std::endl;
	return 0;
}