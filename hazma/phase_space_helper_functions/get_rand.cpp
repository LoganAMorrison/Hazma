#include <random>

using namespace std;

double get_rand() {
  std::mt19937 mt_rand(time(0));
  return mt_rand();
}
