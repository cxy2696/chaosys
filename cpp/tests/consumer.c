/* Minimal installed-package C consumer used by CI. */
#include <chaosys_c.h>

int main(void) {
  chaosys_state state = {1.0, 1.0, 1.0};
  chaosys_params params = chaosys_default_params();
  if (chaosys_rk4(&state, 1000, 0.01, params) != CHAOSYS_OK) return 1;
  return (state.x < 0.0 && state.z > 0.0) ? 0 : 1;
}
