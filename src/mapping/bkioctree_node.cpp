#include <algorithm>
#include <assert.h>
#include <iostream>

#include "mapping/bkioctree_node.h"

namespace semantic_bki {

    /// Default static values
    int Semantics::num_class = 2;
    float Semantics::sf2 = 1.0f;
    float Semantics::ell = 1.0f;
    float Semantics::prior = 0.5f;
    float Semantics::var_thresh = 1000.0f;
    float Semantics::free_thresh = 0.3f;
    float Semantics::occupied_thresh = 0.7f;

    
    void Semantics::get_probs(std::vector<float>& probs) const {
      assert (probs.size() == num_class);
      float sum = 0;
      for (auto m : ms)
        sum += m;
      for (int i = 0; i < num_class; ++i)
        probs[i] = ms[i] / sum;
    }

    void Semantics::get_occupied_probs(std::vector<float>& probs) const {
      assert (probs.size() == num_class - 1);
      float sum = 0;
      for (auto m : ms)
        sum += m;
      for (int i = 1; i < num_class; ++i)
        probs[i - 1] = ms[i] / sum;
    }

    void Semantics::get_vars(std::vector<float>& vars) const {
      assert (vars.size() == num_class);
      float sum = 0;
      for (auto m : ms)
        sum += m;
      for (int i = 0; i < num_class; ++i)
        vars[i] = ((ms[i] / sum) - (ms[i] / sum) * (ms[i] / sum)) / (sum + 1);
    }

    void Semantics::get_features(std::vector<float>& features) const {
      float sum = 0;
      for (int i = 1; i < ms.size(); ++i)  // exclude free points
        sum += ms[i];
      for (int i = 0; i < 5; ++i)
        features[i] = this->fs[i] / sum;
    }

    void Semantics::update(std::vector<float>& ybars, std::vector<float>& fbars) {
      assert(ybars.size() == num_class);
      classified = true;
      for (int i = 0; i < num_class; ++i)
        ms[i] += ybars[i];

      std::vector<float> probs(num_class);
      get_probs(probs);

      // update features
      for (int i = 0; i < 5; ++i) {
        fs[i] += fbars[i];
      }

      semantics = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));

      if (semantics == 0)
        state = State::FREE;
      else {
        state = State::OCCUPIED;
        float p = 1 - probs[0];
        state = p > Semantics::occupied_thresh ? State::OCCUPIED : (p < Semantics::free_thresh ? State::FREE : State::UNKNOWN);
      }
      //else
      //  state = State::OCCUPIED;
    }
}
