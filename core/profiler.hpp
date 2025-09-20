class PROFILE {
public:
    enum TimerType { Chrono, MPI, OMP };

private:
    TimerType timer_type;        // Timing strategy
    double (*getCurrentTime)();  // Function pointer for the current time function

    struct profNode {
        std::string function_name;
        double start_time;
        double total_time;
        std::vector<profNode*> children;

        profNode(const std::string& name)
            : function_name(name), total_time(0.0) {}
    };

    profNode* root;
    profNode* current_node;
    std::stack<profNode*> node_stack;
    int precision = 5;

public:
    // Constructor to initialize with a specific timer
    PROFILE(TimerType type) : timer_type(type) {
        root = new profNode("root");
        current_node = root;

        // Assign the appropriate function pointer directly
        if (timer_type == Chrono) {
            getCurrentTime = []() {
                return std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now().time_since_epoch()
                ).count();
            };
        } else if (timer_type == MPI) {
            getCurrentTime = []() {
              return MPI_Wtime();
            };
        } else if (timer_type == OMP) {
            getCurrentTime = []() {
                return omp_get_wtime();
            };
        }
    }

    ~PROFILE() {
        clearTree(root);
    }

    void reset() {
      clearTree(root);  // Free memory
      root = new profNode("root");  // Reinitialize
      current_node = root;
      while (!node_stack.empty()) node_stack.pop();  // Clear the stack
    }

    void clearTree(profNode* node) {
        for (auto child : node->children) {
            clearTree(child);
        }
        delete node;
    }

    void start(const std::string& function_name) {
        profNode* new_node = nullptr;
        for (auto* child : current_node->children) {
            if (child->function_name == function_name) {
                new_node = child;
                break;
            }
        }
        if (!new_node) {
            new_node = new profNode(function_name);
            current_node->children.push_back(new_node);
        }
        new_node->start_time = getCurrentTime();
        node_stack.push(current_node);
        current_node = new_node;
    }

    void stop() {
        double end_time = getCurrentTime();
        current_node->total_time += end_time - current_node->start_time;
        current_node = node_stack.top();
        node_stack.pop();
    }

    void printNode(profNode* node, int depth = 0, double parent_time = 0) const {
        double percentage = (parent_time == 0) ? 100.0 : (node->total_time / parent_time) * 100.0;
        std::string indent(depth * 4, ' ');

        if (node->function_name != "Hypre_CSR2IJMatrix" && node->function_name != "Chronos_build_dsmat")
        {
          std::cout << indent << "└── " << std::left << std::fixed
                    << std::setw(20) << node->function_name
                    << std::setw(6) << std::setprecision(2) << std::right << percentage << "% "
                    << std::setw(25 - 4 * depth) << std::right << std::fixed << std::setprecision(precision)
                    << node->total_time << "s\n";

          for (auto* child : node->children) {
              printNode(child, depth + 1, node->total_time);
          }
        }
    }

    void report() const {
        std::cout << "\n--------------------------------\n";
        std::cout << "Profile of Execution Times (seconds):\n";
        for (const auto& child : root->children) {
            printNode(child);
        }
    }

  double getTime(const std::string& function_name) const {
      return getTimeRecursive(root, function_name);
  }

  double getTimeRecursive(profNode* node, const std::string& function_name) const {
      if (!node) return 0.0;

      if (node->function_name == function_name) {
          return node->total_time;
      }

      for (auto* child : node->children) {
          double time = getTimeRecursive(child, function_name);
          if (time > 0.0) {
              return time;
          }
      }

      return 0.0; // Return 0 if the function was not found in this subtree
  }

  profNode* getNode(const std::string& function_name) const {
      return getNodeRecursive(root, function_name);
  }

  profNode* getNodeRecursive(profNode* node, const std::string& function_name) const {
    if (!node) return nullptr;

    if (node->function_name == function_name) {
        return node;
    }

    for (auto* child : node->children) {
        profNode* result = getNodeRecursive(child, function_name);
        if (result) {
            return result;
        }
    }

    return nullptr; // Return nullptr if the function was not found
}

  // remove data conversion time 
  void post_process() const {

    double node_chronos_format = getTime("Chronos_build_dsmat");
    double node_hypre_format = getTime("Hypre_CSR2IJMatrix");

    profNode* node_setup_solve = getNode("setup & solve");
    profNode* node_prec = getNode("Prec");
    profNode* node_glo_prec = getNode("Global precond");
    profNode* node_loc_prec = getNode("Local  precond");

    node_setup_solve->total_time -= (node_hypre_format+node_chronos_format);
    node_prec->total_time -= (node_hypre_format+node_chronos_format);

    node_glo_prec->total_time -= node_chronos_format;
    node_loc_prec->total_time -= node_hypre_format;

  }
};

PROFILE profiler(PROFILE::MPI);