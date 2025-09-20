
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept> 
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
namespace IO {

  void print_ptree(const boost::property_tree::ptree& pt,int indent = 0) {
    // Iterate over all the nodes in the property tree
    for (const auto& node : pt) {
      // Print the current key with indentation
      std::cout << std::string(indent,' ') << node.first << ": ";

      // If the node has children, it's an inner node, so print recursively
      if (node.second.empty()) {
        // Print the value if it's a leaf node
        std::cout << node.second.get_value<std::string>() << std::endl;
      } else {
        std::cout << std::endl;
        print_ptree(node.second,indent + 4);  // Recursively print the children
      }
    }
  }

  std::string remove_json_comments(const std::string& input) {
    std::istringstream input_stream(input);
    std::ostringstream output_stream;
    std::string line;
    bool in_multiline_comment = false;

    while (std::getline(input_stream,line)) {
      std::string processed_line;
      size_t pos = 0;

      // Handle multiline comments  
      while (pos < line.length()) {
        if (in_multiline_comment) {
          // Look for end of multiline comment  
          size_t end_comment = line.find("*/",pos);
          if (end_comment != std::string::npos) {
            in_multiline_comment = false;
            pos = end_comment + 2;
          } else {
            break;  // Entire line is in a multiline comment  
          }
        }

        // Check for single-line comment  
        if (!in_multiline_comment) {
          size_t single_line_comment = line.find("//",pos);
          if (single_line_comment != std::string::npos) {
            processed_line += line.substr(pos,single_line_comment - pos);
            break;
          }

          // Check for multiline comment start  
          size_t multiline_start = line.find("/*",pos);
          if (multiline_start != std::string::npos) {
            processed_line += line.substr(pos,multiline_start - pos);
            size_t multiline_end = line.find("*/",multiline_start);

            if (multiline_end != std::string::npos) {
              pos = multiline_end + 2;
            } else {
              in_multiline_comment = true;
              break;
            }
          } else {
            processed_line += line.substr(pos);
            break;
          }
        }
      }

      // Only add non-empty lines  
      if (!processed_line.empty() && processed_line.find_first_not_of(" \t\r\n") != std::string::npos) {
        output_stream << processed_line << '\n';
      }
    }

    return output_stream.str();
  }

  boost::property_tree::ptree read_ptree(int argc,char* argv[]) {
    boost::property_tree::ptree param;

    if (argc < 2) {
      std::cerr << "Usage: " << argv[0] << " param.json" << std::endl;
      return param;
    }

    try {
      // Read the entire file  
      std::ifstream file(argv[1]);
      if (!file) {
        throw std::runtime_error("Could not open file");
      }

      std::string file_contents((std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());

      // Remove comments  
      std::string cleaned_json = remove_json_comments(file_contents);

      // Parse the cleaned JSON  
      std::istringstream json_stream(cleaned_json);
      boost::property_tree::read_json(json_stream,param);
    }
    catch (const boost::property_tree::json_parser_error& e) {
      std::cerr << "Error reading JSON file: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << std::endl;
    }

    return param;
  }

  boost::property_tree::ptree read_ptree(const char* name) {
    boost::property_tree::ptree param;

    try {
      // Read the entire file  
      std::ifstream file(name);
      if (!file) {
        throw std::runtime_error("Could not open file");
      }

      std::string file_contents((std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());

      // Remove comments  
      std::string cleaned_json = remove_json_comments(file_contents);

      // Parse the cleaned JSON  
      std::istringstream json_stream(cleaned_json);
      boost::property_tree::read_json(json_stream,param);
    }
    catch (const boost::property_tree::json_parser_error& e) {
      std::cerr << "Error reading JSON file: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << std::endl;
    }

    return param;
  }

}