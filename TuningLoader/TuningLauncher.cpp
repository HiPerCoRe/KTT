#include <iostream>
#include <map>
#include <regex>
#include <string>

#include <TuningLoader.h>

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Path to the tuning file must be specified as a command line parameter." << std::endl;
        return 1;
    }

    const std::string path = argv[1];

    try
    {
        std::map<std::string, std::string> parameters;

        for (size_t i = 2; i < static_cast<size_t>(argc); ++i)
        {
            const std::string token = argv[i];
            std::vector<std::string> parameterTokens;

            std::regex separator("=");
            std::sregex_token_iterator iterator(token.begin(), token.end(), separator, -1);
            std::copy(iterator, std::sregex_token_iterator(), std::back_inserter(parameterTokens));

            parameters[parameterTokens[0]] = parameterTokens[1];
        }

        ktt::TuningLoader loader;
        loader.LoadTuningFile(path, parameters);
        loader.ExecuteCommands();
    }
    catch (const std::exception& exception)
    {
        std::cerr << "Error: " << exception.what() << std::endl;
        return 1;
    }

    return 0;
}
