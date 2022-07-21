#include <iostream>
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
        ktt::TuningLoader loader;
        loader.LoadTuningFile(path);
        loader.ExecuteCommands();
    }
    catch (const std::exception& exception)
    {
        std::cerr << "Error: " << exception.what() << std::endl;
        return 1;
    }

    return 0;
}
