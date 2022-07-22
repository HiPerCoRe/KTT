import sys

def main():
    if len(sys.argv) < 2 :
        print("Error: Python file to stringify must be given as a command line argument")
        exit(1)
    
    file = open(sys.argv[1], "r")
    lines = file.read().splitlines()
    output = open("StringifyOutput.h", "w")
    output.write("std::string("") +\n")
    
    for line in lines:
        modifiedLine = line.replace('"', '\\"')
        output.write("\"" + modifiedLine + "\\n\" +\n")
    
    output.write("\"\";")
    file.close()
    output.close()
    
if __name__ == "__main__":
    main()
    