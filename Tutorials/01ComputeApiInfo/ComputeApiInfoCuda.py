import pyktt as ktt

def main():
    # Create new tuner which uses CUDA as compute API.
    tuner = ktt.Tuner(0, 0, ktt.ComputeApi.CUDA)

    # Print information about platforms and devices to standard output.
    platforms = tuner.GetPlatformInfo()

    for i in range(len(platforms)):
        print(platforms[i])
        devices = tuner.GetDeviceInfo(i);

        for device in devices:
            print(device)

if __name__ == "__main__":
    main()
